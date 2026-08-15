"""Grounding v2矩阵计划、恢复、配对和蒙太奇回归测试。"""

from __future__ import annotations

import json
import re
import shlex
import shutil
import unittest
from collections import Counter
from pathlib import Path

import av
import numpy as np
import yaml

from collector.build_review_montages import _task_cell, _write_h264
from collector.collect_matrix import _enforce_scene_pairing
from collector.collection_plan import (
    GROUNDING_DATASET_VERSION,
    GROUNDING_REPO_ID,
    PROGRESS_FILENAME,
    build_plan,
    initialize_workspace,
    load_config,
    load_initial_reference,
    load_progress,
    plan_for_mode,
    save_initial_reference,
    validate_frame_count,
    validate_shard_record,
)
from collector.dataset_io import (
    DATASET_REPO_ID,
    DATASET_VERSION,
    LeRobotEpisodeWriter,
    _accessible_mkdtemp,
    configure_hf_datasets_cache,
)
from sim.environment import SceneSnapshot


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "collect_grounding_v2.yaml"


class GroundingPlanTest(unittest.TestCase):
    """验证20×4矩阵、canonical文本和Latin square位置均衡。"""

    @classmethod
    def setUpClass(cls) -> None:
        """只加载一次锁定配置。"""
        cls.config = load_config(CONFIG_PATH)
        cls.plan = build_plan(cls.config)

    def test_matrix_has_80_unique_complete_keys(self) -> None:
        """每个scene必须恰好覆盖四任务且总计80个唯一键。"""
        self.assertEqual(len(self.plan), 80)
        self.assertEqual(len({item.queue_key for item in self.plan}), 80)
        counts = Counter(item.scene_seed for item in self.plan)
        self.assertEqual(set(counts.values()), {4})
        self.assertEqual(len(counts), 20)

    def test_canonical_prompts_match_character_for_character(self) -> None:
        """80条计划不得出现synonym或unseen措辞。"""
        expected = {
            "red_on_blue": "Put the red cube on the blue pad.",
            "green_on_blue": "Put the green cube on the blue pad.",
            "red_on_yellow": "Put the red cube on the yellow pad.",
            "green_on_yellow": "Put the green cube on the yellow pad.",
        }
        for item in self.plan:
            self.assertEqual(item.prompt_mode, "canonical")
            self.assertEqual(item.prompt, expected[item.task_id])

    def test_latin_square_balances_every_collection_position(self) -> None:
        """每个任务在第1至4采集位都必须恰好出现5次。"""
        counts = Counter((item.task_id, item.collection_position) for item in self.plan)
        self.assertEqual(set(counts.values()), {5})
        self.assertEqual(len(counts), 16)

    def test_pilot_contains_only_first_two_scenes(self) -> None:
        """pilot必须严格为seed 210、212的8条组合。"""
        pilot = plan_for_mode(self.config, pilot=True)
        self.assertEqual(len(pilot), 8)
        self.assertEqual(list(dict.fromkeys(item.scene_seed for item in pilot)), [210, 212])

    def test_frame_limit_rejects_before_save(self) -> None:
        """空episode和第401帧必须在视频编码前被拒绝。"""
        validate_frame_count(1, 400)
        validate_frame_count(400, 400)
        with self.assertRaises(ValueError):
            validate_frame_count(0, 400)
        with self.assertRaises(ValueError):
            validate_frame_count(401, 400)


class GroundingResumeTest(unittest.TestCase):
    """验证非空拒绝、配置漂移和损坏进度的严格恢复。"""

    def setUp(self) -> None:
        """为每个测试创建独立中文路径工作区和配置。"""
        self.temporary_root = Path(_accessible_mkdtemp(dir=PROJECT_ROOT)) / "恢复测试"
        self.temporary_root.mkdir(parents=True)
        raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        raw["dataset"]["root"] = str(self.temporary_root / "grounding_v2")
        self.config_path = self.temporary_root / "config.yaml"
        self.config_path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
        self.config = load_config(self.config_path)

    def tearDown(self) -> None:
        """删除测试临时目录。"""
        shutil.rmtree(self.temporary_root.parent, ignore_errors=True)

    def test_nonempty_workspace_requires_explicit_resume(self) -> None:
        """首次初始化后不带resume再次启动必须失败。"""
        initialize_workspace(self.config, resume=False)
        with self.assertRaises(FileExistsError):
            initialize_workspace(self.config, resume=False)
        resumed = initialize_workspace(self.config, resume=True)
        self.assertEqual(resumed["completed"], {})

    def test_corrupt_progress_is_rejected(self) -> None:
        """无法解析的progress不得被当成空进度继续采集。"""
        initialize_workspace(self.config, resume=False)
        (self.config.work_root / PROGRESS_FILENAME).write_text("{broken", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_progress(self.config)
        with self.assertRaises(ValueError):
            initialize_workspace(self.config, resume=True)

    def test_config_prompt_drift_is_rejected(self) -> None:
        """canonical文本有一个字符变化也必须在启动前失败。"""
        raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        raw["canonical_prompts"]["red_on_blue"] = "Move the red cube."
        drifted = self.temporary_root / "drifted.yaml"
        drifted.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_config(drifted)

    def test_missing_shard_files_are_not_skippable(self) -> None:
        """契约、Parquet或视频缺失的完成键不得被resume跳过。"""
        initialize_workspace(self.config, resume=False)
        item = build_plan(self.config)[0]
        record = {"shard_name": item.shard_name, "frame_count": 10}
        errors = validate_shard_record(self.config, item, record)
        self.assertIn("contract", errors)
        self.assertIn("parquet", errors)

    def test_lossless_scene_reference_round_trip(self) -> None:
        """同scene四任务应复用可跨进程读取的无损初始观测。"""
        initialize_workspace(self.config, resume=False)
        state = np.arange(7, dtype=np.float32)
        agent = np.full((256, 256, 3), 23, dtype=np.uint8)
        wrist = np.full((256, 256, 3), 231, dtype=np.uint8)
        poses = np.arange(14, dtype=np.float64).reshape(2, 7)
        save_initial_reference(self.config, 210, state, agent, wrist, poses)
        reference = load_initial_reference(self.config, 210)
        self.assertTrue(np.array_equal(reference["state"], state))
        self.assertTrue(np.array_equal(reference["agent"], agent))
        self.assertTrue(np.array_equal(reference["wrist"], wrist))
        self.assertTrue(np.array_equal(reference["cube_initial_poses"], poses))
        with self.assertRaises(FileExistsError):
            save_initial_reference(self.config, 210, state, agent, wrist, poses)


class GroundingPairingAndMontageTest(unittest.TestCase):
    """验证同scene初始条件硬约束及2×2视频产物。"""

    def test_scene_pairing_accepts_equal_and_rejects_changed_hash(self) -> None:
        """同scene任一路初始原始图像不同都必须在写入前停止。"""
        config = load_config(CONFIG_PATH)
        item = build_plan(config)[1]
        snapshot = SceneSnapshot(
            scene_seed=item.scene_seed,
            cube_initial_poses=np.arange(14, dtype=np.float64).reshape(2, 7),
            pad_positions=np.zeros((2, 3), dtype=np.float64),
        )
        hashes = {
            "initial_robot_state_sha256": "state",
            "initial_agent_raw_sha256": "agent",
            "initial_wrist_raw_sha256": "wrist",
        }
        record = {"scene_seed": item.scene_seed, **hashes, "cube_initial_poses": snapshot.cube_initial_poses.tolist()}
        _enforce_scene_pairing(item, snapshot, hashes, {"previous": record})
        changed = dict(hashes)
        changed["initial_agent_raw_sha256"] = "different"
        with self.assertRaises(RuntimeError):
            _enforce_scene_pairing(item, snapshot, changed, {"previous": record})

    def test_montage_cell_layout_and_nonempty_h264(self) -> None:
        """四个256单元可组成512布局，编码视频必须至少解码一帧。"""
        temporary = Path(_accessible_mkdtemp(dir=PROJECT_ROOT))
        try:
            rgb = np.full((256, 256, 3), 96, dtype=np.uint8)
            cell = _task_cell(
                rgb, rgb, 210, "red_on_blue", "Put the red cube on the blue pad.", 0, 1,
            )
            self.assertEqual(cell.shape, (256, 256, 3))
            montage = np.concatenate([
                np.concatenate([cell, cell], axis=1),
                np.concatenate([cell, cell], axis=1),
            ], axis=0)
            self.assertEqual(montage.shape, (512, 512, 3))
            output = temporary / "review.mp4"
            _write_h264(output, [montage, montage], fps=20)
            with av.open(str(output)) as container:
                frames = list(container.decode(video=0))
            self.assertEqual(len(frames), 2)
            self.assertEqual((frames[0].width, frames[0].height), (512, 512))
        finally:
            shutil.rmtree(temporary, ignore_errors=True)


class GroundingCompatibilityTest(unittest.TestCase):
    """验证v2独立身份及操作文档中的PowerShell命令。"""

    def test_v1_contract_identity_is_unchanged(self) -> None:
        """新增v2不得修改旧v1的repo_id和dataset_version常量。"""
        self.assertEqual(DATASET_REPO_ID, "smolvla_ur10e")
        self.assertEqual(DATASET_VERSION, "smolvla_ur10e_v1")
        self.assertEqual(GROUNDING_REPO_ID, "smolvla_ur10e_grounding_v2")
        self.assertEqual(GROUNDING_DATASET_VERSION, "smolvla_ur10e_grounding_v2")

    def test_documented_powershell_commands_are_tokenizable(self) -> None:
        """文档中的Python代码块必须能解析为模块命令。"""
        document = (PROJECT_ROOT / "Grounding_v2数据采集与验收.md").read_text(encoding="utf-8")
        blocks = re.findall(r"```powershell\n(.*?)```", document, flags=re.DOTALL)
        self.assertGreaterEqual(len(blocks), 8)
        python_commands = 0
        for block in blocks:
            command = block.replace("`\n", " ").strip()
            tokens = shlex.split(command, posix=False)
            self.assertTrue(tokens)
            if tokens[0].lower() == "python":
                python_commands += 1
                self.assertGreaterEqual(len(tokens), 3)
                self.assertEqual(tokens[1:3], ["-m", tokens[2]])
        self.assertGreaterEqual(python_commands, 7)

    def test_v2_single_episode_shard_round_trip(self) -> None:
        """v2扩展契约应能真实写出并重新验签Parquet与双路视频。"""
        temporary = Path(_accessible_mkdtemp(dir=PROJECT_ROOT))
        try:
            raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
            raw["dataset"]["root"] = str(temporary / "final")
            config_path = temporary / "config.yaml"
            config_path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
            config = load_config(config_path)
            item = build_plan(config)[0]
            shard = config.shard_root / item.shard_name
            writer = LeRobotEpisodeWriter(
                shard,
                dataset_version=config.dataset_version,
                repo_id=config.repo_id,
                contract_extras={
                    "config_sha256": config.sha256,
                    "queue_key": item.queue_key,
                    "prompt_mode": "canonical",
                },
            )
            snapshot = SceneSnapshot(
                scene_seed=item.scene_seed,
                cube_initial_poses=np.arange(14, dtype=np.float64).reshape(2, 7),
                pad_positions=np.zeros((2, 3), dtype=np.float64),
            )
            images = {
                "agent": np.full((256, 256, 3), 40, dtype=np.uint8),
                "wrist": np.full((256, 256, 3), 180, dtype=np.uint8),
            }
            state = np.zeros(7, dtype=np.float32)
            action = np.zeros(7, dtype=np.float32)
            for _ in range(2):
                writer.add_frame(images, state, action, snapshot, item.prompt)
            writer.save_episode(item.task_id, "canonical", item.prompt, snapshot, 2)
            writer.close()
            record = {"shard_name": item.shard_name, "frame_count": 2}
            self.assertEqual(validate_shard_record(config, item, record), [])
            cache_root = PROJECT_ROOT / ".hf-gv2-test-cache"
            configure_hf_datasets_cache(cache_root)
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            dataset = LeRobotDataset(config.repo_id, root=shard, video_backend="pyav")
            sample = dataset[0]
            self.assertEqual(tuple(sample["observation.state"].shape), (7,))
            self.assertEqual(tuple(sample["observation.images.agent"].shape), (3, 256, 256))
            shutil.rmtree(cache_root, ignore_errors=True)
        finally:
            shutil.rmtree(PROJECT_ROOT / ".hf-gv2-test-cache", ignore_errors=True)
            shutil.rmtree(temporary, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
