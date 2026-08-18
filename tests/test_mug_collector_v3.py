"""杯子V3队列、schema、恢复、验收和真实seed回归测试。"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import shutil
import unittest
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from collector.v3.collection_plan import (
    MANIFEST_FILENAME,
    build_plan,
    initialize_workspace,
    load_config,
    load_progress,
    plan_for_mode,
    prepare_redo,
    record_completion,
)
from collector.v3.dataset_io import (
    CAMERA_FEATURES,
    MugEpisodeWriter,
    dataset_features,
    validate_episode_shard,
)
from collector.v3.select_seeds import MIN_SELECTED_DISTANCE
from collector.v3.validate_dataset import _has_close_then_release, _validate_records, replay_actions
from sim.mug_environment import MugSceneSnapshot, MugTabletopEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "collect_mug_v3.yaml"
SELECTION_PATH = PROJECT_ROOT / "configs" / "mug_v3_seed_selection.json"


@contextmanager
def _writable_test_directory() -> Iterator[Path]:
    """在项目内创建权限稳定的临时测试目录。

    Windows受管环境可能让 ``tempfile`` 创建当前进程无法继续访问的目录，
    因此测试使用普通目录语义，并在退出时递归清理。

    Yields:
        位于项目根目录且名称唯一的可写临时路径。
    """
    path = PROJECT_ROOT / f".mug-v3-test-{uuid.uuid4().hex}"
    path.mkdir(parents=False, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class _FakeColumn:
    """为验收测试提供最小PyArrow列替身。"""

    def __init__(self, values: object) -> None:
        """保存可由 ``to_pylist`` 返回的值。

        Args:
            values: 列的逐帧Python值。
        """
        self.values = values

    def to_pylist(self) -> object:
        """返回保存的逐帧值。

        Returns:
            构造时传入的值。
        """
        return self.values


class _FakeTable:
    """为V3验收逻辑提供定长向量列替身。"""

    def __init__(self, columns: dict[str, object]) -> None:
        """创建列名到值的内存表。

        Args:
            columns: 每列具有相同帧数的Python值映射。
        """
        self.columns = columns
        self.column_names = list(columns)

    def __getitem__(self, name: str) -> _FakeColumn:
        """按名称返回内存列。

        Args:
            name: feature列名称。

        Returns:
            支持 ``to_pylist`` 的列替身。
        """
        return _FakeColumn(self.columns[name])

    def __len__(self) -> int:
        """返回表的帧数。

        Returns:
            第一列的元素数量。
        """
        return len(next(iter(self.columns.values())))


class _InMemoryDataset:
    """记录MugEpisodeWriter调用但不写视频的测试替身。"""

    def __init__(self) -> None:
        """初始化空帧列表和生命周期标志。"""
        self.frames: list[dict[str, object]] = []
        self.saved = False
        self.finalized = False

    def add_frame(self, frame: dict[str, object]) -> None:
        """保存一帧schema字典。

        Args:
            frame: 写入器已经验证的LeRobot帧。
        """
        self.frames.append(frame)

    def save_episode(self, parallel_encoding: bool = False) -> None:
        """标记一次episode提交。

        Args:
            parallel_encoding: 与真实LeRobot接口兼容的未使用参数。
        """
        del parallel_encoding
        self.saved = True

    def clear_episode_buffer(self, delete_images: bool = True) -> None:
        """模拟清除未确认episode。

        Args:
            delete_images: 与真实LeRobot接口兼容的未使用参数。
        """
        del delete_images
        self.frames.clear()

    def finalize(self) -> None:
        """标记写入器已经关闭。"""
        self.finalized = True


class MugV3PlanAndSchemaTest(unittest.TestCase):
    """验证锁定矩阵、任务顺序、pilot和每帧schema。"""

    @classmethod
    def setUpClass(cls) -> None:
        """加载一次真实锁定配置与计划。"""
        cls.config = load_config(CONFIG_PATH)
        cls.plan = build_plan(cls.config)

    def test_matrix_has_40_unique_paired_keys(self) -> None:
        """20个scene必须各有蓝黄两条且总键数为40。"""
        self.assertEqual(len(self.plan), 40)
        self.assertEqual(len({item.queue_key for item in self.plan}), 40)
        for seed in self.config.scene_seeds:
            tasks = {item.task_id for item in self.plan if item.scene_seed == seed}
            self.assertEqual(tasks, {"mug_on_blue", "mug_on_yellow"})

    def test_task_order_alternates_and_pilot_has_four(self) -> None:
        """偶数scene蓝黄、奇数scene黄蓝，pilot必须恰好四条。"""
        for scene_index, seed in enumerate(self.config.scene_seeds):
            tasks = [item.task_id for item in self.plan if item.scene_seed == seed]
            expected = (
                ["mug_on_blue", "mug_on_yellow"]
                if scene_index % 2 == 0
                else ["mug_on_yellow", "mug_on_blue"]
            )
            self.assertEqual(tasks, expected)
        pilot = plan_for_mode(self.config, pilot=True)
        self.assertEqual(len(pilot), 4)
        self.assertEqual({item.scene_seed for item in pilot}, set(self.config.pilot_scene_seeds))

    def test_schema_and_camera_sources_are_locked(self) -> None:
        """V3必须只含杯子pose及agentview、d435i两路训练视频。"""
        features = dataset_features()
        self.assertEqual(set(features), {
            "observation.images.agent", "observation.images.wrist",
            "observation.state", "action", "scene_seed", "mug_initial_pose",
        })
        self.assertEqual(features["mug_initial_pose"]["shape"], (7,))
        self.assertEqual(CAMERA_FEATURES, {
            "observation.images.agent": "agentview",
            "observation.images.wrist": "d435i_rgb",
        })

    def test_seed_report_covers_grid_and_distance(self) -> None:
        """固化报告必须覆盖20格且实际中心两两至少4厘米。"""
        report = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
        selected = report["selected"]
        self.assertEqual(report["scene_seeds"], list(self.config.scene_seeds))
        self.assertEqual(len({(item["column"], item["row"]) for item in selected}), 20)
        positions = np.asarray([(item["x"], item["y"]) for item in selected])
        minimum = min(
            np.linalg.norm(positions[left] - positions[right])
            for left in range(20) for right in range(left + 1, 20)
        )
        self.assertGreaterEqual(minimum, MIN_SELECTED_DISTANCE)


class MugV3ResumeAndAtomicityTest(unittest.TestCase):
    """验证默认拒绝覆盖、严格resume、原子完成和局部redo。"""

    def test_workspace_requires_resume_and_rejects_manifest_drift(self) -> None:
        """非空工作区必须显式resume且配置漂移必须失败。"""
        base = load_config(CONFIG_PATH)
        with _writable_test_directory() as temporary:
            config = replace(base, root=Path(temporary) / "dataset")
            initialize_workspace(config, resume=False)
            with self.assertRaises(FileExistsError):
                initialize_workspace(config, resume=False)
            initialize_workspace(config, resume=True)
            manifest_path = config.work_root / MANIFEST_FILENAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["fps"] = 10
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                initialize_workspace(config, resume=True)

    def test_completion_is_unique_and_redo_archives_only_one_key(self) -> None:
        """完成键不得重复，redo只归档指定分片并恢复其待采状态。"""
        base = load_config(CONFIG_PATH)
        with _writable_test_directory() as temporary:
            config = replace(base, root=Path(temporary) / "dataset")
            initialize_workspace(config, resume=False)
            item = build_plan(config)[0]
            shard = config.shard_root / item.shard_name
            shard.mkdir(parents=True)
            record = {
                "queue_key": item.queue_key,
                "shard_name": item.shard_name,
                "config_sha256": config.sha256,
            }
            with patch("collector.v3.dataset_io.validate_episode_shard"):
                record_completion(config, item, record)
                with self.assertRaises(ValueError):
                    record_completion(config, item, record)
            prepare_redo(config, item.queue_key)
            self.assertNotIn(item.queue_key, load_progress(config)["completed"])
            self.assertFalse(shard.exists())
            self.assertEqual(len(list((config.work_root / "archived_shards").rglob(item.shard_name))), 1)

    def test_in_memory_writer_enforces_schema_and_lifecycle(self) -> None:
        """测试替身应覆盖add、save、discard和finalize而无需真实录制。"""
        fake = _InMemoryDataset()
        with _writable_test_directory() as temporary:
            writer = MugEpisodeWriter(
                Path(temporary) / "shard",
                contract_extras={"queue_key": "fake"},
                dataset_factory=lambda root, features: fake,
            )
            snapshot = MugSceneSnapshot(
                scene_seed=1,
                mug_initial_pose=np.asarray([0.3, 0.0, 0.82, 1.0, 0.0, 0.0, 0.0]),
                pad_positions=np.asarray([[0.55, -0.22, 0.8005], [0.55, 0.22, 0.8005]]),
            )
            images = {
                "agent": np.zeros((256, 256, 3), dtype=np.uint8),
                "wrist": np.zeros((256, 256, 3), dtype=np.uint8),
            }
            writer.add_frame(images, np.zeros(7, np.float32), np.zeros(7, np.float32), snapshot, "Put the mug on the blue pad.")
            self.assertEqual(len(fake.frames), 1)
            writer.save_episode("mug_on_blue", "Put the mug on the blue pad.", snapshot, 1)
            self.assertTrue(fake.saved)
            self.assertEqual(writer.total_episodes, 1)
            writer.discard_episode()
            self.assertEqual(fake.frames, [])
            writer.close()
            self.assertTrue(fake.finalized)

    def test_real_lerobot_shard_round_trip(self) -> None:
        """真实写出两路视频和Parquet后必须通过分片完整验签。"""
        with _writable_test_directory() as temporary:
            shard = temporary / "real_shard"
            writer = MugEpisodeWriter(
                shard,
                contract_extras={
                    "queue_key": "scene=1|task=mug_on_blue|prompt=canonical",
                    "config_sha256": "test-config",
                },
            )
            snapshot = MugSceneSnapshot(
                scene_seed=1,
                mug_initial_pose=np.asarray([0.3, 0.0, 0.82, 1.0, 0.0, 0.0, 0.0]),
                pad_positions=np.asarray([[0.55, -0.22, 0.8005], [0.55, 0.22, 0.8005]]),
            )
            for frame_index in range(2):
                image = np.full((256, 256, 3), 30 + frame_index * 20, dtype=np.uint8)
                action = np.zeros(7, dtype=np.float32)
                action[6] = float(frame_index == 0)
                writer.add_frame(
                    {"agent": image, "wrist": np.flip(image, axis=1).copy()},
                    np.zeros(7, dtype=np.float32),
                    action,
                    snapshot,
                    "Put the mug on the blue pad.",
                )
            writer.save_episode(
                "mug_on_blue", "Put the mug on the blue pad.", snapshot, frame_count=2,
            )
            writer.close()
            record = {
                "queue_key": "scene=1|task=mug_on_blue|prompt=canonical",
                "config_sha256": "test-config",
                "scene_seed": 1,
                "task_id": "mug_on_blue",
                "frame_count": 2,
                "task": "Put the mug on the blue pad.",
                "mug_initial_pose": snapshot.mug_initial_pose.tolist(),
            }
            validate_episode_shard(shard, record)


class MugV3ValidationTest(unittest.TestCase):
    """使用内存分片替身验证40条分布、配对和动作门禁。"""

    def test_gripper_requires_close_then_release(self) -> None:
        """只有闭合后再次释放的动作序列才能通过专家轨迹门禁。"""
        valid = np.zeros((3, 7), dtype=np.float32)
        valid[1, 6] = 1.0
        self.assertTrue(_has_close_then_release(valid))
        self.assertFalse(_has_close_then_release(np.ones((3, 7), dtype=np.float32)))

    def test_replay_holds_terminal_action_for_success_detection_tick(self) -> None:
        """回放应补齐采集器未写入的末端成功检测周期。"""

        class TerminalTickEnv:
            """仅在两个记录动作后的额外保持周期成功的环境替身。"""

            def __init__(self) -> None:
                """初始化2毫秒仿真步长和动作周期计数。"""
                self.model = SimpleNamespace(opt=SimpleNamespace(timestep=0.002))
                self.applied_ticks = 0

            def reset(self, seed: int) -> None:
                """重置动作周期计数。

                Args:
                    seed: 当前场景seed，本替身不使用其数值。
                """
                del seed
                self.applied_ticks = 0

            def apply_joint_action(self, action: np.ndarray, physics_steps: int) -> None:
                """记录一次20 Hz绝对动作执行。

                Args:
                    action: 当前七维绝对动作。
                    physics_steps: 当前控制周期包含的MuJoCo步数。
                """
                self.assert_action = np.asarray(action)
                if physics_steps != 25:
                    raise AssertionError(f"预期25个物理步，实际{physics_steps}")
                self.applied_ticks += 1

            def evaluate_task(
                self,
                task_id: str,
                elapsed_seconds: float | None = None,
                timeout_seconds: float | None = None,
            ) -> SimpleNamespace:
                """在第三个控制周期模拟严格稳定成功。

                Args:
                    task_id: 当前任务标识。
                    elapsed_seconds: 当前回放时间。
                    timeout_seconds: 当前回放超时阈值。

                Returns:
                    具有成功标志和失败分类的最小评估结果。
                """
                del task_id, elapsed_seconds, timeout_seconds
                success = self.applied_ticks >= 3
                return SimpleNamespace(
                    success=success,
                    failure_mode="success" if success else "in_progress",
                )

        item = build_plan(load_config(CONFIG_PATH))[0]
        actions = np.zeros((2, 7), dtype=np.float32)
        env = TerminalTickEnv()
        success, failure_mode = replay_actions(env, item, actions, fps=20)
        self.assertTrue(success)
        self.assertEqual(failure_mode, "success")
        self.assertEqual(env.applied_ticks, 3)

    def test_full_fake_validation_passes_40_paired_records(self) -> None:
        """内存替身应覆盖40项验收而不要求用户先录制真实数据。"""
        base = load_config(CONFIG_PATH)
        with _writable_test_directory() as temporary:
            config = replace(base, root=Path(temporary) / "dataset")
            config.work_root.mkdir(parents=True)
            with (config.work_root / "review_status.csv").open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["scene_seed", "status"])
                writer.writerows((seed, "pass") for seed in config.scene_seeds)
            plan = build_plan(config)
            records: dict[str, dict[str, object]] = {}
            tables: dict[str, _FakeTable] = {}
            tasks: dict[str, str] = {}
            poses: dict[int, list[float]] = {}
            for item in plan:
                pose = poses.setdefault(
                    item.scene_seed,
                    [0.30 + item.scene_index * 0.001, 0.0, 0.82, 1.0, 0.0, 0.0, 0.0],
                )
                shard_name = item.shard_name
                records[item.queue_key] = {
                    "queue_key": item.queue_key,
                    "shard_name": shard_name,
                    "scene_seed": item.scene_seed,
                    "task": item.prompt,
                    "task_id": item.task_id,
                    "frame_count": 3,
                    "initial_robot_state_sha256": f"state-{item.scene_seed}",
                    "mug_initial_pose": pose,
                }
                actions = np.zeros((3, 7), dtype=np.float32)
                actions[1, 6] = 1.0
                tables[shard_name] = _FakeTable({
                    "observation.state": np.zeros((3, 7), dtype=np.float32).tolist(),
                    "action": actions.tolist(),
                    "scene_seed": [[item.scene_seed]] * 3,
                    "mug_initial_pose": [pose] * 3,
                })
                tasks[shard_name] = item.prompt

            class FakeEnv:
                """为验收提供限位和确定性reset的最小环境。"""

                def __init__(self) -> None:
                    """建立覆盖六关节的宽松actuator限位。"""
                    self.model = SimpleNamespace(
                        actuator_ctrlrange=np.asarray([[-10.0, 10.0]] * 6),
                    )

                def __enter__(self) -> "FakeEnv":
                    """返回上下文中的自身。

                    Returns:
                        当前替身环境。
                    """
                    return self

                def __exit__(self, *args: object) -> None:
                    """退出上下文且不抑制异常。

                    Args:
                        args: 上下文管理器异常参数。
                    """
                    del args

                def reset(self, seed: int) -> MugSceneSnapshot:
                    """返回与记录完全一致的初始杯子位姿。

                    Args:
                        seed: 当前scene seed。

                    Returns:
                        内存构造的杯子快照。
                    """
                    return MugSceneSnapshot(
                        scene_seed=seed,
                        mug_initial_pose=np.asarray(poses[seed], dtype=np.float64),
                        pad_positions=np.zeros((2, 3), dtype=np.float64),
                    )

            progress = {"completed": records}
            with (
                patch("collector.v3.validate_dataset.load_progress", return_value=progress),
                patch("collector.v3.validate_dataset.validate_episode_shard"),
                patch("collector.v3.validate_dataset.read_shard_table", side_effect=lambda path: tables[path.name]),
                patch("collector.v3.validate_dataset.task_texts", side_effect=lambda path: [tasks[path.name]]),
                patch("collector.v3.validate_dataset.MugTabletopEnv", FakeEnv),
                patch("collector.v3.validate_dataset.replay_actions", return_value=(True, "success")),
                patch("collector.v3.validate_dataset._montage_is_valid", return_value=True),
            ):
                report, rows = _validate_records(config, plan)
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["episode_count"], 40)
            self.assertEqual(report["task_counts"], {"mug_on_blue": 20, "mug_on_yellow": 20})
            self.assertEqual(len(rows), 40)


class MugV3RealEnvironmentTest(unittest.TestCase):
    """验证固化seed仍能在真实headless环境中复现。"""

    def test_all_selected_seeds_reproduce_reported_stable_xy(self) -> None:
        """20个seed必须reset成功且实际稳定XY与筛选报告一致。"""
        report = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
        with MugTabletopEnv() as env:
            for selected in report["selected"]:
                snapshot = env.reset(selected["seed"])
                self.assertTrue(np.isfinite(snapshot.mug_initial_pose).all())
                np.testing.assert_allclose(
                    snapshot.mug_initial_pose[:2],
                    [selected["x"], selected["y"]],
                    rtol=0.0,
                    atol=1e-12,
                )

    def test_original_cube_core_hashes_are_unchanged(self) -> None:
        """杯子采集开发不得修改原积木环境、XML和启动入口。"""
        expected = {
            "sim/environment.py": "9146dbaa1a41045d567adba7788ee190e83e2716f028b9001a94ac24d1b0d567",
            "assets/mujoco/scene.xml": "9af63a065a9193b366bdf60adfca5d7aec1e6384ae3b7f03e8c18f01af6e216b",
            "view_scene.py": "78cb1543997555ffba2b00a832daF110efca5a9c08e4fe1492815c148d216840".lower(),
        }
        for relative, digest in expected.items():
            actual = hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(actual, digest, relative)

    def test_all_v3_classes_and_functions_have_chinese_docstrings(self) -> None:
        """新增Python类、函数和方法必须带中文Google风格文档字符串。"""
        for path in (PROJECT_ROOT / "collector" / "v3").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    docstring = ast.get_docstring(node) or ""
                    self.assertTrue(docstring, f"缺少docstring: {path.name}:{node.name}")
                    self.assertRegex(docstring, "[\u4e00-\u9fff]", f"docstring缺少中文: {path.name}:{node.name}")


if __name__ == "__main__":
    unittest.main()
