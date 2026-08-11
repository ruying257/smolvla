"""验证 P4 云端命令、动作适配、结果文件和无 GPU 短闭环。"""

from __future__ import annotations

import json
import shutil
import sys
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import numpy as np

from cloud.common import convert_policy_action, find_pretrained_model, load_yaml_config
from cloud.rollout import build_prompt, run_single_rollout, write_results
from cloud.train import build_parser as build_train_parser, build_train_command


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def workspace_temp_dir() -> Iterator[Path]:
    """在项目内创建权限稳定的测试临时目录。

    Yields:
        本测试独占的临时目录。
    """
    path = PROJECT_ROOT / f".cloud-test-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class CloudTrainingCommandTests(unittest.TestCase):
    """验证训练配置被稳定转换为 LeRobot CLI。"""

    def test_smoke_command_locks_dataset_features_and_single_step(self) -> None:
        """smoke 模式必须推断七维特征并只训练一步。"""
        config = load_yaml_config(PROJECT_ROOT / "configs" / "cloud_train.yaml")
        command = build_train_command(
            config,
            Path("/srv/smolvla-data/smolvla_ur10e"),
            Path("/srv/smolvla/outputs/smoke/train"),
            "smoke",
            smoke=True,
        )
        self.assertIn("--policy.input_features=null", command)
        self.assertIn("--policy.output_features=null", command)
        self.assertIn("--policy.empty_cameras=1", command)
        self.assertIn("--steps=1", command)
        self.assertIn("--batch_size=1", command)
        self.assertIn("--policy.use_amp=true", command)
        self.assertIn("--policy.push_to_hub=false", command)

    def test_default_dataset_is_inside_project(self) -> None:
        """未显式传参时应读取项目内的 smolvla-data 目录。"""
        args = build_train_parser().parse_args([])
        self.assertEqual(
            args.dataset_root,
            PROJECT_ROOT / "smolvla-data" / "smolvla_ur10e",
        )

    def test_checkpoint_locator_accepts_only_complete_pretrained_model(self) -> None:
        """缺少处理器的单独权重文件不得被当作可评测 checkpoint。"""
        with workspace_temp_dir() as output:
            model = output / "checkpoints" / "000001" / "pretrained_model"
            model.mkdir(parents=True)
            for filename in (
                "config.json",
                "model.safetensors",
                "policy_preprocessor.json",
                "policy_postprocessor.json",
            ):
                (model / filename).write_bytes(b"test")
            self.assertEqual(find_pretrained_model(output), model.resolve())


class CloudActionTests(unittest.TestCase):
    """验证策略动作的七维契约和安全限位。"""

    def test_action_is_clipped_to_arm_and_gripper_ranges(self) -> None:
        """越界动作应被裁剪并显式记录。"""
        ranges = np.asarray([[-1.0, 1.0]] * 6, dtype=np.float64)
        safe = convert_policy_action(np.asarray([[2.0, 0, 0, 0, 0, 0, -0.2]]), ranges)
        self.assertTrue(safe.clipped)
        np.testing.assert_allclose(safe.command, [1.0, 0, 0, 0, 0, 0, 0])

    def test_non_finite_or_wrong_dimension_is_rejected(self) -> None:
        """错误 shape 和非有限数不得进入 MuJoCo。"""
        ranges = np.asarray([[-1.0, 1.0]] * 6, dtype=np.float64)
        with self.assertRaises(ValueError):
            convert_policy_action(np.zeros(6), ranges)
        with self.assertRaises(ValueError):
            convert_policy_action(np.asarray([0, 0, 0, 0, 0, 0, np.nan]), ranges)

    def test_prompt_types_are_separate(self) -> None:
        """canonical 和未见措辞必须保持独立。"""
        self.assertEqual(build_prompt("red_on_blue", "canonical"), "Put the red cube on the blue pad.")
        self.assertEqual(build_prompt("red_on_blue", "unseen"), "Move the red cube to the blue pad.")


class FakePolicy:
    """返回当前关节状态的无模型测试策略。"""

    def __init__(self) -> None:
        self.config = SimpleNamespace(use_amp=False)

    def reset(self) -> None:
        """假策略没有需要清理的缓存。"""

    def select_action(self, observation: dict[str, object]) -> object:
        """保持当前关节目标并打开夹爪。"""
        return observation["observation.state"]


class CloudRolloutTests(unittest.TestCase):
    """使用假策略验证无 GPU 的短闭环产物。"""

    @unittest.skipUnless(sys.platform.startswith("linux"), "完整 headless smoke 仅在 Ubuntu 云端运行")
    def test_fake_policy_writes_csv_summary_and_video(self) -> None:
        """短 rollout 必须保留 seed、任务、失败分类和非空视频。"""
        try:
            import imageio  # noqa: F401
            import mujoco  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"本机缺少短闭环依赖: {exc}")

        with workspace_temp_dir() as output:
            result = run_single_rollout(
                FakePolicy(),
                lambda value: value,
                lambda value: value,
                scene_seed=123,
                task_id="red_on_blue",
                prompt_type="canonical",
                output_dir=output,
                fps=20,
                max_steps=2,
                device="cpu",
            )
            self.assertEqual(result.scene_seed, 123)
            self.assertEqual(result.failure_mode, "timeout")
            self.assertEqual(result.error, "")
            self.assertTrue(Path(result.video_path).is_file())
            self.assertGreater(Path(result.video_path).stat().st_size, 0)
            write_results(output, [result])
            self.assertTrue((output / "rollouts.csv").is_file())
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["rollouts"], 1)


if __name__ == "__main__":
    unittest.main()
