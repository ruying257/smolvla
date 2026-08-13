"""验证本机 SmolVLA 模型评测入口、配置和短闭环产物。"""

from __future__ import annotations

import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import numpy as np

from evaluate.common import convert_policy_action, find_pretrained_model, load_yaml_config
from evaluate.rollout import build_parser, build_prompt, run_single_rollout, write_results


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def workspace_temp_dir() -> Iterator[Path]:
    """在项目内创建本测试独占的临时目录。

    Yields:
        测试完成后自动删除的临时目录。
    """
    path = PROJECT_ROOT / f".evaluate-test-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class EvaluationContractTests(unittest.TestCase):
    """验证评测配置、checkpoint、指令和动作契约。"""

    def test_default_config_is_local_evaluation_config(self) -> None:
        """默认入口应读取 configs/eval.yaml 并使用 CUDA。"""
        args = build_parser().parse_args(["--checkpoint", "checkpoint"])
        self.assertEqual(args.config, PROJECT_ROOT / "configs" / "eval.yaml")
        self.assertEqual(args.device, "cuda")

    def test_config_combination_counts(self) -> None:
        """基础配置应有 4 组，标准配置应有 80 组。"""
        for filename, expected in (("eval.yaml", 4), ("eval_standard.yaml", 80)):
            evaluation = load_yaml_config(PROJECT_ROOT / "configs" / filename)["evaluation"]
            count = (
                len(evaluation["scene_seeds"])
                * len(evaluation["task_ids"])
                * len(evaluation["prompt_types"])
            )
            self.assertEqual(count, expected)

    def test_checkpoint_locator_accepts_only_complete_model(self) -> None:
        """只有配置、权重和处理器齐全的目录才能用于评测。"""
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

    def test_action_is_clipped_and_invalid_action_is_rejected(self) -> None:
        """动作应限制在控制范围内，并拒绝错误 shape 和非有限数。"""
        ranges = np.asarray([[-1.0, 1.0]] * 6, dtype=np.float64)
        safe = convert_policy_action(np.asarray([[2.0, 0, 0, 0, 0, 0, -0.2]]), ranges)
        self.assertTrue(safe.clipped)
        np.testing.assert_allclose(safe.command, [1.0, 0, 0, 0, 0, 0, 0])
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


class LocalRolloutTests(unittest.TestCase):
    """在 Windows 本机使用假策略验证短闭环产物。"""

    def test_fake_policy_writes_csv_summary_and_video(self) -> None:
        """短 rollout 必须写出 seed、失败分类、CSV、JSON 和非空视频。"""
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
            self.assertGreater(Path(result.video_path).stat().st_size, 0)
            write_results(output, [result])
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["rollouts"], 1)
            self.assertTrue((output / "rollouts.csv").is_file())


if __name__ == "__main__":
    unittest.main()
