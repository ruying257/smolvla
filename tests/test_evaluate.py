"""验证本机SmolVLA评测的矩阵、复现、恢复、统计和视频产物。"""

from __future__ import annotations

import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import numpy as np

from evaluate.common import convert_policy_action, find_pretrained_model, load_yaml_config
from evaluate.rollout import (
    RolloutResult,
    RolloutSpec,
    append_jsonl,
    bootstrap_success_ci,
    build_parser,
    build_prompt,
    build_specs,
    load_jsonl,
    prepare_run,
    retain_videos,
    resolve_execution_horizon,
    run_single_rollout,
    set_policy_seed,
    source_sha256,
    summarize_action_clipping,
    summarize_results,
    write_results,
)


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


def fake_result(
    root: Path,
    scene: int,
    policy_seed: int,
    task: str,
    prompt: str,
    success: bool,
    failure_mode: str | None = None,
) -> RolloutResult:
    """构造带非空视频的统计测试结果。"""
    spec = RolloutSpec(scene, task, prompt, policy_seed)
    video = root / f"{scene}_{policy_seed}_{task}_{prompt}.mp4"
    video.write_bytes(b"video")
    return RolloutResult(
        rollout_key=spec.key,
        scene_seed=scene,
        policy_seed=policy_seed,
        task_id=task,
        task=build_prompt(task, prompt),
        prompt_type=prompt,
        success=success,
        failure_mode=failure_mode or ("success" if success else "timeout"),
        steps=200 if success else 400,
        elapsed_seconds=1.0,
        latency_mean_ms=10.0,
        latency_p95_ms=12.0,
        clipped_action_steps=20,
        clipped_action_rate=0.1 if success else 0.05,
        action_trace_path="",
        checkpoint_sha256="abc",
        video_path=str(video),
        video_retained=True,
        error="",
        completed_at="2026-08-13T00:00:00+00:00",
    )


class EvaluationContractTests(unittest.TestCase):
    """验证评测配置、checkpoint、指令、动作和随机性契约。"""

    def test_default_config_is_local_evaluation_config(self) -> None:
        """默认入口应读取configs/eval.yaml并使用CUDA。"""
        args = build_parser().parse_args(["--checkpoint", "checkpoint"])
        self.assertEqual(args.config, PROJECT_ROOT / "configs" / "eval.yaml")
        self.assertEqual(args.device, "cuda")
        self.assertFalse(args.resume)
        self.assertIsNone(args.execution_horizon)

    def test_source_hash_uses_current_collector_task_definition(self) -> None:
        """采集目录重组后，评测源码哈希仍应读取实际任务定义。"""
        first = source_sha256()
        second = source_sha256()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_execution_horizon_uses_cli_override_and_validates_chunk_size(self) -> None:
        """执行步数应支持命令行覆盖且不能超过模型chunk长度。"""
        with workspace_temp_dir() as output:
            (output / "config.json").write_text(
                json.dumps({"chunk_size": 50, "n_action_steps": 50}),
                encoding="utf-8",
            )
            self.assertEqual(resolve_execution_horizon(output, {}, 10), (10, 50))
            self.assertEqual(resolve_execution_horizon(output, {"execution_horizon": 20}, None), (20, 50))
            with self.assertRaisesRegex(ValueError, "chunk_size"):
                resolve_execution_horizon(output, {}, 51)

    def test_config_combination_counts(self) -> None:
        """基础、seen场景和正式配置应分别有12、24和120组。"""
        for filename, expected in (("eval.yaml", 12), ("eval_seen.yaml", 24), ("eval_standard.yaml", 120)):
            evaluation = load_yaml_config(PROJECT_ROOT / "configs" / filename)["evaluation"]
            specs = build_specs(evaluation)
            self.assertEqual(len(specs), expected)
            self.assertEqual(len({spec.key for spec in specs}), expected)
        standard = load_yaml_config(PROJECT_ROOT / "configs" / "eval_standard.yaml")["evaluation"]
        self.assertEqual(standard["max_steps"], 400)
        self.assertEqual(standard["prompt_types"], ["canonical", "synonym", "unseen"])
        self.assertEqual(standard["policy_seeds"], [20260])
        seen = load_yaml_config(PROJECT_ROOT / "configs" / "eval_seen.yaml")["evaluation"]
        self.assertEqual(seen["scene_seeds"], [0, 1, 2, 3, 4, 5])
        self.assertEqual(seen["prompt_types"], ["canonical"])
        self.assertEqual(seen["policy_seeds"], [20260])

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
        """动作应限制在控制范围内，并拒绝错误shape和非有限数。"""
        ranges = np.asarray([[-1.0, 1.0]] * 6, dtype=np.float64)
        safe = convert_policy_action(np.asarray([[2.0, 0, 0, 0, 0, 0, -0.2]]), ranges)
        self.assertTrue(safe.clipped)
        np.testing.assert_allclose(safe.command, [1.0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(safe.clipped_mask, [True, False, False, False, False, False, True])
        np.testing.assert_allclose(safe.clip_amount, [1.0, 0, 0, 0, 0, 0, -0.2])
        with self.assertRaises(ValueError):
            convert_policy_action(np.zeros(6), ranges)
        with self.assertRaises(ValueError):
            convert_policy_action(np.asarray([0, 0, 0, 0, 0, 0, np.nan]), ranges)

    def test_all_prompt_types_are_separate(self) -> None:
        """canonical、synonym和unseen文本必须保持独立。"""
        self.assertEqual(build_prompt("red_on_blue", "canonical"), "Put the red cube on the blue pad.")
        self.assertEqual(build_prompt("red_on_blue", "synonym"), "Place the red cube onto the blue pad.")
        self.assertEqual(build_prompt("red_on_blue", "unseen"), "Move the red cube to the blue pad.")

    def test_policy_seed_reproduces_random_stream(self) -> None:
        """相同policy seed应复现NumPy和PyTorch随机流。"""
        import torch

        set_policy_seed(20260)
        first = (np.random.random(3), torch.randn(3))
        set_policy_seed(20260)
        second = (np.random.random(3), torch.randn(3))
        np.testing.assert_allclose(first[0], second[0])
        self.assertTrue(torch.equal(first[1], second[1]))


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
    """在Windows本机使用假策略验证短闭环产物。"""

    def test_fake_policy_writes_extended_result_and_video(self) -> None:
        """短rollout必须写出随机种子、裁剪率、结果文件和非空视频。"""
        with workspace_temp_dir() as output:
            spec = RolloutSpec(123, "red_on_blue", "canonical", 20260)
            result = run_single_rollout(
                FakePolicy(),
                lambda value: value,
                lambda value: value,
                spec=spec,
                output_dir=output,
                fps=20,
                max_steps=2,
                device="cpu",
                checkpoint_sha256="abc",
                execution_horizon=1,
            )
            self.assertEqual(result.rollout_key, spec.key)
            self.assertEqual(result.policy_seed, 20260)
            self.assertEqual(result.failure_mode, "timeout")
            self.assertEqual(result.error, "")
            self.assertGreater(Path(result.video_path).stat().st_size, 0)
            trace_lines = Path(result.action_trace_path).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(trace_lines), 2)
            first_trace = json.loads(trace_lines[0])
            self.assertEqual(first_trace["model_output"], first_trace["physical_action"])
            self.assertEqual(first_trace["physical_action"], first_trace["executed_action"])
            self.assertTrue(first_trace["chunk_start"])
            summary = write_results(output, [result])
            self.assertEqual(summary["rollouts"], 1)
            self.assertEqual(summary["action_clipping"]["trace_steps"], 2)
            self.assertTrue((output / "rollouts.csv").is_file())
            self.assertTrue((output / "action_clipping_summary.json").is_file())
            self.assertTrue((output / "action_clipping_by_dimension.csv").is_file())


class ResumeAndStatisticsTests(unittest.TestCase):
    """验证日志恢复、manifest一致性、统计和视频清理。"""

    def test_jsonl_rejects_corruption_and_duplicates(self) -> None:
        """损坏JSONL和重复实验键必须显式失败。"""
        with workspace_temp_dir() as output:
            result = fake_result(output, 1, 10, "red_on_blue", "canonical", True)
            journal = output / "rollouts.jsonl"
            append_jsonl(journal, result)
            append_jsonl(journal, result)
            with self.assertRaisesRegex(ValueError, "重复实验键"):
                load_jsonl(journal)
            journal.write_text("{broken\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "损坏"):
                load_jsonl(journal)

    def test_resume_rejects_manifest_change_and_retries_exception(self) -> None:
        """续跑应拒绝身份变化并移除可重试控制异常。"""
        with workspace_temp_dir() as output:
            manifest = {"schema_version": 1, "checkpoint_path": "x", "rollout_keys": ["a"]}
            write_path = output / "run_manifest.json"
            write_path.write_text(json.dumps(manifest), encoding="utf-8")
            changed = {**manifest, "checkpoint_path": "y"}
            with self.assertRaisesRegex(ValueError, "manifest"):
                prepare_run(output, changed, resume=True)

            exception = fake_result(
                output, 1, 10, "red_on_blue", "canonical", False, failure_mode="control_exception"
            )
            append_jsonl(output / "rollouts.jsonl", exception)
            self.assertEqual(prepare_run(output, manifest, resume=True), [])
            self.assertEqual(load_jsonl(output / "rollouts.jsonl"), [])

    def test_summary_bootstrap_and_stability(self) -> None:
        """汇总应包含分层置信区间、语言差距和两种子稳定性。"""
        with workspace_temp_dir() as output:
            results = [
                fake_result(output, 1, 10, "red_on_blue", "canonical", True),
                fake_result(output, 1, 11, "red_on_blue", "canonical", True),
                fake_result(output, 1, 10, "red_on_blue", "unseen", False),
                fake_result(output, 1, 11, "red_on_blue", "unseen", True),
                fake_result(output, 2, 10, "red_on_blue", "synonym", False),
                fake_result(output, 2, 11, "red_on_blue", "synonym", False),
            ]
            summary = summarize_results(results)
            self.assertEqual(summary["stability"]["stable_success_2_of_2"], 1)
            self.assertEqual(summary["stability"]["sampling_sensitive_1_of_2"], 1)
            self.assertEqual(summary["stability"]["stable_failure_0_of_2"], 1)
            self.assertEqual(len(bootstrap_success_ci(results, repeats=100)), 2)
            self.assertIn("language_generalization_gap", summary)

    def test_action_clipping_summary_finds_most_frequent_dimension(self) -> None:
        """逐维汇总应识别裁剪次数最多的关节。"""
        with workspace_temp_dir() as output:
            result = fake_result(output, 1, 10, "red_on_blue", "canonical", False)
            result.steps = 2
            trace = output / "trace.jsonl"
            base = {
                "rollout_key": result.rollout_key,
                "action_lower": [-1.0] * 6 + [0.0],
                "action_upper": [1.0] * 7,
            }
            records = [
                {
                    **base,
                    "clipped_mask": [False, False, True, False, False, False, True],
                    "clip_amount": [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.1],
                },
                {
                    **base,
                    "clipped_mask": [False, False, True, False, False, False, False],
                    "clip_amount": [0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0],
                },
            ]
            trace.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            result.action_trace_path = str(trace)
            clipping = summarize_action_clipping([result])
            self.assertEqual(clipping["most_frequently_clipped_dimensions"], ["elbow"])
            self.assertEqual(clipping["clipped_trace_step_rate"], 1.0)
            self.assertEqual(clipping["clipped_action_elements"], 3)

    def test_video_retention_keeps_failures_and_one_success_per_group(self) -> None:
        """每组只保留首条成功视频，所有失败视频必须保留。"""
        with workspace_temp_dir() as output:
            results = [
                fake_result(output, 1, 10, "red_on_blue", "canonical", True),
                fake_result(output, 1, 11, "red_on_blue", "canonical", True),
                fake_result(output, 2, 10, "red_on_blue", "canonical", False),
            ]
            retention = retain_videos(output, results, keep_all=False)
            self.assertEqual(len(retention["retained_rollout_keys"]), 2)
            self.assertEqual(len(retention["removed_rollout_keys"]), 1)
            self.assertTrue(Path(results[0].video_path).is_file())
            self.assertEqual(results[1].video_path, "")
            self.assertTrue(Path(results[2].video_path).is_file())

    def test_report_and_csv_are_written(self) -> None:
        """正式汇总应生成CSV、JSON和Markdown报告。"""
        with workspace_temp_dir() as output:
            result = fake_result(output, 1, 10, "red_on_blue", "canonical", True)
            manifest = {"checkpoint_path": "checkpoint"}
            write_results(output, [result], manifest)
            for filename in ("rollouts.csv", "summary.json", "report.md"):
                self.assertGreater((output / filename).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
