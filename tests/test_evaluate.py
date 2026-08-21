"""验证本机SmolVLA评测的矩阵、复现、恢复、统计和视频产物。"""

from __future__ import annotations

import csv
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

from evaluate.common import (
    JointMotionLimiter,
    MotionLimits,
    convert_policy_action,
    find_pretrained_model,
    load_motion_limits,
    load_yaml_config,
)
from evaluate.rollout import (
    GripperHysteresisFilter,
    MugStageTracker,
    RolloutResult,
    RolloutSpec,
    append_jsonl,
    bootstrap_success_ci,
    build_parser,
    build_prompt,
    build_specs,
    load_jsonl,
    prepare_failure_update,
    prepare_run,
    retain_videos,
    resolve_execution_horizon,
    resolve_gripper_filter,
    resolve_stage_detection,
    resolve_motion_limiter,
    run_single_rollout,
    select_scene_specs,
    set_policy_seed,
    source_sha256,
    summarize_action_clipping,
    summarize_results,
    write_stage_metrics_outputs,
    write_results,
    update_manifest_identity,
)
from scripts.calibrate_motion_limits import calibrate


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
        self.assertIsNone(args.scene_seed)

    def test_scene_seed_filter_keeps_all_config_dimensions(self) -> None:
        """单场景参数应只收缩场景维度，保留任务、措辞和policy seed。"""
        evaluation = load_yaml_config(PROJECT_ROOT / "configs" / "eval_seen.yaml")["evaluation"]
        selected = select_scene_specs(build_specs(evaluation), 3)
        self.assertEqual(len(selected), 4)
        self.assertEqual({spec.scene_seed for spec in selected}, {3})
        with self.assertRaisesRegex(ValueError, "不在配置场景中"):
            select_scene_specs(build_specs(evaluation), 99999)

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

    def test_green_mug_smoke_config_is_locked_to_four_h25_rollouts(self) -> None:
        """绿白杯实验应只包含两个seen场景和两个canonical任务。"""
        evaluation = load_yaml_config(
            PROJECT_ROOT / "configs" / "eval" / "mug_v1_seen_green_2seeds.yaml"
        )["evaluation"]
        specs = build_specs(evaluation)
        self.assertEqual(len(specs), 4)
        self.assertEqual({spec.scene_seed for spec in specs}, {2291, 6705})
        self.assertEqual(evaluation["appearance_variant"], "green_white")
        self.assertEqual(evaluation["execution_horizon"], 25)

    def test_h20_motion_limiter_configs_share_the_same_matrix(self) -> None:
        """h20基线与限制器必须只在motion_limiter和输出目录上不同。"""
        baseline = load_yaml_config(PROJECT_ROOT / "configs" / "eval" / "mug_v1_seen_h20_baseline.yaml")["evaluation"]
        limited = load_yaml_config(PROJECT_ROOT / "configs" / "eval" / "mug_v1_seen_h20_motion_limiter.yaml")["evaluation"]
        self.assertEqual(len(build_specs(baseline)), 40)
        self.assertEqual(build_specs(baseline), build_specs(limited))
        self.assertEqual(baseline["execution_horizon"], 20)
        self.assertFalse(baseline["motion_limiter"]["enabled"])
        self.assertTrue(limited["motion_limiter"]["enabled"])

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

    def test_motion_limit_file_and_config_are_validated(self) -> None:
        """启用限制器时应锁定20Hz六关节限制文件，并拒绝fps漂移。"""
        with workspace_temp_dir() as output:
            path = output / "limits.json"
            payload = {
                "schema_version": 1,
                "fps": 20,
                "joint_names": ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"],
                "velocity_limits_rad_s": [1.0] * 6,
                "acceleration_limits_rad_s2": [2.0] * 6,
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            limits = load_motion_limits(path, 20)
            np.testing.assert_allclose(limits.velocity_limits_rad_s, np.ones(6))
            with self.assertRaisesRegex(ValueError, "fps"):
                load_motion_limits(path, 10)
            settings = resolve_motion_limiter({"motion_limiter": {"enabled": True, "limits_path": str(path)}}, 20)
            self.assertTrue(settings["enabled"])

    def test_gripper_filter_config_defaults_disable_and_validation(self) -> None:
        """夹爪过滤配置应提供稳定默认值、关闭兼容和严格校验。"""
        self.assertEqual(
            resolve_gripper_filter({}),
            {
                "enabled": True,
                "close_threshold": 0.8,
                "open_threshold": 0.05,
                "close_confirmation_frames": 2,
                "open_confirmation_frames": 3,
            },
        )
        self.assertEqual(
            resolve_gripper_filter({"gripper_filter": {"enabled": False}}),
            {"enabled": False},
        )
        invalid_sections = (
            {"close_threshold": 0.05, "open_threshold": 0.05},
            {"close_threshold": 1.1},
            {"open_threshold": -0.1},
            {"close_confirmation_frames": 0},
            {"open_confirmation_frames": 1.5},
        )
        for section in invalid_sections:
            with self.subTest(section=section), self.assertRaises(ValueError):
                resolve_gripper_filter({"gripper_filter": section})

    def test_gripper_filter_hysteresis_and_confirmation(self) -> None:
        """双阈值应锁存中间值，并按2帧闭合、3帧打开。"""
        gripper_filter = GripperHysteresisFilter()
        first_high = gripper_filter.apply(0.8)
        self.assertEqual(first_high.command, 0.0)
        self.assertEqual(first_high.confirmation_count, 1)
        closed = gripper_filter.apply(0.9)
        self.assertEqual(closed.command, 1.0)
        self.assertEqual(closed.transition, "close")

        middle = gripper_filter.apply(0.4)
        self.assertEqual(middle.command, 1.0)
        self.assertEqual(middle.confirmation_count, 0)
        self.assertEqual(gripper_filter.apply(0.05).command, 1.0)
        self.assertEqual(gripper_filter.apply(0.03).command, 1.0)
        opened = gripper_filter.apply(0.0)
        self.assertEqual(opened.command, 0.0)
        self.assertEqual(opened.transition, "open")

        self.assertEqual(gripper_filter.apply(0.8).command, 0.0)
        self.assertEqual(gripper_filter.apply(1.0).command, 1.0)

    def test_stage_detection_config_defaults_and_validation(self) -> None:
        """阶段检测应默认启用固定阈值，并拒绝非法观测参数。"""
        self.assertEqual(
            resolve_stage_detection({}),
            {
                "enabled": True,
                "gripper_closed_threshold": 0.5,
                "lift_delta_m": 0.015,
            },
        )
        self.assertEqual(
            resolve_stage_detection({"stage_detection": {"enabled": False}}),
            {"enabled": False},
        )
        for section in (
            {"gripper_closed_threshold": -0.1},
            {"gripper_closed_threshold": 1.1},
            {"lift_delta_m": 0.0},
            {"lift_delta_m": "0.015"},
        ):
            with self.subTest(section=section), self.assertRaises(ValueError):
                resolve_stage_detection({"stage_detection": section})

    def test_mug_stage_tracker_reaches_s1_to_s5_in_order(self) -> None:
        """阶段跟踪器应按真实信号依次推进并记录首次命中时刻。"""
        tracker = MugStageTracker(initial_bottom_z=0.8)
        base = {
            "bottom_z": 0.815,
            "center_inside": False,
            "target_inside": False,
            "gripper_released": False,
        }
        s1 = tracker.update(1, 0.05, 0.5, base, False)
        self.assertEqual(s1.sequential_stage, "S1")
        s2 = tracker.update(2, 0.10, 0.5, {**base, "center_inside": True}, False)
        self.assertEqual(s2.sequential_stage, "S2")
        placed = {**base, "bottom_z": 0.8, "center_inside": True, "target_inside": True}
        s3 = tracker.update(3, 0.15, 0.5, placed, False)
        self.assertEqual(s3.sequential_stage, "S3")
        s4 = tracker.update(4, 0.20, 0.0, {**placed, "gripper_released": True}, False)
        self.assertEqual(s4.sequential_stage, "S4")
        s5 = tracker.update(5, 0.25, 0.0, {**placed, "gripper_released": True}, True)
        self.assertEqual(s5.sequential_stage, "S5")
        self.assertEqual(s5.first_reached_step, {"S1": 1, "S2": 2, "S3": 3, "S4": 4, "S5": 5})
        self.assertFalse(s5.order_anomaly)

    def test_mug_stage_tracker_requires_release_while_inside(self) -> None:
        """历史上放入后在目标外松爪不得直接命中S4。"""
        tracker = MugStageTracker(initial_bottom_z=0.8)
        placed = {
            "bottom_z": 0.8,
            "center_inside": True,
            "target_inside": True,
            "gripper_released": False,
        }
        tracker.update(1, 0.05, 0.5, placed, False)
        released_outside = tracker.update(
            2,
            0.10,
            0.0,
            {**placed, "center_inside": False, "target_inside": False, "gripper_released": True},
            False,
        )
        self.assertFalse(released_outside.direct_reached["S4"])

    def test_mug_stage_tracker_records_order_anomaly_without_promotion(self) -> None:
        """直接命中后序阶段时不得补齐前序阶段。"""
        tracker = MugStageTracker(initial_bottom_z=0.8)
        result = tracker.update(
            1,
            0.05,
            0.0,
            {
                "bottom_z": 0.8,
                "center_inside": True,
                "target_inside": True,
                "gripper_released": True,
            },
            False,
        )
        self.assertFalse(result.direct_reached["S1"])
        self.assertFalse(result.direct_reached["S2"])
        self.assertTrue(result.direct_reached["S3"])
        self.assertTrue(result.direct_reached["S4"])
        self.assertEqual(result.sequential_stage, "none")
        self.assertTrue(result.order_anomaly)

    def test_gripper_filter_resets_confirmation_when_sequence_breaks(self) -> None:
        """候选序列被中间值打断后必须重新累计确认帧。"""
        gripper_filter = GripperHysteresisFilter()
        gripper_filter.apply(1.0)
        gripper_filter.apply(1.0)
        self.assertEqual(gripper_filter.apply(0.04).confirmation_count, 1)
        self.assertEqual(gripper_filter.apply(0.2).confirmation_count, 0)
        self.assertEqual(gripper_filter.apply(0.04).command, 1.0)
        self.assertEqual(gripper_filter.apply(0.03).command, 1.0)
        self.assertEqual(gripper_filter.apply(0.02).command, 0.0)

    def test_gripper_filter_blocks_scene_2277_spurious_release(self) -> None:
        """本例277至314步仅一个低值帧，不得解除闭爪锁存。"""
        gripper_filter = GripperHysteresisFilter()
        gripper_filter.apply(0.99)
        gripper_filter.apply(0.99)
        failure_commands = [
            0.082, 0.746, 0.235, 0.763, 0.087, 0.223, 0.100, 0.119,
            0.099, 0.095, 0.098, 0.095, 0.096, 0.105, 0.101, 0.108,
            0.114, 0.120, 0.120, 0.112, 0.100, 0.099, 0.094, 0.092,
            0.071, 0.064, 0.132, 0.274, 0.090, 0.063, 0.054, 0.043,
            0.054, 0.062, 0.305, 0.145, 0.189, 0.104,
        ]
        filtered = [gripper_filter.apply(value).command for value in failure_commands]
        self.assertEqual(filtered, [1.0] * len(failure_commands))
        self.assertEqual(gripper_filter.apply(0.04).command, 1.0)
        self.assertEqual(gripper_filter.apply(0.03).command, 1.0)
        self.assertEqual(gripper_filter.apply(0.02).command, 0.0)

    def test_expert_calibration_keeps_episode_boundaries_separate(self) -> None:
        """专家动作标定不得把两个episode首尾拼接为一个大跳变。"""
        trajectories = [
            np.tile(np.asarray([[0.0], [0.1], [0.4]]), (1, 6)),
            np.tile(np.asarray([[10.0], [10.05], [10.2]]), (1, 6)),
        ]
        velocity, acceleration = calibrate(trajectories, fps=10, quantile=0.5, margin=1.0)
        self.assertTrue(np.all(velocity < 3.0))
        self.assertTrue(np.all(acceleration > 0.0))

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

    def test_prepare_failure_update_accepts_legacy_results_and_rejects_mismatch(self) -> None:
        """失败项更新应接受无manifest旧目录，但拒绝实验键或权重不一致。"""
        with workspace_temp_dir() as output:
            specs = [
                RolloutSpec(1, "red_on_blue", "canonical", 20260),
                RolloutSpec(2, "red_on_blue", "canonical", 20260),
            ]
            first = fake_result(output, 1, 20260, "red_on_blue", "canonical", True)
            second = fake_result(output, 2, 20260, "red_on_blue", "canonical", False)
            first.checkpoint_sha256 = "checkpoint-a"
            second.checkpoint_sha256 = "checkpoint-a"
            append_jsonl(output / "rollouts.jsonl", first)
            append_jsonl(output / "rollouts.jsonl", second)
            manifest = {"checkpoint_sha256": "checkpoint-a"}
            results, failures, legacy = prepare_failure_update(
                output,
                manifest,
                specs,
                "checkpoint-a",
            )
            self.assertEqual(len(results), 2)
            self.assertEqual([result.rollout_key for result in failures], [second.rollout_key])
            self.assertTrue(legacy)
            with self.assertRaisesRegex(ValueError, "checkpoint哈希"):
                prepare_failure_update(output, manifest, specs, "checkpoint-b")
            with self.assertRaisesRegex(ValueError, "实验键不一致"):
                prepare_failure_update(output, manifest, specs[:1], "checkpoint-a")

    def test_failure_update_treats_missing_motion_limiter_as_disabled(self) -> None:
        """旧manifest缺少motion_limiter时应等同于当前默认未启用状态。"""
        legacy = update_manifest_identity({"motion_limiter": None})
        current = update_manifest_identity({"motion_limiter": {"enabled": False}})
        self.assertEqual(legacy, current)

    def test_failure_update_treats_missing_gripper_filter_as_disabled(self) -> None:
        """旧manifest缺少夹爪过滤字段时应解释为连续值透传。"""
        legacy = update_manifest_identity({"gripper_filter": None})
        current = update_manifest_identity({"gripper_filter": {"enabled": False}})
        self.assertEqual(legacy, current)


class FakePolicy:
    """返回当前关节状态的无模型测试策略。"""

    def __init__(self) -> None:
        self.config = SimpleNamespace(use_amp=False)

    def reset(self) -> None:
        """假策略没有需要清理的缓存。"""

    def select_action(self, observation: dict[str, object]) -> object:
        """保持当前关节目标并打开夹爪。"""
        return observation["observation.state"]


class FixedGripperPolicy(FakePolicy):
    """保持机械臂状态并输出固定连续夹爪值。"""

    def __init__(self, gripper: float) -> None:
        super().__init__()
        self.gripper = float(gripper)

    def select_action(self, observation: dict[str, object]) -> object:
        """返回带固定夹爪维度的动作副本。"""
        action = observation["observation.state"].clone()
        action[..., 6] = self.gripper
        return action


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
            self.assertTrue(first_trace["gripper_filter_enabled"])
            self.assertEqual(first_trace["gripper_raw_command"], 0.0)
            self.assertEqual(first_trace["gripper_filtered_command"], 0.0)
            self.assertEqual(first_trace["gripper_filter_state"], "open")
            self.assertEqual(first_trace["gripper_confirmation_count"], 0)
            self.assertFalse(first_trace["gripper_state_transitioned"])
            self.assertEqual(first_trace["gripper_transition"], "none")
            self.assertTrue(first_trace["chunk_start"])
            self.assertIn("actual_qpos_after", first_trace)
            self.assertIn("actual_qvel_after", first_trace)
            self.assertIn("ee_position_after", first_trace)
            self.assertFalse(first_trace["stage_detection_applicable"])
            summary = write_results(output, [result])
            self.assertEqual(summary["rollouts"], 1)
            self.assertEqual(summary["action_clipping"]["trace_steps"], 2)
            self.assertTrue((output / "rollouts.csv").is_file())
            self.assertTrue((output / "action_clipping_summary.json").is_file())
            self.assertTrue((output / "action_clipping_by_dimension.csv").is_file())
            self.assertTrue((output / "motion_metrics_by_rollout.csv").is_file())
            self.assertTrue((output / "motion_metrics_summary.json").is_file())
            self.assertTrue((output / "motion_metrics_report.md").is_file())
            self.assertTrue((output / "stage_metrics_by_rollout.csv").is_file())
            self.assertTrue((output / "stage_metrics_summary.json").is_file())

    def test_disabled_gripper_filter_preserves_continuous_command(self) -> None:
        """关闭过滤器时最终执行值必须与原连续夹爪值一致。"""
        with workspace_temp_dir() as output:
            result = run_single_rollout(
                FixedGripperPolicy(0.37),
                lambda value: value,
                lambda value: value,
                spec=RolloutSpec(123, "red_on_blue", "canonical", 20260),
                output_dir=output,
                fps=20,
                max_steps=1,
                device="cpu",
                checkpoint_sha256="abc",
                execution_horizon=1,
                gripper_filter_settings={"enabled": False},
            )
            trace = json.loads(Path(result.action_trace_path).read_text(encoding="utf-8"))
            self.assertFalse(trace["gripper_filter_enabled"])
            self.assertAlmostEqual(trace["physical_action"][6], 0.37, places=6)
            self.assertAlmostEqual(trace["range_safe_action"][6], 0.37, places=6)
            self.assertAlmostEqual(trace["executed_action"][6], 0.37, places=6)
            self.assertEqual(trace["gripper_filter_state"], "passthrough")

    def test_mug_rollout_writes_post_step_stage_telemetry(self) -> None:
        """杯子短rollout应写出执行后任务指标和双轨阶段状态。"""
        with workspace_temp_dir() as output:
            result = run_single_rollout(
                FakePolicy(),
                lambda value: value,
                lambda value: value,
                spec=RolloutSpec(123, "mug_on_blue", "canonical", 20260),
                output_dir=output,
                fps=20,
                max_steps=1,
                device="cpu",
                checkpoint_sha256="abc",
                execution_horizon=1,
                environment="mug",
            )
            trace = json.loads(Path(result.action_trace_path).read_text(encoding="utf-8"))
            self.assertTrue(trace["stage_detection_applicable"])
            self.assertTrue(trace["stage_detection_enabled"])
            self.assertIn("bottom_z", trace)
            self.assertIn("target_inside", trace["task_metrics"])
            self.assertEqual(set(trace["stage_current_conditions"]), {"S1", "S2", "S3", "S4", "S5"})
            self.assertEqual(set(trace["stage_direct_reached"]), {"S1", "S2", "S3", "S4", "S5"})

    def test_joint_motion_limiter_limits_second_order_reference_and_preserves_gripper(self) -> None:
        """限制器必须同时限制速度、加速度，并原样透传夹爪指令。"""
        ranges = np.asarray([[-1.0, 1.0]] * 6, dtype=np.float64)
        limiter = JointMotionLimiter(
            MotionLimits(np.full(6, 1.0), np.full(6, 4.0)),
            dt=0.1,
            arm_ctrlrange=ranges,
        )
        limiter.reset(np.zeros(6))
        first, first_mask, _ = limiter.limit(np.asarray([1.0] * 6 + [0.73]))
        second, second_mask, _ = limiter.limit(np.asarray([1.0] * 6 + [0.21]))
        np.testing.assert_allclose(first[:6], np.full(6, 0.04), atol=1e-7)
        np.testing.assert_allclose(second[:6], np.full(6, 0.12), atol=1e-7)
        self.assertTrue(first_mask[:6].all())
        self.assertTrue(second_mask[:6].all())
        self.assertAlmostEqual(float(first[6]), 0.73)
        self.assertAlmostEqual(float(second[6]), 0.21)
        self.assertTrue(np.all(np.abs(limiter.reference_velocity) <= 1.0))


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

    def test_stage_metrics_classify_first_unreached_and_legacy_status(self) -> None:
        """阶段汇总应归因首个未达成阶段，并区分旧日志与非杯子任务。"""
        with workspace_temp_dir() as output:
            mug = fake_result(output, 1, 10, "mug_on_blue", "canonical", False)
            trace = output / "mug_trace.jsonl"
            direct = {"S1": True, "S2": True, "S3": False, "S4": False, "S5": False}
            first_steps = {"S1": 10, "S2": 20, "S3": None, "S4": None, "S5": None}
            first_seconds = {"S1": 0.5, "S2": 1.0, "S3": None, "S4": None, "S5": None}
            trace.write_text(
                json.dumps(
                    {
                        "rollout_key": mug.rollout_key,
                        "stage_direct_reached": direct,
                        "stage_first_reached_step": first_steps,
                        "stage_first_reached_seconds": first_seconds,
                        "sequential_stage": "S2",
                        "highest_direct_stage": "S2",
                        "stage_order_anomaly": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            mug.action_trace_path = str(trace)
            legacy = fake_result(output, 2, 10, "mug_on_yellow", "canonical", False)
            legacy_trace = output / "legacy_trace.jsonl"
            legacy_trace.write_text(
                json.dumps({"rollout_key": legacy.rollout_key, "step": 1}) + "\n",
                encoding="utf-8",
            )
            legacy.action_trace_path = str(legacy_trace)
            cube = fake_result(output, 3, 10, "red_on_blue", "canonical", False)

            summary = write_stage_metrics_outputs(output, [mug, legacy, cube])
            with (output / "stage_metrics_by_rollout.csv").open(encoding="utf-8-sig") as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertEqual(rows[0]["failure_stage"], "S3")
            self.assertEqual(rows[1]["analysis_status"], "unavailable")
            self.assertEqual(rows[2]["analysis_status"], "not_applicable")
            self.assertEqual(summary["failures_by_stage"]["S3"], 1)
            self.assertEqual(summary["status_counts"]["unavailable"], 1)

    def test_stage_metrics_cover_s1_to_s5_success_and_partial_exception(self) -> None:
        """逐轨迹归因应覆盖五个失败阶段，并排除控制异常的候选归因。"""
        with workspace_temp_dir() as output:
            results: list[RolloutResult] = []
            expected: dict[str, str] = {}
            for failure_index, failure_stage in enumerate(("S1", "S2", "S3", "S4", "S5")):
                result = fake_result(output, 10 + failure_index, 10, "mug_on_blue", "canonical", False)
                direct = {
                    stage: index < failure_index
                    for index, stage in enumerate(("S1", "S2", "S3", "S4", "S5"))
                }
                trace = output / f"failure_{failure_stage}.jsonl"
                trace.write_text(
                    json.dumps(
                        {
                            "rollout_key": result.rollout_key,
                            "stage_direct_reached": direct,
                            "stage_first_reached_step": {
                                stage: index + 1 if direct[stage] else None
                                for index, stage in enumerate(direct)
                            },
                            "stage_first_reached_seconds": {
                                stage: (index + 1) / 20 if direct[stage] else None
                                for index, stage in enumerate(direct)
                            },
                            "sequential_stage": "none" if failure_index == 0 else f"S{failure_index}",
                            "highest_direct_stage": "none" if failure_index == 0 else f"S{failure_index}",
                            "stage_order_anomaly": False,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                result.action_trace_path = str(trace)
                results.append(result)
                expected[result.rollout_key] = failure_stage

            success = fake_result(output, 20, 10, "mug_on_yellow", "canonical", True)
            success_trace = output / "success.jsonl"
            all_reached = {stage: True for stage in ("S1", "S2", "S3", "S4", "S5")}
            success_trace.write_text(
                json.dumps(
                    {
                        "rollout_key": success.rollout_key,
                        "stage_direct_reached": all_reached,
                        "stage_first_reached_step": {stage: index + 1 for index, stage in enumerate(all_reached)},
                        "stage_first_reached_seconds": {stage: (index + 1) / 20 for index, stage in enumerate(all_reached)},
                        "sequential_stage": "S5",
                        "highest_direct_stage": "S5",
                        "stage_order_anomaly": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            success.action_trace_path = str(success_trace)
            results.append(success)
            expected[success.rollout_key] = "none"

            partial = fake_result(output, 21, 10, "mug_on_blue", "canonical", False, "control_exception")
            partial.error = "RuntimeError: synthetic"
            partial_trace = output / "partial.jsonl"
            partial_payload = json.loads(Path(results[1].action_trace_path).read_text(encoding="utf-8"))
            partial_payload["rollout_key"] = partial.rollout_key
            partial_trace.write_text(json.dumps(partial_payload) + "\n", encoding="utf-8")
            partial.action_trace_path = str(partial_trace)
            results.append(partial)

            summary = write_stage_metrics_outputs(output, results)
            with (output / "stage_metrics_by_rollout.csv").open(encoding="utf-8-sig") as csv_file:
                rows = {row["rollout_key"]: row for row in csv.DictReader(csv_file)}
            for rollout_key, failure_stage in expected.items():
                self.assertEqual(rows[rollout_key]["failure_stage"], failure_stage)
            self.assertEqual(rows[partial.rollout_key]["analysis_status"], "partial")
            self.assertEqual(rows[partial.rollout_key]["failure_stage"], "unavailable")
            self.assertEqual(summary["analyzed_mug_rollouts"], 6)
            self.assertEqual(summary["status_counts"]["partial"], 1)

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
