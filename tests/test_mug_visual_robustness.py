"""验证Mug视觉鲁棒性扰动评测工具的扰动函数、矩阵、审计与薄副本兼容性。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from evaluate import rollout as rollout_module
from evaluate import rollout_robustness as robustness_module
from evaluate.diagnose_mug_visual_robustness import (
    IMAGE_SHAPE,
    PerturbationSpec,
    RobustnessCondition,
    apply_pixel_perturbation,
    build_conditions,
    build_image_transform,
    build_summary,
    collapse_threshold,
    condition_artifact_subdir,
    format_intensity_component,
    parse_config,
    stable_condition_seed,
    verify_pixel_perturbation,
)
from evaluate.rollout import RolloutSpec
from evaluate.rollout_robustness import run_single_rollout


def random_image(seed: int = 0) -> np.ndarray:
    """生成确定性的测试参考图。"""
    return np.random.default_rng(seed).integers(0, 256, size=IMAGE_SHAPE, dtype=np.uint8)


def sample_settings() -> dict:
    """构造合法的诊断配置字典（与YAML结构一致）。"""
    return {
        "environment": "mug",
        "output_dir": "outputs/eval/tmp",
        "fps": 20,
        "max_steps": 360,
        "scene_seeds": [2291, 6705],
        "task_ids": ["mug_on_blue", "mug_on_yellow"],
        "prompt_type": "canonical",
        "policy_seeds": [20260],
        "appearance_variants": ["original", "green_white"],
        "pixel_perturbations": {"brightness": [0.5, 0.9], "gamma": [0.6, 1.4]},
        "lighting_presets": {},
        "holdout_presets": [],
        "collapse_baseline_drop_pp": 20,
    }


class FakePolicy:
    """返回当前关节状态的假策略（沿用test_evaluate.py模式）。"""

    def __init__(self) -> None:
        self.config = SimpleNamespace(use_amp=False)

    def reset(self) -> None:
        """假策略没有缓存需要清理。"""

    def select_action(self, observation: dict[str, object]) -> object:
        """保持当前关节目标并打开夹爪。"""
        return observation["observation.state"]


class ThinCopyReexportTests(unittest.TestCase):
    """薄副本re-export的符号必须与原模块是同一对象。"""

    def test_reexported_symbols_are_identical(self) -> None:
        """关键符号与原模块同一对象，run_single_rollout为本地版本。"""
        self.assertIs(robustness_module.RolloutSpec, rollout_module.RolloutSpec)
        self.assertIs(robustness_module.RolloutResult, rollout_module.RolloutResult)
        self.assertIs(robustness_module.MugStageTracker, rollout_module.MugStageTracker)
        self.assertIs(robustness_module.bootstrap_success_ci, rollout_module.bootstrap_success_ci)
        self.assertIs(robustness_module.np, rollout_module.np)
        self.assertIsNot(
            robustness_module.run_single_rollout,
            rollout_module.run_single_rollout,
        )


class PixelPerturbationTests(unittest.TestCase):
    """像素扰动必须保持格式、改变像素、随机扰动可复现。"""

    def test_all_perturbations_preserve_format(self) -> None:
        """每种扰动输出保持uint8且shape不变。"""
        image = random_image(1)
        for name in ("brightness", "contrast", "gamma", "gaussian_noise", "gaussian_blur", "jpeg"):
            with self.subTest(perturbation=name):
                rng = np.random.default_rng(42)
                intensity = {"gaussian_noise": 10.0, "jpeg": 40.0}.get(name, 0.7)
                output = apply_pixel_perturbation(image, name, intensity, rng)
                self.assertEqual(output.shape, IMAGE_SHAPE)
                self.assertEqual(output.dtype, np.uint8)
                self.assertTrue(np.isfinite(output).all())

    def test_brightness_identity_scale_is_noop(self) -> None:
        """亮度scale=1.0应等价于原图。"""
        image = random_image(2)
        output = apply_pixel_perturbation(image, "brightness", 1.0, np.random.default_rng(0))
        np.testing.assert_array_equal(output, image)

    def test_brightness_half_changes_pixels(self) -> None:
        """亮度scale=0.5应显著改变像素。"""
        image = random_image(3)
        output = apply_pixel_perturbation(image, "brightness", 0.5, np.random.default_rng(0))
        self.assertGreater(int(np.count_nonzero(image != output)), 0)

    def test_gaussian_noise_reproducible_with_seed(self) -> None:
        """高斯噪声同seed两次一致、不同seed不同。"""
        image = random_image(4)
        rng_a = np.random.default_rng(123)
        first = apply_pixel_perturbation(image, "gaussian_noise", 15.0, rng_a)
        rng_b = np.random.default_rng(123)
        second = apply_pixel_perturbation(image, "gaussian_noise", 15.0, rng_b)
        np.testing.assert_array_equal(first, second)
        rng_c = np.random.default_rng(456)
        third = apply_pixel_perturbation(image, "gaussian_noise", 15.0, rng_c)
        self.assertGreater(int(np.count_nonzero(first != third)), 0)

    def test_jpeg_and_blur_alter_random_image(self) -> None:
        """JPEG重压缩与高斯模糊应改变随机图。"""
        image = random_image(5)
        jpeg_out = apply_pixel_perturbation(image, "jpeg", 30.0, np.random.default_rng(0))
        blur_out = apply_pixel_perturbation(image, "gaussian_blur", 3.0, np.random.default_rng(0))
        self.assertGreater(int(np.count_nonzero(image != jpeg_out)), 0)
        self.assertGreater(int(np.count_nonzero(image != blur_out)), 0)

    def test_unknown_perturbation_raises(self) -> None:
        """未知扰动名称应抛ValueError。"""
        with self.assertRaises(ValueError):
            apply_pixel_perturbation(random_image(6), "unknown", 1.0, np.random.default_rng(0))


class ConfigAndConditionTests(unittest.TestCase):
    """配置解析与条件矩阵构建。"""

    def test_parse_config_accepts_valid(self) -> None:
        """合法配置应解析为规范化settings。"""
        settings = parse_config({"diagnostic": sample_settings()})
        self.assertEqual(settings["fps"], 20)
        self.assertEqual(settings["scene_seeds"], [2291, 6705])
        self.assertEqual(settings["pixel_perturbations"]["brightness"], [0.5, 0.9])

    def test_parse_config_rejects_cube_task(self) -> None:
        """非mug任务应拒绝。"""
        settings = sample_settings()
        settings["task_ids"] = ["red_on_blue"]
        with self.assertRaises(ValueError):
            parse_config({"diagnostic": settings})

    def test_parse_config_rejects_non_locked_policy_seed(self) -> None:
        """policy_seed不是20260应拒绝。"""
        settings = sample_settings()
        settings["policy_seeds"] = [1]
        with self.assertRaises(ValueError):
            parse_config({"diagnostic": settings})

    def test_parse_config_rejects_unknown_perturbation(self) -> None:
        """未知扰动维度应拒绝。"""
        settings = sample_settings()
        settings["pixel_perturbations"]["rgb_shift"] = [0.1]
        with self.assertRaises(ValueError):
            parse_config({"diagnostic": settings})

    def test_build_conditions_enumerates_matrix(self) -> None:
        """条件数量=scenes×tasks×(外观数+像素档数)，键唯一。"""
        conditions = build_conditions(sample_settings())
        expected = 2 * 2 * (2 + 2 + 2)
        self.assertEqual(len(conditions), expected)
        keys = [condition.key for condition in conditions]
        self.assertEqual(len(keys), len(set(keys)))

    def test_build_conditions_includes_lighting_cross_product(self) -> None:
        """配置光照预设后应生成外观×预设全交叉条件并映射为显式参数。"""
        settings = sample_settings()
        settings["lighting_presets"] = {
            "default": {"a_scale": 1.0, "b_azimuth_deg": 0.0, "c_scale": 1.0},
            "alt": {"a_scale": 1.4, "b_azimuth_deg": 25.0, "c_scale": 1.2},
            "holdout_1": {"a_scale": 1.2, "b_azimuth_deg": 15.0, "c_scale": 1.1},
        }
        settings["holdout_presets"] = ["holdout_1"]
        conditions = build_conditions(settings)
        combos = [c for c in conditions if c.perturbation is None]
        self.assertEqual(len(combos), 2 * 2 * (2 * 3))
        for condition in combos:
            self.assertEqual(condition.lighting_params, settings["lighting_presets"][condition.lighting_preset])
            if condition.lighting_preset == "alt":
                self.assertEqual(condition.lighting_params["a_scale"], 1.4)
                self.assertEqual(condition.lighting_params["b_azimuth_deg"], 25.0)
        presets_seen = {c.lighting_preset for c in combos}
        self.assertEqual(presets_seen, {"default", "alt", "holdout_1"})
        keys = [condition.key for condition in conditions]
        self.assertEqual(len(keys), len(set(keys)))

    def test_condition_key_distinguishes_appearance_and_pixel(self) -> None:
        """外观×光照与像素条件键必须互不冲突。"""
        appearance = RobustnessCondition(1, "mug_on_blue", 20260, appearance_variant="original")
        pixel = RobustnessCondition(
            1, "mug_on_blue", 20260, perturbation=PerturbationSpec("brightness", 0.9)
        )
        self.assertNotEqual(appearance.key, pixel.key)
        self.assertIn("app=original|light=default", appearance.key)
        self.assertIn("pert=brightness-0.9", pixel.key)

    def test_condition_artifact_subdir_uses_parameter_hierarchy(self) -> None:
        """外观与像素条件应映射到稳定的参数目录。"""
        appearance = RobustnessCondition(
            1, "mug_on_blue", 20260, appearance_variant="green_white"
        )
        pixel = RobustnessCondition(
            1,
            "mug_on_blue",
            20260,
            perturbation=PerturbationSpec("gaussian_noise", 15.0),
        )
        self.assertEqual(
            condition_artifact_subdir(appearance), Path("appearance/green_white/default")
        )
        self.assertEqual(condition_artifact_subdir(pixel), Path("pixel/gaussian_noise/15"))
        self.assertEqual(format_intensity_component(0.5), "0.5")

    def test_stable_condition_seed_is_deterministic(self) -> None:
        """条件派生种子必须确定且随条件变化。"""
        condition_a = RobustnessCondition(
            1, "mug_on_blue", 20260, perturbation=PerturbationSpec("gaussian_noise", 10.0)
        )
        condition_b = RobustnessCondition(
            2, "mug_on_blue", 20260, perturbation=PerturbationSpec("gaussian_noise", 10.0)
        )
        self.assertEqual(stable_condition_seed(condition_a), stable_condition_seed(condition_a))
        self.assertNotEqual(stable_condition_seed(condition_a), stable_condition_seed(condition_b))

    def test_image_transform_applies_to_both_cameras(self) -> None:
        """image_transform必须作用于agent与wrist两路。"""
        condition = RobustnessCondition(
            1, "mug_on_blue", 20260, perturbation=PerturbationSpec("brightness", 0.5)
        )
        transform = build_image_transform(condition)
        images = {"agent": random_image(7), "wrist": random_image(8)}
        output = transform(images)
        self.assertEqual(set(output), {"agent", "wrist"})
        self.assertGreater(int(np.count_nonzero(images["agent"] != output["agent"])), 0)
        self.assertGreater(int(np.count_nonzero(images["wrist"] != output["wrist"])), 0)


class AuditAndSummaryTests(unittest.TestCase):
    """扰动审计与汇总逻辑。"""

    def test_verify_pixel_perturbation_accepts_change(self) -> None:
        """有像素变化时应返回变化统计。"""
        original = random_image(9)
        perturbed = apply_pixel_perturbation(original, "brightness", 0.5, np.random.default_rng(0))
        audit = verify_pixel_perturbation(original, perturbed, "brightness", 0.5)
        self.assertGreater(audit["changed_pixels"], 0)
        self.assertGreater(audit["changed_fraction"], 0.0)

    def test_verify_pixel_perturbation_rejects_noop(self) -> None:
        """无像素变化时审计应失败。"""
        original = random_image(10)
        with self.assertRaises(RuntimeError):
            verify_pixel_perturbation(original, original, "brightness", 0.5)

    def test_collapse_threshold_finds_first_below(self) -> None:
        """崩溃阈值取首次跌破threshold的最低强度档。"""
        threshold = collapse_threshold([0.5, 0.7, 0.9], [0.9, 0.6, 0.2], baseline_rate=0.9, drop_pp=20)
        self.assertEqual(threshold, 0.7)

    def test_collapse_threshold_not_collapsed(self) -> None:
        """全程未跌破阈值时返回not_collapsed。"""
        threshold = collapse_threshold([0.5, 0.7, 0.9], [0.9, 0.8, 0.75], baseline_rate=0.9, drop_pp=20)
        self.assertEqual(threshold, "not_collapsed")

    def test_build_summary_aggregates_groups(self) -> None:
        """汇总应正确聚合基线与各扰动维度行。"""
        records = [
            _fake_record("appearance", "original", scene=1),
            _fake_record("appearance", "original", scene=2, success=True),
            _fake_record("pixel", "brightness", 0.5, scene=1, success=True),
            _fake_record("pixel", "brightness", 0.5, scene=2),
            _fake_record("pixel", "brightness", 0.9, scene=1),
            _fake_record("pixel", "brightness", 0.9, scene=2),
        ]
        settings = sample_settings()
        settings["scene_seeds"] = [1, 2]
        settings["pixel_perturbations"] = {"brightness": [0.5, 0.9]}
        settings["appearance_variants"] = ["original"]
        summary = build_summary(records, settings)
        self.assertAlmostEqual(summary["baseline_rate"], 0.5)
        appearance_rows = [row for row in summary["aggregate_rows"] if row["perturbation"] == "appearance"]
        brightness_rows = [
            row for row in summary["aggregate_rows"] if row["perturbation"] == "brightness"
        ]
        self.assertEqual(len(appearance_rows), 1)
        self.assertEqual(len(brightness_rows), 2)
        self.assertAlmostEqual(brightness_rows[0]["success_rate"], 0.5)
        self.assertAlmostEqual(brightness_rows[1]["success_rate"], 0.0)
        self.assertIn("brightness", summary["pixel_collapse"])


def _fake_record(
    kind: str,
    name: str,
    intensity: object | None = None,
    scene: int = 1,
    success: bool = False,
) -> dict:
    """构造与JSONL记录同构的汇总输入。"""
    if kind == "appearance":
        return {
            "rollout_key": f"scene={scene}|appearance={name}",
            "condition_key": f"scene={scene}|appearance={name}",
            "scene_seed": scene,
            "policy_seed": 20260,
            "task_id": "mug_on_blue",
            "task": "Put the mug on the blue pad.",
            "prompt_type": "canonical",
            "success": success,
            "failure_mode": "success" if success else "timeout",
            "steps": 10,
            "elapsed_seconds": 0.5,
            "latency_mean_ms": 10.0,
            "latency_p95_ms": 20.0,
            "clipped_action_steps": 0,
            "clipped_action_rate": 0.0,
            "action_trace_path": "",
            "checkpoint_sha256": "abc",
            "video_path": "",
            "video_retained": True,
            "error": "",
            "completed_at": "2026-01-01T00:00:00+00:00",
            "appearance_variant": name,
            "perturbation_name": None,
            "perturbation_intensity": None,
        }
    return {
        "rollout_key": f"scene={scene}|pert={name}-{intensity}",
        "condition_key": f"scene={scene}|pert={name}-{intensity}",
        "scene_seed": scene,
        "policy_seed": 20260,
        "task_id": "mug_on_blue",
        "task": "Put the mug on the blue pad.",
        "prompt_type": "canonical",
        "success": success,
        "failure_mode": "success" if success else "timeout",
        "steps": 10,
        "elapsed_seconds": 0.5,
        "latency_mean_ms": 10.0,
        "latency_p95_ms": 20.0,
        "clipped_action_steps": 0,
        "clipped_action_rate": 0.0,
        "action_trace_path": "",
        "checkpoint_sha256": "abc",
        "video_path": "",
        "video_retained": True,
        "error": "",
        "completed_at": "2026-01-01T00:00:00+00:00",
        "appearance_variant": None,
        "perturbation_name": name,
        "perturbation_intensity": intensity,
    }


class ThinCopyRolloutIntegrationTests(unittest.TestCase):
    """副本run_single_rollout的image_transform注入行为（本机CPU短闭环）。"""

    def test_image_transform_is_invoked_and_preserves_semantics(self) -> None:
        """注入的transform必须每步被调用，且结果与原逻辑一致。"""
        from tests.test_evaluate import workspace_temp_dir

        calls = {"count": 0}

        def counting_transform(images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            calls["count"] += 1
            return images

        with workspace_temp_dir() as output:
            spec = RolloutSpec(123, "mug_on_blue", "canonical", 20260)
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
                environment="mug",
                image_transform=counting_transform,
            )
            self.assertEqual(calls["count"], 2)
            self.assertEqual(result.failure_mode, "timeout")
            self.assertEqual(result.error, "")
            self.assertEqual(Path(result.video_path).parent, output / "videos")
            self.assertEqual(Path(result.action_trace_path).parent, output / "action_traces")
            lines = Path(result.action_trace_path).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            trace = json.loads(lines[0])
            self.assertTrue(trace["stage_detection_enabled"])

    def test_image_transform_none_matches_identity(self) -> None:
        """image_transform=None与identity变换结果一致。"""
        from tests.test_evaluate import workspace_temp_dir

        results: list[dict] = []
        for transform in (None, lambda images: images):
            with workspace_temp_dir() as output:
                spec = RolloutSpec(123, "mug_on_blue", "canonical", 20260)
                result = run_single_rollout(
                    FakePolicy(),
                    lambda value: value,
                    lambda value: value,
                    spec=spec,
                    output_dir=output,
                    fps=20,
                    max_steps=1,
                    device="cpu",
                    checkpoint_sha256="abc",
                    execution_horizon=1,
                    environment="mug",
                    image_transform=transform,
                )
                results.append(
                    {
                        "failure_mode": result.failure_mode,
                        "steps": result.steps,
                        "error": result.error,
                        "success": result.success,
                    }
                )
        self.assertEqual(results[0], results[1])

    def test_artifact_stem_override_isolates_artifacts(self) -> None:
        """同一spec下artifact_stem_override必须产生独立视频与动作日志。"""
        from tests.test_evaluate import workspace_temp_dir

        with workspace_temp_dir() as output:
            spec = RolloutSpec(123, "mug_on_blue", "canonical", 20260)
            first = run_single_rollout(
                FakePolicy(),
                lambda value: value,
                lambda value: value,
                spec=spec,
                output_dir=output,
                fps=20,
                max_steps=1,
                device="cpu",
                checkpoint_sha256="abc",
                execution_horizon=1,
                environment="mug",
                artifact_stem_override="scene_123_policy_20260_mug_on_blue_canonical__appearance_green_white",
                artifact_subdir_override="appearance/green_white",
            )
            second = run_single_rollout(
                FakePolicy(),
                lambda value: value,
                lambda value: value,
                spec=spec,
                output_dir=output,
                fps=20,
                max_steps=1,
                device="cpu",
                checkpoint_sha256="abc",
                execution_horizon=1,
                environment="mug",
                artifact_stem_override="scene_123_policy_20260_mug_on_blue_canonical__pert_brightness_0.5",
                artifact_subdir_override="pixel/brightness/0.5",
            )
            self.assertNotEqual(first.video_path, second.video_path)
            self.assertNotEqual(first.action_trace_path, second.action_trace_path)
            self.assertTrue(Path(first.video_path).is_file())
            self.assertTrue(Path(second.video_path).is_file())
            self.assertEqual(
                Path(first.video_path).parent.relative_to(output / "videos"),
                Path("appearance/green_white"),
            )
            self.assertEqual(
                Path(first.action_trace_path).parent.relative_to(output / "action_traces"),
                Path("appearance/green_white"),
            )
            self.assertEqual(
                Path(second.video_path).parent.relative_to(output / "videos"),
                Path("pixel/brightness/0.5"),
            )
            self.assertEqual(
                Path(second.action_trace_path).parent.relative_to(output / "action_traces"),
                Path("pixel/brightness/0.5"),
            )


if __name__ == "__main__":
    unittest.main()
