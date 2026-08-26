"""验证Mug视觉鲁棒性扰动评测工具的扰动函数、矩阵、审计与薄副本兼容性。"""

from __future__ import annotations

import json
import shutil
import tempfile
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
    prepare_diagnostic_output,
    stable_condition_seed,
    validate_completed_results,
    wilson_score_interval,
    verify_pixel_perturbation,
    write_report,
)
from evaluate.common import load_yaml_config
from evaluate.rollout import RolloutSpec
from evaluate.rollout_robustness import run_single_rollout
from scripts.generate_mug_color_holdouts import (
    HOLDOUT_COLORS,
    build_body_mask,
    generate_holdout_textures,
)


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

    def test_parse_config_accepts_multiple_policy_seeds(self) -> None:
        """多个非重复policy seed应合法。"""
        settings = sample_settings()
        settings["policy_seeds"] = [20260, 20261]
        parsed = parse_config({"diagnostic": settings})
        self.assertEqual(parsed["policy_seeds"], [20260, 20261])

    def test_parse_config_rejects_duplicate_policy_seed(self) -> None:
        """重复policy seed应拒绝。"""
        settings = sample_settings()
        settings["policy_seeds"] = [20260, 20260]
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

    def test_explicit_appearance_conditions_avoid_cross_product(self) -> None:
        """显式外观条件只生成列出的组合并支持多policy seed。"""
        raw = sample_settings()
        raw.pop("appearance_variants")
        raw["policy_seeds"] = [20260, 20261]
        raw["pixel_perturbations"] = {}
        raw["lighting_presets"] = {
            "default": {"a_scale": 1.0, "b_azimuth_deg": 0.0, "c_scale": 1.0},
            "new_light": {"a_scale": 1.5, "b_azimuth_deg": 0.0, "c_scale": 0.55},
        }
        raw["appearance_conditions"] = [
            {"variant": "original", "lighting": "default"},
            {"variant": "holdout_gray", "lighting": "new_light"},
        ]
        settings = parse_config({"diagnostic": raw})
        conditions = build_conditions(settings)
        self.assertEqual(len(conditions), 2 * 2 * 2 * 2)
        pairs = {(item.appearance_variant, item.lighting_preset) for item in conditions}
        self.assertEqual(pairs, {("original", "default"), ("holdout_gray", "new_light")})

    def test_explicit_and_legacy_appearance_fields_are_mutually_exclusive(self) -> None:
        """显式组合与旧外观列表同时配置应拒绝。"""
        raw = sample_settings()
        raw["appearance_conditions"] = [{"variant": "original", "lighting": "default"}]
        with self.assertRaisesRegex(ValueError, "不可同时配置"):
            parse_config({"diagnostic": raw})

    def test_official_color_ood_and_legacy_matrix_counts(self) -> None:
        """正式新矩阵为576条，旧DR矩阵保持120条。"""
        project_root = Path(__file__).resolve().parents[1]
        color_config = load_yaml_config(
            project_root / "configs/eval/mug_robustness/diagnose_mug_color_ood_dr.yaml"
        )
        legacy_config = load_yaml_config(
            project_root / "configs/eval/mug_robustness/diagnose_mug_robustness_dr.yaml"
        )
        color_conditions = build_conditions(parse_config(color_config))
        legacy_conditions = build_conditions(parse_config(legacy_config))
        self.assertEqual(len(color_conditions), 576)
        self.assertEqual(len(legacy_conditions), 120)
        self.assertEqual(len({condition.key for condition in color_conditions}), 576)
        training_domain_config = (
            project_root / "configs/domain_randomize.yaml"
        ).read_text(encoding="utf-8")
        self.assertTrue(all(name not in training_domain_config for name in HOLDOUT_COLORS))

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

    def test_wilson_interval_keeps_uncertainty_for_all_successes(self) -> None:
        """少量全成功样本的Wilson下界不得退化为1。"""
        low, high = wilson_score_interval(10, 10)
        self.assertLess(low, 0.8)
        self.assertAlmostEqual(high, 1.0)

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

    def test_color_summary_reports_macros_gaps_breakdowns_and_failures(self) -> None:
        """纯色实验应汇总颜色宏平均、基准差距、分组和失败索引。"""
        pairs = [
            ("original", "default", True),
            ("holdout_gray", "default", True),
            ("holdout_gray", "new_light", False),
            ("holdout_purple", "default", True),
            ("holdout_purple", "new_light", True),
            ("holdout_orange", "default", False),
            ("holdout_orange", "new_light", False),
        ]
        records = []
        for index, (variant, lighting, succeeded) in enumerate(pairs, start=1):
            record = _fake_record("appearance", variant, scene=index, success=succeeded)
            record["lighting_preset"] = lighting
            record["condition_key"] += f"|light={lighting}"
            records.append(record)
        settings = sample_settings()
        settings["appearance_variants"] = [
            "original",
            "holdout_gray",
            "holdout_purple",
            "holdout_orange",
        ]
        settings["appearance_conditions"] = [
            {"variant": variant, "lighting": lighting}
            for variant, lighting, _ in pairs
        ]
        settings["lighting_presets"] = {
            "default": {"a_scale": 1.0, "b_azimuth_deg": 0.0, "c_scale": 1.0},
            "new_light": {"a_scale": 1.5, "b_azimuth_deg": 0.0, "c_scale": 0.55},
        }
        settings["pixel_perturbations"] = {}
        settings["descriptive_only"] = True
        summary = build_summary(records, settings)
        color = summary["color_generalization"]
        self.assertAlmostEqual(color["default_macro_success_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(color["new_light_macro_success_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(color["default_gap_from_baseline"], 1.0 / 3.0)
        self.assertEqual(len(summary["failure_rows"]), 3)
        self.assertTrue(summary["task_rows"])
        self.assertTrue(summary["policy_seed_rows"])
        self.assertFalse(summary["combo_collapse"])
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "report.md"
            write_report(report_path, summary)
            report = report_path.read_text(encoding="utf-8")
            self.assertIn("未见纯色泛化摘要", report)
            self.assertIn("failed_rollouts.csv", report)


class HoldoutTextureGenerationTests(unittest.TestCase):
    """未见纯色纹理生成必须确定且只改变杯身掩码。"""

    def test_generation_is_deterministic_and_mask_limited(self) -> None:
        """重复生成内容一致，掩码外像素逐值不变。"""
        from PIL import Image

        project_root = Path(__file__).resolve().parents[1]
        source_path = project_root / "assets/mujoco/mug_5/visual/image0.png"
        with tempfile.TemporaryDirectory() as temporary:
            visual_dir = Path(temporary)
            shutil.copy2(source_path, visual_dir / "image0.png")
            first = generate_holdout_textures(visual_dir)
            second = generate_holdout_textures(visual_dir)
            self.assertEqual(first, second)
            source = np.asarray(Image.open(visual_dir / "image0.png").convert("RGB"))
            mask = build_body_mask(source)
            for variant, rgb in HOLDOUT_COLORS.items():
                output = np.asarray(
                    Image.open(visual_dir / f"image0_{variant}.png").convert("RGB")
                )
                self.assertEqual(output.shape, source.shape)
                np.testing.assert_array_equal(output[~mask], source[~mask])
                expected = np.broadcast_to(np.asarray(rgb, dtype=np.uint8), output[mask].shape)
                np.testing.assert_array_equal(output[mask], expected)


class DiagnosticResumeTests(unittest.TestCase):
    """视觉评测断点续跑必须绑定实验身份并拒绝损坏结果。"""

    def test_manifest_mismatch_is_rejected(self) -> None:
        """同一输出目录不得混用不同条件矩阵。"""
        manifest = {
            "schema_version": 2,
            "created_at": "2026-01-01T00:00:00+00:00",
            "conditions": ["a"],
            "appearance_render_audits": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            prepare_diagnostic_output(output, dict(manifest))
            changed = dict(manifest)
            changed["conditions"] = ["b"]
            with self.assertRaisesRegex(ValueError, "manifest"):
                prepare_diagnostic_output(output, changed)

    def test_duplicate_completed_condition_is_rejected(self) -> None:
        """重复condition_key会破坏统计，续跑前必须拒绝。"""
        condition = RobustnessCondition(1, "mug_on_blue", 20260)
        record = {
            "condition_key": condition.key,
            "checkpoint_sha256": "abc",
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "rollouts.jsonl").write_text(
                json.dumps(record) + "\n" + json.dumps(record) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "重复"):
                validate_completed_results(output, [condition], "abc")


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
