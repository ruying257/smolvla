"""验证视觉反事实因果诊断的矩阵、干预、指标和分流。"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from evaluate.diagnose_visual_counterfactual import (
    VISUAL_VARIANTS,
    audit_visual_intervention,
    build_conditions,
    capture_rgb_and_masks,
    capture_scene_base,
    causal_selectivity,
    consistency_label,
    diagnose_causal_stage,
    mask_to_token_weights,
    parse_config,
    repeat_control_keys,
    restore_scene_base,
    spatial_metrics,
    visual_variant_context,
    weighted_tokens,
)
from evaluate.common import load_yaml_config
from sim.environment import CleanTabletopEnv


LOCKED_CONFIG = {
    "diagnostic": {
        "scene_seeds": [0, 1, 2, 3, 4, 5],
        "source_colors": ["red", "green"],
        "pad_color": "blue",
        "prompt_type": "canonical",
        "visual_variants": list(VISUAL_VARIANTS),
        "neutral_rgba": [0.5, 0.5, 0.5, 1.0],
        "policy_seed": 20260,
        "analysis_horizons": [10, 50],
    }
}


class VisualCounterfactualMatrixTests(unittest.TestCase):
    """验证48条件、12重复控制和锁定配置。"""

    def test_matrix_and_repeat_controls(self) -> None:
        """6场景×2指令×4视觉版本必须得到48个唯一条件。"""
        settings = parse_config(LOCKED_CONFIG)
        conditions = build_conditions(settings)
        self.assertEqual(len(conditions), 48)
        self.assertEqual(len({condition.key for condition in conditions}), 48)
        self.assertEqual(len(repeat_control_keys(conditions)), 12)
        self.assertEqual({condition.policy_seed for condition in conditions}, {20260})

    def test_policy_seed_is_locked(self) -> None:
        """视觉反事实首轮只允许policy seed 20260。"""
        changed = {"diagnostic": {**LOCKED_CONFIG["diagnostic"], "policy_seed": 20261}}
        with self.assertRaisesRegex(ValueError, "20260"):
            parse_config(changed)

    def test_repository_config_and_readme_match_cli_contract(self) -> None:
        """仓库配置、命令和Markdown代码块必须与真实入口保持一致。"""
        root = Path(__file__).resolve().parents[1]
        settings = parse_config(
            load_yaml_config(root / "configs" / "diagnose_visual_counterfactual.yaml")
        )
        self.assertEqual(len(build_conditions(settings)), 48)
        readme = (root / "evaluate" / "README.md").read_text(encoding="utf-8")
        self.assertIn("python -m evaluate.diagnose_visual_counterfactual", readme)
        self.assertIn("--config configs\\diagnose_visual_counterfactual.yaml", readme)
        self.assertEqual(readme.count("```") % 2, 0)


class VisualCounterfactualMetricTests(unittest.TestCase):
    """验证ROI聚合、CSI、空间跟随和故障分流。"""

    def test_mask_maps_to_expected_visual_token(self) -> None:
        """256 mask经等价512缩放后应正确聚合至8×8网格。"""
        mask = np.zeros((256, 256), dtype=bool)
        mask[:32, 32:64] = True
        weights = mask_to_token_weights(mask)
        self.assertEqual(weights.shape, (64,))
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertEqual(int(np.argmax(weights)), 1)
        tokens = np.arange(64 * 2, dtype=np.float32).reshape(64, 2)
        np.testing.assert_array_equal(weighted_tokens(tokens, weights), tokens[1])

    def test_empty_roi_and_nonfinite_tokens_fail(self) -> None:
        """空ROI和非有限特征必须显式失败，不能静默进入统计。"""
        empty = np.zeros((256, 256), dtype=bool)
        with self.assertRaises(ValueError):
            mask_to_token_weights(empty)
        np.testing.assert_array_equal(
            mask_to_token_weights(empty, allow_empty=True),
            np.zeros(64, dtype=np.float32),
        )
        tokens = np.zeros((64, 2), dtype=np.float32)
        tokens[0, 0] = np.nan
        with self.assertRaises(ValueError):
            weighted_tokens(tokens, np.ones(64) / 64)

    def test_csi_handles_direction_and_zero_denominator(self) -> None:
        """CSI应保留方向并安全处理完全不敏感条件。"""
        self.assertAlmostEqual(causal_selectivity(3.0, 1.0), 0.5)
        self.assertAlmostEqual(causal_selectivity(1.0, 3.0), -0.5)
        self.assertEqual(causal_selectivity(0.0, 0.0), 0.0)
        with self.assertRaises(ValueError):
            causal_selectivity(float("nan"), 1.0)

    def test_consistency_labels(self) -> None:
        """六scene方向统计必须覆盖一致、混合和反向/不敏感。"""
        self.assertEqual(consistency_label([1]), "insufficient_scenes")
        self.assertEqual(consistency_label([1, 1, 1, 1, 1, 1]), "consistent")
        self.assertEqual(consistency_label([1, 1, 1, -1, -1, -1]), "mixed")
        self.assertEqual(
            consistency_label([1, 1, -1, -1, -1, -1]),
            "opposite_or_insensitive",
        )

    def test_four_failure_routes(self) -> None:
        """假特征和假动作必须命中四条最早故障分流。"""
        self.assertEqual(
            diagnose_causal_stage(False, "consistent", "consistent", "consistent"),
            "vision_encoder_or_connector_insensitive",
        )
        self.assertEqual(
            diagnose_causal_stage(True, "opposite_or_insensitive", "consistent", "consistent"),
            "cross_modal_grounding_insufficient",
        )
        self.assertEqual(
            diagnose_causal_stage(True, "consistent", "opposite_or_insensitive", "consistent"),
            "action_expert_ignores_visual_grounding",
        )
        self.assertEqual(
            diagnose_causal_stage(True, "consistent", "consistent", "opposite_or_insensitive"),
            "spatial_action_mapping_insufficient",
        )
        self.assertEqual(
            diagnose_causal_stage(True, "mixed", "opposite_or_insensitive", "consistent"),
            "cross_modal_grounding_mixed",
        )
        self.assertEqual(
            diagnose_causal_stage(True, "consistent", "mixed", "consistent"),
            "action_expert_visual_use_mixed",
        )

    def test_spatial_following_aligned_case(self) -> None:
        """动作末端迁移与目标位移同向时余弦和follow gain应为1。"""
        original = np.zeros((50, 3), dtype=np.float64)
        swapped = np.tile(np.asarray([0.2, 0.0, 0.0]), (50, 1))
        metrics = spatial_metrics(
            original,
            swapped,
            np.asarray([0.0, 0.0, 0.0]),
            np.asarray([0.2, 0.0, 0.0]),
            np.asarray([0.0, 1.0, 0.0]),
            np.asarray([0.2, 1.0, 0.0]),
            50,
        )
        self.assertAlmostEqual(metrics["position_follow_cosine"], 1.0)
        self.assertAlmostEqual(metrics["follow_gain"], 1.0)


class VisualCounterfactualMujocoTests(unittest.TestCase):
    """在真实MuJoCo模型上验证干预、分割和自动恢复。"""

    def test_interventions_are_isolated_and_restorable(self) -> None:
        """交换完整qpos、中性化不移动物体，退出上下文后应完整恢复。"""
        with CleanTabletopEnv() as env:
            env.reset(0)
            base = capture_scene_base(env)
            original_images, original_masks = capture_rgb_and_masks(env)
            for variant in VISUAL_VARIANTS[1:]:
                with visual_variant_context(env, base, variant, [0.5, 0.5, 0.5, 1.0]):
                    changed = capture_scene_base(env)
                    changed_images, changed_masks = capture_rgb_and_masks(env)
                    np.testing.assert_array_equal(changed.robot_state, base.robot_state)
                    if variant == "swap_positions":
                        np.testing.assert_allclose(changed.cube_qpos[0], base.cube_qpos[1])
                        np.testing.assert_allclose(changed.cube_qpos[1], base.cube_qpos[0])
                    else:
                        np.testing.assert_allclose(changed.cube_qpos, base.cube_qpos)
                    audit = audit_visual_intervention(
                        original_images,
                        original_masks,
                        changed_images,
                        changed_masks,
                        variant,
                    )
                    self.assertTrue(all(value["changed_pixel_count"] > 0 for value in audit.values()))
                    for camera in changed_masks.values():
                        self.assertTrue(all(mask.any() for mask in camera.values()))
                restored = capture_scene_base(env)
                np.testing.assert_allclose(restored.cube_qpos, base.cube_qpos)
                np.testing.assert_allclose(restored.cube_rgba, base.cube_rgba)
            restore_scene_base(env, base)

    def test_single_camera_occlusion_uses_zero_roi_without_faking_mask(self) -> None:
        """腕部相机遮挡积木时应保留零mask，但外部相机必须仍能观测。"""
        with CleanTabletopEnv() as env:
            env.reset(3)
            base = capture_scene_base(env)
            original_images, original_masks = capture_rgb_and_masks(env)
            self.assertTrue(original_masks["agent"]["green"].any())
            self.assertFalse(original_masks["wrist"]["green"].any())
            with visual_variant_context(
                env, base, "neutralize_green", [0.5, 0.5, 0.5, 1.0]
            ):
                changed_images, changed_masks = capture_rgb_and_masks(env)
                audit = audit_visual_intervention(
                    original_images,
                    original_masks,
                    changed_images,
                    changed_masks,
                    "neutralize_green",
                )
            self.assertFalse(audit["wrist"]["affected_object_visible"])
            self.assertEqual(audit["wrist"]["changed_pixel_count"], 0)


if __name__ == "__main__":
    unittest.main()
