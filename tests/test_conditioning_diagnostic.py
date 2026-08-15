"""验证固定条件VLM特征与Action Chunk诊断的核心不变量。"""

from __future__ import annotations

import unittest

import numpy as np

from evaluate.diagnose_conditioning import (
    ConditioningSpec,
    action_distance,
    array_sha256,
    build_condition_specs,
    build_pair_specs,
    clip_action_chunk,
    compare_action_sets,
    compare_feature_sets,
    diagnostic_hint,
    parse_diagnostic_config,
    processed_input_hashes,
    repeat_is_deterministic,
)


LOCKED_CONFIG = {
    "diagnostic": {
        "scene_seeds": [0, 1, 2, 3, 4, 5],
        "prompt_types": ["canonical", "synonym", "unseen"],
        "policy_seed": 20260,
    }
}


class ConditioningMatrixTests(unittest.TestCase):
    """验证72条件与两类各36组配对。"""

    def test_locked_matrix_has_72_unique_conditions(self) -> None:
        """6场景、4任务和3措辞必须生成72个唯一条件。"""
        settings = parse_diagnostic_config(LOCKED_CONFIG)
        specs = build_condition_specs(settings)
        self.assertEqual(len(specs), 72)
        self.assertEqual(len({spec.key for spec in specs}), 72)
        self.assertEqual({spec.policy_seed for spec in specs}, {20260})
        self.assertEqual({spec.task_id for spec in specs}, {
            "red_on_blue", "red_on_yellow", "green_on_blue", "green_on_yellow"
        })

    def test_pair_matrix_has_36_source_and_36_target_controls(self) -> None:
        """同一批条件必须同时支持红绿与蓝黄严格配对。"""
        specs = build_condition_specs(parse_diagnostic_config(LOCKED_CONFIG))
        pairs = build_pair_specs(specs)
        source = [pair for pair in pairs if pair["pair_type"] == "cube_color"]
        target = [pair for pair in pairs if pair["pair_type"] == "pad_color"]
        self.assertEqual(len(source), 36)
        self.assertEqual(len(target), 36)
        self.assertEqual(len(pairs), 72)

    def test_policy_seed_is_locked(self) -> None:
        """诊断不得悄然扩展为多policy seed实验。"""
        config = {"diagnostic": {**LOCKED_CONFIG["diagnostic"], "policy_seed": 20261}}
        with self.assertRaisesRegex(ValueError, "20260"):
            parse_diagnostic_config(config)

    def test_condition_key_contains_all_identity_fields(self) -> None:
        """条件键必须包含场景、任务、措辞和唯一policy seed。"""
        spec = ConditioningSpec(3, "green", "yellow", "unseen", 20260)
        self.assertEqual(
            spec.key,
            "scene_3_green_on_yellow_unseen_policy_20260",
        )


class ConditioningMetricTests(unittest.TestCase):
    """验证动作、裁剪、哈希和分层提示。"""

    def test_array_hash_changes_with_dtype_shape_or_value(self) -> None:
        """输入哈希必须同时对dtype、shape和值敏感。"""
        base = np.zeros((2, 3), dtype=np.float32)
        self.assertEqual(array_sha256(base), array_sha256(base.copy()))
        self.assertNotEqual(array_sha256(base), array_sha256(base.astype(np.float64)))
        self.assertNotEqual(array_sha256(base), array_sha256(base.reshape(3, 2)))
        changed = base.copy()
        changed[0, 0] = 1.0
        self.assertNotEqual(array_sha256(base), array_sha256(changed))

    def test_processed_hashes_exclude_language(self) -> None:
        """固定输入校验只比较视觉和状态，不应把预期变化的Token纳入。"""
        first = {
            "observation.images.agent": np.zeros((1, 3, 2, 2), dtype=np.float32),
            "observation.state": np.zeros((1, 7), dtype=np.float32),
            "observation.language.tokens": np.asarray([[1, 2]], dtype=np.int64),
        }
        second = {**first, "observation.language.tokens": np.asarray([[9, 2]], dtype=np.int64)}
        self.assertEqual(processed_input_hashes(first), processed_input_hashes(second))

    def test_action_distance_reports_time_and_dimension_metrics(self) -> None:
        """50步七维动作应同时生成总体、逐步和逐维指标。"""
        left = np.zeros((50, 7), dtype=np.float32)
        right = left.copy()
        right[:10, 2] = 0.5
        metrics = action_distance(left, right, 10)
        self.assertGreater(metrics["l2"], 0.0)
        self.assertEqual(len(metrics["per_step_l2"]), 10)
        self.assertEqual(len(metrics["per_dimension_mae"]), 7)
        self.assertEqual(metrics["per_dimension_mae"][2], 0.5)
        self.assertEqual(metrics["gripper_mae"], 0.0)

    def test_clipping_can_collapse_action_difference(self) -> None:
        """不同物理动作越过同一上限后可变成相同执行动作。"""
        ranges = np.tile(np.asarray([[-1.0, 1.0]], dtype=np.float64), (6, 1))
        left = np.full((50, 7), 2.0, dtype=np.float32)
        right = np.full((50, 7), 3.0, dtype=np.float32)
        left_clipped, left_mask = clip_action_chunk(left, ranges)
        right_clipped, right_mask = clip_action_chunk(right, ranges)
        self.assertTrue(left_mask.all())
        self.assertTrue(right_mask.all())
        np.testing.assert_array_equal(left_clipped, right_clipped)
        hint = diagnostic_hint(1.0, 2.0, 0.0, 0.0)
        self.assertEqual(hint, "action_difference_collapsed_by_clipping")

    def test_diagnostic_paths_cover_insensitive_and_expert_suppression(self) -> None:
        """假模型指标应覆盖无差异和VLM有差异但动作相同两条路径。"""
        self.assertEqual(diagnostic_hint(0.0, 0.0, 0.0, 0.0), "vlm_and_action_insensitive")
        self.assertEqual(
            diagnostic_hint(1.0, 0.0, 0.0, 0.0),
            "vlm_sensitive_action_insensitive",
        )

    def test_repeat_tolerance_marks_numerical_drift(self) -> None:
        """锁定容差内视为确定，明显差异必须标记为非确定。"""
        first = np.zeros((50, 7), dtype=np.float32)
        close = first + 1e-7
        far = first.copy()
        far[0, 0] = 1e-3
        self.assertTrue(repeat_is_deterministic(first, close))
        self.assertFalse(repeat_is_deterministic(first, far))

    def test_feature_comparison_reports_changed_color_token(self) -> None:
        """特征CSV应显式给出颜色Token位置和该位置的距离。"""
        pair = {"pair_type": "cube_color", "left_key": "red", "right_key": "green"}
        red = np.zeros((3, 4), dtype=np.float32)
        green = red.copy()
        green[1] = 1.0
        metadata = {
            "active_token_indices": np.arange(3),
            "prefix_token_indices": np.arange(3),
        }
        features = {
            "red": {**metadata, "token_ids": np.asarray([1, 2, 3]), "final_hidden": red},
            "green": {**metadata, "token_ids": np.asarray([1, 9, 3]), "final_hidden": green},
        }
        row = compare_feature_sets(pair, features)[0]
        self.assertEqual(row["changed_token_positions"], "[1]")
        self.assertEqual(row["changed_token_l2"], 2.0)

    def test_action_comparison_counts_clipping_collapse(self) -> None:
        """动作CSV应显式统计裁剪后消失的差异元素和步骤。"""
        pair = {"pair_type": "cube_color", "left_key": "red", "right_key": "green"}
        physical_red = np.zeros((50, 7), dtype=np.float32)
        physical_green = physical_red.copy()
        physical_red[0, 0] = 2.0
        physical_green[0, 0] = 3.0
        clipped = np.zeros((50, 7), dtype=np.float32)
        actions = {
            "red": {"normalized": physical_red, "physical": physical_red, "clipped": clipped},
            "green": {"normalized": physical_green, "physical": physical_green, "clipped": clipped},
        }
        rows = compare_action_sets(pair, actions)
        row = next(item for item in rows if item["stage"] == "physical" and item["horizon"] == 10)
        self.assertEqual(row["clipping_collapsed_element_count"], 1)
        self.assertEqual(row["clipping_collapsed_step_count"], 1)
        self.assertEqual(row["clipped_difference_retained_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()
