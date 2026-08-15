"""验证红绿积木语言Token差异诊断的矩阵、配对和归档产物。"""

from __future__ import annotations

import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from evaluate.diagnose_language import (
    build_manifest,
    build_pairwise_comparisons,
    build_prompt_matrix,
    build_summary,
    build_token_records,
    load_preprocessor,
    write_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "train"
    / "smolvla_ur10e_b8_s15000_r2"
    / "checkpoints"
    / "010000"
    / "pretrained_model"
)


@contextmanager
def workspace_temp_dir() -> Iterator[Path]:
    """在项目内创建并自动清理测试目录。

    Yields:
        测试独占临时目录。
    """
    path = PROJECT_ROOT / f".language-diagnostic-test-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class PromptMatrixTests(unittest.TestCase):
    """验证无需checkpoint即可确认的12条提示矩阵。"""

    def test_prompt_matrix_has_all_color_and_template_combinations(self) -> None:
        """矩阵必须覆盖2种积木、2种底板和3种措辞。"""
        matrix = build_prompt_matrix()
        self.assertEqual(len(matrix), 12)
        keys = {
            (sample["cube_color"], sample["pad_color"], sample["prompt_type"])
            for sample in matrix
        }
        self.assertEqual(len(keys), 12)
        self.assertIn(("red", "blue", "canonical"), keys)
        self.assertIn(("green", "yellow", "unseen"), keys)


@unittest.skipUnless(CHECKPOINT.is_dir(), "本机缺少010000 checkpoint，跳过真实preprocessor集成测试")
class CheckpointLanguageDiagnosticTests(unittest.TestCase):
    """使用010000 checkpoint真实preprocessor验证颜色Token差异。"""

    @classmethod
    def setUpClass(cls) -> None:
        """只加载一次CPU预处理器并生成完整诊断结果。"""
        cls.preprocessor, cls.tokenizer, cls.settings = load_preprocessor(CHECKPOINT)
        cls.records = build_token_records(cls.preprocessor, cls.tokenizer, cls.settings)
        cls.comparisons = build_pairwise_comparisons(cls.records)
        cls.summary = build_summary(cls.records, cls.comparisons)

    def test_canonical_red_and_green_have_known_distinct_tokens(self) -> None:
        """canonical红绿Token应只在索引2分别使用2382和2654。"""
        indexed = {
            (record["task_id"], record["prompt_type"]): record for record in self.records
        }
        red = indexed[("red_on_blue", "canonical")]
        green = indexed[("green_on_blue", "canonical")]
        self.assertEqual(red["active_length"], 10)
        self.assertEqual(green["active_length"], 10)
        self.assertEqual(red["active_token_ids"][2], 2382)
        self.assertEqual(green["active_token_ids"][2], 2654)
        differences = [
            index
            for index, pair in enumerate(zip(red["active_token_ids"], green["active_token_ids"]))
            if pair[0] != pair[1]
        ]
        self.assertEqual(differences, [2])

    def test_padding_attention_mask_and_newline_are_preserved(self) -> None:
        """全部记录应为48位、无截断且以有效换行Token结束。"""
        for record in self.records:
            self.assertEqual(len(record["token_ids"]), 48)
            self.assertEqual(len(record["attention_mask"]), 48)
            self.assertFalse(record["truncated"])
            self.assertTrue(record["direct_tokenizer_matches_preprocessor"])
            self.assertEqual(record["active_token_ids"][-1], 198)
            self.assertEqual(record["active_token_texts"][-1], "Ċ")

    def test_all_color_pairs_pass_locked_criteria(self) -> None:
        """6组红绿和6组蓝黄配对都必须只差一个有效颜色Token。"""
        cube_pairs = [item for item in self.comparisons if item["pair_type"] == "cube_color"]
        pad_pairs = [item for item in self.comparisons if item["pair_type"] == "pad_color"]
        self.assertEqual(len(cube_pairs), 6)
        self.assertEqual(len(pad_pairs), 6)
        self.assertTrue(all(item["passed"] for item in cube_pairs))
        self.assertTrue(all(item["passed"] for item in pad_pairs))
        self.assertEqual(self.summary["status"], "pass")

    def test_all_diagnostic_artifacts_are_non_empty(self) -> None:
        """五类归档文件必须存在、非空且记录数量正确。"""
        with workspace_temp_dir() as output:
            manifest = build_manifest(
                CHECKPOINT,
                self.settings,
                self.records,
                self.comparisons,
            )
            write_outputs(output, manifest, self.records, self.comparisons, self.summary)
            filenames = (
                "run_manifest.json",
                "token_records.jsonl",
                "pairwise_comparison.csv",
                "summary.json",
                "report.md",
            )
            for filename in filenames:
                self.assertGreater((output / filename).stat().st_size, 0)
            self.assertEqual(len((output / "token_records.jsonl").read_text(encoding="utf-8").splitlines()), 12)
            saved_summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_summary["comparison_count"], 12)
            self.assertFalse(manifest["model_weights_loaded"])
            self.assertEqual(manifest["environment"]["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
