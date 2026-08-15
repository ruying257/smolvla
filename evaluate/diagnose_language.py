"""诊断SmolVLA任务文本经过checkpoint预处理后的Token差异。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from evaluate.common import find_pretrained_model, resolve_path, write_json
from evaluate.rollout import build_prompt, make_policy_observation


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


PROMPT_TYPES = ("canonical", "synonym", "unseen")
CUBE_COLORS = ("red", "green")
PAD_COLORS = ("blue", "yellow")
TASK_IDS = {
    ("red", "blue"): "red_on_blue",
    ("red", "yellow"): "red_on_yellow",
    ("green", "blue"): "green_on_blue",
    ("green", "yellow"): "green_on_yellow",
}


def build_parser() -> argparse.ArgumentParser:
    """创建语言Token诊断命令行解析器。"""
    parser = argparse.ArgumentParser(description="诊断红绿积木任务文本的SmolVLA Token差异")
    parser.add_argument("--checkpoint", type=Path, required=True, help="完整模型、checkpoint或训练输出目录")
    parser.add_argument("--output-dir", type=Path, required=True, help="独立诊断产物目录")
    return parser


def sha256_file(path: Path) -> str:
    """流式计算文件SHA-256。

    Args:
        path: 需要计算哈希的文件。

    Returns:
        小写十六进制SHA-256。
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tokenizer_settings(checkpoint: Path) -> dict[str, Any]:
    """从checkpoint预处理配置读取真实Tokenizer参数。

    Args:
        checkpoint: 完整pretrained_model目录。

    Returns:
        Tokenizer名称、长度、padding及截断配置。

    Raises:
        ValueError: checkpoint没有且仅有一个Tokenizer处理步骤时抛出。
    """
    path = checkpoint / "policy_preprocessor.json"
    content = json.loads(path.read_text(encoding="utf-8"))
    tokenizer_steps = [
        step for step in content.get("steps", []) if step.get("registry_name") == "tokenizer_processor"
    ]
    if len(tokenizer_steps) != 1:
        raise ValueError(f"checkpoint必须包含一个tokenizer_processor，实际={len(tokenizer_steps)}")
    config = dict(tokenizer_steps[0].get("config", {}))
    required = {"tokenizer_name", "max_length", "padding_side", "padding", "truncation"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"tokenizer_processor缺少配置: {missing}")
    return config


def load_preprocessor(checkpoint: Path) -> tuple[Any, Any, dict[str, Any]]:
    """仅加载CPU预处理器和Tokenizer，不加载策略模型权重。

    Args:
        checkpoint: 完整pretrained_model目录。

    Returns:
        checkpoint预处理器、对应Tokenizer及Tokenizer配置。
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from transformers import AutoTokenizer

    tokenizer_settings = load_tokenizer_settings(checkpoint)
    config = PreTrainedConfig.from_pretrained(
        str(checkpoint),
        cli_overrides=["--device=cpu", "--push_to_hub=false"],
    )
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_settings["tokenizer_name"],
        local_files_only=True,
        padding_side=tokenizer_settings["padding_side"],
    )
    return preprocessor, tokenizer, tokenizer_settings


def build_prompt_matrix() -> list[dict[str, str]]:
    """生成两种积木、两种底板和三种措辞的12条矩阵。

    Returns:
        包含任务标识、颜色、措辞和原始文本的有序记录。
    """
    matrix: list[dict[str, str]] = []
    for prompt_type in PROMPT_TYPES:
        for cube_color in CUBE_COLORS:
            for pad_color in PAD_COLORS:
                task_id = TASK_IDS[(cube_color, pad_color)]
                matrix.append(
                    {
                        "task_id": task_id,
                        "cube_color": cube_color,
                        "pad_color": pad_color,
                        "prompt_type": prompt_type,
                        "original_prompt": build_prompt(task_id, prompt_type),
                    }
                )
    return matrix


def process_prompt(
    sample: dict[str, str],
    preprocessor: Any,
    tokenizer: Any,
    tokenizer_settings: dict[str, Any],
) -> dict[str, Any]:
    """使用完整checkpoint预处理链路处理一条任务文本。

    Args:
        sample: 一条提示矩阵记录。
        preprocessor: checkpoint保存的真实预处理器。
        tokenizer: 与预处理器配置一致的Tokenizer。
        tokenizer_settings: 最大长度、padding和截断设置。

    Returns:
        原始文本、补换行文本、完整Token张量及有效Token信息。
    """
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    images = {
        "agent": np.zeros((256, 256, 3), dtype=np.uint8),
        "wrist": np.zeros((256, 256, 3), dtype=np.uint8),
    }
    state = np.zeros(7, dtype=np.float32)
    original_prompt = sample["original_prompt"]
    newline_prompt = original_prompt if original_prompt.endswith("\n") else f"{original_prompt}\n"
    processed = preprocessor(make_policy_observation(images, state, original_prompt))
    token_ids = processed[OBS_LANGUAGE_TOKENS][0].detach().cpu().tolist()
    attention_mask = processed[OBS_LANGUAGE_ATTENTION_MASK][0].detach().cpu().bool().tolist()
    active_ids = [int(token_id) for token_id, active in zip(token_ids, attention_mask) if active]
    direct = tokenizer(
        [newline_prompt],
        max_length=int(tokenizer_settings["max_length"]),
        padding=tokenizer_settings["padding"],
        truncation=bool(tokenizer_settings["truncation"]),
        return_tensors="pt",
    )
    direct_ids = direct["input_ids"][0].tolist()
    direct_mask = direct["attention_mask"][0].bool().tolist()
    untruncated_ids = tokenizer(newline_prompt, truncation=False)["input_ids"]
    token_texts = tokenizer.convert_ids_to_tokens(token_ids)
    active_token_texts = tokenizer.convert_ids_to_tokens(active_ids)
    active_decoded_texts = [tokenizer.decode([token_id]).strip() for token_id in active_ids]
    return {
        **sample,
        "newline_prompt": newline_prompt,
        "max_length": int(tokenizer_settings["max_length"]),
        "token_ids": [int(value) for value in token_ids],
        "token_texts": token_texts,
        "attention_mask": [bool(value) for value in attention_mask],
        "active_token_ids": active_ids,
        "active_token_texts": active_token_texts,
        "active_decoded_texts": active_decoded_texts,
        "active_length": int(sum(attention_mask)),
        "untruncated_length": len(untruncated_ids),
        "truncated": len(untruncated_ids) > int(tokenizer_settings["max_length"]),
        "direct_tokenizer_matches_preprocessor": token_ids == direct_ids and attention_mask == direct_mask,
    }


def build_token_records(
    preprocessor: Any,
    tokenizer: Any,
    tokenizer_settings: dict[str, Any],
) -> list[dict[str, Any]]:
    """处理完整12条提示矩阵。

    Args:
        preprocessor: checkpoint真实预处理器。
        tokenizer: checkpoint指定Tokenizer。
        tokenizer_settings: checkpoint指定Tokenizer参数。

    Returns:
        12条完整Token诊断记录。
    """
    return [process_prompt(sample, preprocessor, tokenizer, tokenizer_settings) for sample in build_prompt_matrix()]


def compare_pair(
    left: dict[str, Any],
    right: dict[str, Any],
    pair_type: str,
    fixed_dimension: str,
    fixed_value: str,
    expected_left: str,
    expected_right: str,
) -> dict[str, Any]:
    """比较一组只改变一个颜色词的Token记录。

    Args:
        left: 红色或蓝色一侧记录。
        right: 绿色或黄色一侧记录。
        pair_type: ``cube_color``或``pad_color``。
        fixed_dimension: 配对中保持不变的颜色维度。
        fixed_value: 配对中保持不变的颜色值。
        expected_left: 左侧预期差异词。
        expected_right: 右侧预期差异词。

    Returns:
        序列、mask、差异位置、Token及通过判定。
    """
    left_ids = left["token_ids"]
    right_ids = right["token_ids"]
    if len(left_ids) != len(right_ids):
        raise ValueError("配对Token序列长度不一致")
    difference_positions = [index for index, pair in enumerate(zip(left_ids, right_ids)) if pair[0] != pair[1]]
    differences = []
    for index in difference_positions:
        differences.append(
            {
                "index": index,
                "left_id": left_ids[index],
                "right_id": right_ids[index],
                "left_token": left["token_texts"][index],
                "right_token": right["token_texts"][index],
                "left_decoded": left["active_decoded_texts"][index] if index < left["active_length"] else "",
                "right_decoded": right["active_decoded_texts"][index] if index < right["active_length"] else "",
            }
        )
    decoded_matches = (
        len(differences) == 1
        and differences[0]["left_decoded"].lower() == expected_left
        and differences[0]["right_decoded"].lower() == expected_right
    )
    masks_equal = left["attention_mask"] == right["attention_mask"]
    active_lengths_equal = left["active_length"] == right["active_length"]
    difference_is_active = bool(
        len(difference_positions) == 1
        and difference_positions[0] < left["active_length"]
        and difference_positions[0] < right["active_length"]
    )
    passed = bool(
        left_ids != right_ids
        and masks_equal
        and active_lengths_equal
        and len(difference_positions) == 1
        and difference_is_active
        and decoded_matches
        and not left["truncated"]
        and not right["truncated"]
        and left["direct_tokenizer_matches_preprocessor"]
        and right["direct_tokenizer_matches_preprocessor"]
    )
    return {
        "pair_type": pair_type,
        "prompt_type": left["prompt_type"],
        "fixed_dimension": fixed_dimension,
        "fixed_value": fixed_value,
        "left_task_id": left["task_id"],
        "right_task_id": right["task_id"],
        "left_prompt": left["original_prompt"],
        "right_prompt": right["original_prompt"],
        "sequence_equal": left_ids == right_ids,
        "attention_mask_equal": masks_equal,
        "left_active_length": left["active_length"],
        "right_active_length": right["active_length"],
        "difference_count": len(difference_positions),
        "difference_positions": difference_positions,
        "differences": differences,
        "active_hamming_distance": len(difference_positions),
        "active_hamming_rate": len(difference_positions) / max(left["active_length"], right["active_length"], 1),
        "difference_is_active": difference_is_active,
        "decoded_matches_expected_colors": decoded_matches,
        "passed": passed,
    }


def build_pairwise_comparisons(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """构造6组红绿配对和6组蓝黄控制配对。

    Args:
        records: 12条完整Token诊断记录。

    Returns:
        12组只改变一个颜色词的配对比较。
    """
    indexed = {
        (record["cube_color"], record["pad_color"], record["prompt_type"]): record for record in records
    }
    comparisons: list[dict[str, Any]] = []
    for prompt_type in PROMPT_TYPES:
        for pad_color in PAD_COLORS:
            comparisons.append(
                compare_pair(
                    indexed[("red", pad_color, prompt_type)],
                    indexed[("green", pad_color, prompt_type)],
                    pair_type="cube_color",
                    fixed_dimension="pad_color",
                    fixed_value=pad_color,
                    expected_left="red",
                    expected_right="green",
                )
            )
        for cube_color in CUBE_COLORS:
            comparisons.append(
                compare_pair(
                    indexed[(cube_color, "blue", prompt_type)],
                    indexed[(cube_color, "yellow", prompt_type)],
                    pair_type="pad_color",
                    fixed_dimension="cube_color",
                    fixed_value=cube_color,
                    expected_left="blue",
                    expected_right="yellow",
                )
            )
    return comparisons


def build_summary(records: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    """依据锁定标准生成语言输入阶段结论。

    Args:
        records: 12条Token记录。
        comparisons: 12组配对比较。

    Returns:
        总体通过状态、异常数和后续实验分流建议。
    """
    cube_pairs = [item for item in comparisons if item["pair_type"] == "cube_color"]
    pad_pairs = [item for item in comparisons if item["pair_type"] == "pad_color"]
    direct_mismatches = sum(not record["direct_tokenizer_matches_preprocessor"] for record in records)
    truncated_records = sum(bool(record["truncated"]) for record in records)
    passed = bool(
        len(records) == 12
        and len(cube_pairs) == 6
        and len(pad_pairs) == 6
        and all(item["passed"] for item in comparisons)
        and direct_mismatches == 0
        and truncated_records == 0
    )
    return {
        "status": "pass" if passed else "fail",
        "language_input_distinguishable": passed,
        "record_count": len(records),
        "comparison_count": len(comparisons),
        "cube_color_pairs": len(cube_pairs),
        "cube_color_pairs_passed": sum(bool(item["passed"]) for item in cube_pairs),
        "pad_color_control_pairs": len(pad_pairs),
        "pad_color_control_pairs_passed": sum(bool(item["passed"]) for item in pad_pairs),
        "direct_tokenizer_preprocessor_mismatches": direct_mismatches,
        "truncated_records": truncated_records,
        "unexpected_pair_differences": sum(not item["passed"] for item in comparisons),
        "conclusion": (
            "红绿与蓝黄颜色词经过真实preprocessor后均保持独立Token；输入处理阶段不是颜色不区分的原因。"
            if passed
            else "语言输入处理未满足颜色可区分性标准；应先修复prompt或preprocessor再进入模型内部诊断。"
        ),
        "next_step": (
            "固定图像、状态和Flow Matching噪声，比较红绿指令的VLM语言特征与action chunk差异。"
            if passed
            else "检查newline、tokenizer、padding、attention mask及颜色词截断。"
        ),
        "boundary": "Token不同只证明输入可区分，不证明冻结VLM或动作专家实际使用颜色信息。",
    }


def build_manifest(
    checkpoint: Path,
    tokenizer_settings: dict[str, Any],
    records: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    """构造语言诊断运行清单。

    Args:
        checkpoint: 完整pretrained_model目录。
        tokenizer_settings: checkpoint指定Tokenizer参数。
        records: 已生成的Token记录。
        comparisons: 已生成的配对结果。

    Returns:
        checkpoint、哈希、版本和实验规模清单。
    """
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_config_sha256": sha256_file(checkpoint / "config.json"),
        "preprocessor_sha256": sha256_file(checkpoint / "policy_preprocessor.json"),
        "tokenizer": tokenizer_settings,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "lerobot": importlib.metadata.version("lerobot"),
            "transformers": importlib.metadata.version("transformers"),
            "device": "cpu",
        },
        "model_weights_loaded": False,
        "mujoco_rollout_executed": False,
        "prompt_count": len(records),
        "pairwise_comparison_count": len(comparisons),
    }


def write_report(
    path: Path,
    summary: dict[str, Any],
    records: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> None:
    """写出人工可读的Markdown诊断报告。

    Args:
        path: 报告输出路径。
        summary: 总体判定。
        records: 12条Token记录。
        comparisons: 12组配对结果。
    """
    lines = [
        "# 红绿积木语言Token差异诊断",
        "",
        f"- 状态：`{summary['status']}`",
        f"- 红绿配对通过：{summary['cube_color_pairs_passed']}/{summary['cube_color_pairs']}",
        f"- 蓝黄控制配对通过：{summary['pad_color_control_pairs_passed']}/{summary['pad_color_control_pairs']}",
        f"- 截断记录：{summary['truncated_records']}",
        f"- Tokenizer与完整preprocessor不一致：{summary['direct_tokenizer_preprocessor_mismatches']}",
        "",
        f"结论：{summary['conclusion']}",
        "",
        f"下一步：{summary['next_step']}",
        "",
        f"边界：{summary['boundary']}",
        "",
        "## Token记录",
        "",
        "| 任务 | 措辞 | 有效长度 | 有效Token IDs | 有效Token文本 |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for record in records:
        lines.append(
            f"| {record['task_id']} | {record['prompt_type']} | {record['active_length']} | "
            f"`{record['active_token_ids']}` | `{record['active_token_texts']}` |"
        )
    lines.extend(
        [
            "",
            "## 配对比较",
            "",
            "| 类型 | 措辞 | 固定条件 | 差异位置 | 差异Token | 通过 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for comparison in comparisons:
        tokens = [f"{item['left_token']}→{item['right_token']}" for item in comparison["differences"]]
        lines.append(
            f"| {comparison['pair_type']} | {comparison['prompt_type']} | "
            f"{comparison['fixed_dimension']}={comparison['fixed_value']} | "
            f"{comparison['difference_positions']} | `{tokens}` | {comparison['passed']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    output_dir: Path,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    """写出manifest、JSONL、CSV、JSON和Markdown产物。

    Args:
        output_dir: 独立诊断目录。
        manifest: 运行清单。
        records: 12条Token记录。
        comparisons: 12组配对结果。
        summary: 总体诊断结论。
    """
    write_json(output_dir / "run_manifest.json", manifest)
    with (output_dir / "token_records.jsonl").open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    csv_rows = []
    for comparison in comparisons:
        row = dict(comparison)
        row["difference_positions"] = json.dumps(row["difference_positions"], ensure_ascii=False)
        row["differences"] = json.dumps(row["differences"], ensure_ascii=False)
        csv_rows.append(row)
    with (output_dir / "pairwise_comparison.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    write_json(output_dir / "summary.json", summary)
    write_report(output_dir / "report.md", summary, records, comparisons)


def main(argv: Sequence[str] | None = None) -> int:
    """运行完整语言Token差异诊断并返回通过状态。"""
    args = build_parser().parse_args(argv)
    checkpoint = find_pretrained_model(args.checkpoint)
    output_dir = resolve_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"诊断输出目录已存在且非空，请更换目录: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessor, tokenizer, tokenizer_settings = load_preprocessor(checkpoint)
    records = build_token_records(preprocessor, tokenizer, tokenizer_settings)
    comparisons = build_pairwise_comparisons(records)
    summary = build_summary(records, comparisons)
    manifest = build_manifest(checkpoint, tokenizer_settings, records, comparisons)
    write_outputs(output_dir, manifest, records, comparisons, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
