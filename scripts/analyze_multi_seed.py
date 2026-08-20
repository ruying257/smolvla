"""多 policy_seed 评测结果的跨 seed 聚合与 Bootstrap 置信区间分析。

读取 ``evaluate`` 闭环评测输出的 ``rollouts.csv``（每行含 scene_seed、policy_seed、
task_id、prompt_type、success 等列），输出：

- per-seed 成功率表（每个 policy seed 单独的成功率及其 scene 分层 Bootstrap 95% CI）；
- 总体成功率及 scene 分层 cluster Bootstrap 95% CI（与 ``evaluate.rollout.bootstrap_success_ci``
  口径一致：以 scene_seed 为聚类单元整组重采样，B=10000，rng 固定 20260813）；
- 跨 seed 聚合（均值 ± 样本标准差、min/max、极差）；
- K-seed 敏感性统计（每个 (scene, task, prompt) 三元组在 N 个 seed 下的成功数分类）；
- 按任务、按措辞的跨 seed 汇总。

用法::

    python scripts/analyze_multi_seed.py --input outputs/eval/formal_020000_multiseed
    python scripts/analyze_multi_seed.py --csv outputs/eval/formal_020000_multiseed/rollouts.csv \
        --output outputs/eval/formal_020000_multiseed/analysis

本脚本只依赖标准库与 numpy，不加载 MuJoCo 或 LeRobot。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SCENE_BOOTSTRAP_REPEATS = 10_000
SCENE_BOOTSTRAP_SEED = 20260813


def build_parser() -> argparse.ArgumentParser:
    """创建命令行解析器。"""
    parser = argparse.ArgumentParser(description="多 policy_seed 评测结果的跨 seed 聚合分析")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input",
        type=Path,
        help="评测输出目录（自动读取其中的 rollouts.csv）",
    )
    source.add_argument(
        "--csv",
        type=Path,
        help="rollouts.csv 文件路径（与 --input 二选一）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="分析产物输出目录；默认与输入 rollouts.csv 同目录下的 analysis 子目录",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=SCENE_BOOTSTRAP_REPEATS,
        help=f"Bootstrap 重采样次数（默认 {SCENE_BOOTSTRAP_REPEATS}）",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=SCENE_BOOTSTRAP_SEED,
        help=f"Bootstrap 随机种子（默认 {SCENE_BOOTSTRAP_SEED}，与评测报告口径一致）",
    )
    return parser


def _as_bool(value: Any) -> bool:
    """把 CSV 单元格中的 success 值解析为布尔值。"""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", ""}:
        return False
    raise ValueError(f"无法解析布尔值: {value!r}")


def load_rollouts(csv_path: Path) -> list[dict[str, Any]]:
    """读取 rollouts.csv 并解析关键列。"""
    if not csv_path.is_file():
        raise FileNotFoundError(f"rollouts.csv 不存在: {csv_path}")
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"scene_seed", "policy_seed", "success"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"rollouts.csv 缺少必需列: {sorted(missing)}")
        for line_number, raw in enumerate(reader, start=2):
            row: dict[str, Any] = dict(raw)
            try:
                row["scene_seed"] = int(row["scene_seed"])
                row["policy_seed"] = int(row["policy_seed"])
                row["success"] = _as_bool(row["success"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"rollouts.csv 第{line_number}行解析失败: {exc}") from exc
            rows.append(row)
    if not rows:
        raise ValueError(f"rollouts.csv 为空: {csv_path}")
    return rows


def scene_bootstrap_ci(
    rows: list[dict[str, Any]],
    repeats: int,
    bootstrap_seed: int,
) -> list[float]:
    """以 scene_seed 为聚类单元整组重采样，返回成功率 95% 置信区间。

    与 ``evaluate.rollout.bootstrap_success_ci`` 口径一致：每次重采样先按 scene
    整组抽取（可重复），再把选中 scene 的全部 rollout 合并求成功率。
    """
    grouped: dict[int, np.ndarray] = {}
    for row in rows:
        grouped.setdefault(int(row["scene_seed"]), []).append(1.0 if row["success"] else 0.0)
    scenes = sorted(grouped)
    if not scenes:
        return [0.0, 0.0]
    arrays = {scene: np.asarray(grouped[scene], dtype=np.float64) for scene in scenes}
    rng = np.random.default_rng(bootstrap_seed)
    samples = np.empty(repeats, dtype=np.float64)
    scene_array = np.asarray(scenes, dtype=np.int64)
    for index in range(repeats):
        chosen = rng.choice(scene_array, size=len(scenes), replace=True)
        samples[index] = np.concatenate([arrays[int(scene)] for scene in chosen]).mean()
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def _rate(successes: int, total: int) -> float:
    return successes / total if total else 0.0


def per_seed_stats(
    rows: list[dict[str, Any]],
    repeats: int,
    bootstrap_seed: int,
) -> dict[str, dict[str, Any]]:
    """按 policy_seed 分组计算成功率及各自 scene bootstrap CI。"""
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["policy_seed"])].append(row)
    output: dict[str, dict[str, Any]] = {}
    for seed in sorted(grouped):
        group = grouped[seed]
        successes = sum(1 for row in group if row["success"])
        output[str(seed)] = {
            "successes": successes,
            "rollouts": len(group),
            "success_rate": _rate(successes, len(group)),
            "ci95_scene_bootstrap": scene_bootstrap_ci(group, repeats, bootstrap_seed),
        }
    return output


def cross_seed_summary(per_seed: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """汇总 N 个 seed 成功率的均值、标准差、min/max 与极差。"""
    rates = [value["success_rate"] for value in per_seed.values()]
    if not rates:
        return {"seeds": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "range": 0.0}
    mean = float(np.mean(rates))
    std = float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0
    minimum = float(min(rates))
    maximum = float(max(rates))
    return {
        "seeds": len(rates),
        "mean": mean,
        "std": std,
        "min": minimum,
        "max": maximum,
        "range": maximum - minimum,
    }


def seed_sensitivity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """按 (scene, task, prompt) 三元组统计 N 个 seed 下的成功数并分类。

    - ``N/N`` 成功 -> 稳定成功（stable_success）；
    - ``0/N`` 成功 -> 稳定失败（stable_failure）；
    - 其余 -> seed 敏感（sampling_sensitive）。
    """
    grouped: dict[tuple[int, str, str], list[bool]] = defaultdict(list)
    for row in rows:
        key = (int(row["scene_seed"]), str(row.get("task_id", "")), str(row.get("prompt_type", "")))
        grouped[key].append(row["success"])
    counts = {"stable_success": 0, "sampling_sensitive": 0, "stable_failure": 0}
    incomplete = 0
    for successes in grouped.values():
        total = len(successes)
        if total < 2:
            incomplete += 1
            continue
        if all(successes):
            counts["stable_success"] += 1
        elif not any(successes):
            counts["stable_failure"] += 1
        else:
            counts["sampling_sensitive"] += 1
    return {
        "condition_groups": len(grouped),
        "incomplete_groups": incomplete,
        "stable_success": counts["stable_success"],
        "sampling_sensitive": counts["sampling_sensitive"],
        "stable_failure": counts["stable_failure"],
    }


def attribute_rates(rows: list[dict[str, Any]], attribute: str) -> dict[str, dict[str, Any]]:
    """按 CSV 中的某个分组列（task_id / prompt_type）计算聚合成功率。"""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(attribute, ""))].append(row)
    output: dict[str, dict[str, Any]] = {}
    for value in sorted(grouped):
        group = grouped[value]
        successes = sum(1 for row in group if row["success"])
        output[value] = {
            "successes": successes,
            "rollouts": len(group),
            "success_rate": _rate(successes, len(group)),
        }
    return output


def _fmt_rate(value: float) -> str:
    return f"{value:.2%}"


def _fmt_ci(ci: list[float]) -> str:
    return f"[{ci[0]:.2%}, {ci[1]:.2%}]"


def write_report(
    report_path: Path,
    rows: list[dict[str, Any]],
    overall: dict[str, Any],
    per_seed: dict[str, dict[str, Any]],
    cross: dict[str, Any],
    sensitivity: dict[str, Any],
    by_task: dict[str, dict[str, Any]],
    by_prompt: dict[str, dict[str, Any]],
    repeats: int,
    bootstrap_seed: int,
) -> None:
    """生成人工可读的多 seed 聚合 Markdown 报告。"""
    seeds = sorted(per_seed)
    lines = [
        "# SmolVLA 多 Seed 统计 + Bootstrap 置信区间报告",
        "",
        f"- Rollout：{overall['rollouts']}（{len(seeds)} 个 policy seed）",
        f"- 总体严格成功率：{_fmt_rate(overall['success_rate'])}",
        f"- Scene 分层 Bootstrap 95% CI：{_fmt_ci(overall['ci95_scene_bootstrap'])}",
        f"- 跨 seed 成功率：均值 {_fmt_rate(cross['mean'])} ± 标准差 {_fmt_rate(cross['std'])}，"
        f"范围 [{_fmt_rate(cross['min'])}, {_fmt_rate(cross['max'])}]，极差 {_fmt_rate(cross['range'])}",
        f"- Bootstrap：B={repeats}，rng={bootstrap_seed}，聚类单元=scene_seed",
        "",
        "## 分 policy seed 成功率",
        "",
        "| policy seed | 成功数 | 总数 | 成功率 | Scene Bootstrap 95% CI |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed in seeds:
        value = per_seed[seed]
        lines.append(
            f"| {seed} | {value['successes']} | {value['rollouts']} | "
            f"{_fmt_rate(value['success_rate'])} | {_fmt_ci(value['ci95_scene_bootstrap'])} |"
        )
    lines.extend(
        [
            "",
            "## 跨 seed 敏感性（按 scene × task × prompt 三元组）",
            "",
            f"- 稳定成功（N/N 成功）：{sensitivity['stable_success']} 组",
            f"- Seed 敏感（部分 seed 成功）：{sensitivity['sampling_sensitive']} 组",
            f"- 稳定失败（0/N 成功）：{sensitivity['stable_failure']} 组",
            f"- 不完整分组（rollout 数不足两个 seed）：{sensitivity['incomplete_groups']} 组",
            "",
            "## 分任务成功率（跨 seed 聚合）",
            "",
            "| 任务 | 成功数 | 总数 | 成功率 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for task_id, value in by_task.items():
        lines.append(f"| {task_id} | {value['successes']} | {value['rollouts']} | {_fmt_rate(value['success_rate'])} |")
    lines.extend(
        [
            "",
            "## 分措辞成功率（跨 seed 聚合）",
            "",
            "| 措辞 | 成功数 | 总数 | 成功率 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for prompt, value in by_prompt.items():
        lines.append(f"| {prompt} | {value['successes']} | {value['rollouts']} | {_fmt_rate(value['success_rate'])} |")
    lines.extend(
        [
            "",
            "## 结果边界",
            "",
            "结果仅代表本机 MuJoCo 仿真闭环能力，不代表真实 UR10e 成功率；"
            "Bootstrap 区间反映场景布局与 Flow Matching 采样随机性，不消除硬件 FP16 非确定性。",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """运行多 seed 聚合分析并写出 JSON 与 Markdown 报告。"""
    args = build_parser().parse_args(argv)
    csv_path = args.csv or (args.input / "rollouts.csv")
    if csv_path is None:
        raise ValueError("必须提供 --input 或 --csv")
    rows = load_rollouts(csv_path)

    overall_ci = scene_bootstrap_ci(rows, args.repeats, args.bootstrap_seed)
    overall = {
        "rollouts": len(rows),
        "successes": sum(1 for row in rows if row["success"]),
        "success_rate": _rate(sum(1 for row in rows if row["success"]), len(rows)),
        "ci95_scene_bootstrap": overall_ci,
    }
    per_seed = per_seed_stats(rows, args.repeats, args.bootstrap_seed)
    cross = cross_seed_summary(per_seed)
    sensitivity = seed_sensitivity(rows)
    by_task = attribute_rates(rows, "task_id")
    by_prompt = attribute_rates(rows, "prompt_type")

    summary: dict[str, Any] = {
        "schema_version": 1,
        "input_csv": str(csv_path.resolve()),
        "rollouts": overall["rollouts"],
        "overall": overall,
        "per_policy_seed": per_seed,
        "cross_seed": cross,
        "seed_sensitivity": sensitivity,
        "by_task": by_task,
        "by_prompt": by_prompt,
        "bootstrap": {"repeats": args.repeats, "seed": args.bootstrap_seed, "cluster": "scene_seed"},
    }

    output_dir = (args.output or csv_path.parent / "analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "multi_seed_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path = output_dir / "multi_seed_report.md"
    write_report(
        report_path,
        rows,
        overall,
        per_seed,
        cross,
        sensitivity,
        by_task,
        by_prompt,
        args.repeats,
        args.bootstrap_seed,
    )
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
