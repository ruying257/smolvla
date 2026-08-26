"""比较同一 checkpoint 的基线组与联合控制组闭环评测结果。

脚本按 ``rollout_key`` 严格配对两个完整评测目录，验证 checkpoint、实验矩阵和
控制开关，并输出成功率差值的 scene 聚类配对 Bootstrap 区间、成败转移、分 seed
统计、轨迹平滑度和 motion limiter 触发率。

用法::

    python -m scripts.compare_control_stack \
        --baseline outputs/eval/s12000_unseen_multiseed_baseline \
        --controlled outputs/eval/s12000_unseen_multiseed_k4_limiter \
        --output outputs/eval/s12000_control_stack_comparison
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from scripts.analyze_chunk_boundary_jitter import analyze_records, load_records
from scripts.analyze_multi_seed import (
    SCENE_BOOTSTRAP_REPEATS,
    SCENE_BOOTSTRAP_SEED,
    attribute_rates,
    cross_seed_summary,
    per_seed_stats,
    scene_bootstrap_ci,
    seed_sensitivity,
)


PAIR_FIELDS = (
    "rollout_key",
    "scene_seed",
    "policy_seed",
    "task_id",
    "prompt_type",
    "baseline_success",
    "controlled_success",
    "transition",
    "baseline_failure_mode",
    "controlled_failure_mode",
    "baseline_steps",
    "controlled_steps",
)

MOTION_FIELDS = (
    "mean_delta_q_rad",
    "p95_delta_q_rad",
    "mean_delta2_q_rad",
    "p95_delta2_q_rad",
    "mean_ee_speed_m_s",
    "p95_ee_speed_m_s",
    "mean_ee_jerk_m_s3",
    "p95_ee_jerk_m_s3",
    "mean_chunk_boundary_jump_rad",
    "p95_chunk_boundary_jump_rad",
    "chunk_boundary_jump_ratio",
    "gripper_excess_toggle_count",
)

TRACE_FIELDS = (
    "exec_jump_ratio",
    "model_jump_ratio",
    "boundary_dir_flip_fraction",
    "mean_d_obs",
    "p95_d_obs",
)


def build_parser() -> argparse.ArgumentParser:
    """创建命令行解析器。"""
    parser = argparse.ArgumentParser(description="s12000 K0基线与K4+limiter联合控制组配对分析")
    parser.add_argument("--baseline", type=Path, required=True, help="K0且limiter关闭的评测目录")
    parser.add_argument("--controlled", type=Path, required=True, help="K4且limiter开启的评测目录")
    parser.add_argument("--output", type=Path, required=True, help="配对分析输出目录")
    parser.add_argument("--expected-pairs", type=int, default=120, help="期望的配对条件数")
    parser.add_argument("--repeats", type=int, default=SCENE_BOOTSTRAP_REPEATS, help="Bootstrap重复次数")
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=SCENE_BOOTSTRAP_SEED,
        help="Bootstrap随机种子",
    )
    return parser


def _as_bool(value: Any) -> bool:
    """把 CSV 或 JSON 中的真假值解析为布尔值。"""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", ""}:
        return False
    raise ValueError(f"无法解析布尔值: {value!r}")


def load_manifest(run_dir: Path) -> dict[str, Any]:
    """读取并校验评测 manifest。"""
    path = run_dir / "run_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"缺少run_manifest.json: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"manifest根节点必须是映射: {path}")
    return payload


def load_rollouts(run_dir: Path) -> dict[str, dict[str, Any]]:
    """读取 rollouts.csv，并按唯一 rollout_key 建立索引。"""
    path = run_dir / "rollouts.csv"
    if not path.is_file():
        raise FileNotFoundError(f"缺少rollouts.csv: {path}")
    output: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {
            "rollout_key",
            "scene_seed",
            "policy_seed",
            "task_id",
            "prompt_type",
            "success",
            "failure_mode",
            "steps",
            "checkpoint_sha256",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}缺少列: {sorted(missing)}")
        for line_number, raw in enumerate(reader, start=2):
            key = str(raw["rollout_key"])
            if not key:
                raise ValueError(f"{path}第{line_number}行rollout_key为空")
            if key in output:
                raise ValueError(f"{path}包含重复rollout_key: {key}")
            row: dict[str, Any] = dict(raw)
            row["scene_seed"] = int(row["scene_seed"])
            row["policy_seed"] = int(row["policy_seed"])
            row["success"] = _as_bool(row["success"])
            row["steps"] = int(row["steps"])
            output[key] = row
    if not output:
        raise ValueError(f"rollouts.csv为空: {path}")
    return output


def validate_runs(
    baseline_manifest: dict[str, Any],
    controlled_manifest: dict[str, Any],
    baseline_rows: dict[str, dict[str, Any]],
    controlled_rows: dict[str, dict[str, Any]],
    expected_pairs: int,
) -> None:
    """验证两组 checkpoint、矩阵、开关和完成状态满足配对实验契约。"""
    if expected_pairs <= 0:
        raise ValueError("expected-pairs必须大于零")
    if len(baseline_rows) != expected_pairs or len(controlled_rows) != expected_pairs:
        raise ValueError(
            f"两组必须各有{expected_pairs}条结果，实际为"
            f"baseline={len(baseline_rows)}、controlled={len(controlled_rows)}"
        )
    baseline_keys = set(baseline_rows)
    controlled_keys = set(controlled_rows)
    if baseline_keys != controlled_keys:
        missing = sorted(baseline_keys - controlled_keys)
        extra = sorted(controlled_keys - baseline_keys)
        raise ValueError(f"两组rollout_key不一致，controlled缺少={missing[:5]}，额外={extra[:5]}")

    checkpoint_hash = str(baseline_manifest.get("checkpoint_sha256", ""))
    if not checkpoint_hash or checkpoint_hash != str(controlled_manifest.get("checkpoint_sha256", "")):
        raise ValueError("两组checkpoint SHA-256不一致或缺失")
    for label, rows in (("baseline", baseline_rows), ("controlled", controlled_rows)):
        row_hashes = {str(row["checkpoint_sha256"]) for row in rows.values()}
        if row_hashes != {checkpoint_hash}:
            raise ValueError(f"{label}结果中的checkpoint哈希与manifest不一致: {sorted(row_hashes)}")

    shared_fields = (
        "source_sha256",
        "fps",
        "max_steps",
        "chunk_size",
        "execution_horizon",
        "appearance_variant",
        "appearance_texture_sha256",
    )
    for field in shared_fields:
        if baseline_manifest.get(field) != controlled_manifest.get(field):
            raise ValueError(f"两组manifest字段不一致: {field}")
    if int(baseline_manifest.get("chunk_blend", -1)) != 0:
        raise ValueError("baseline必须记录chunk_blend=0")
    if int(controlled_manifest.get("chunk_blend", -1)) != 4:
        raise ValueError("controlled必须记录chunk_blend=4")
    if bool((baseline_manifest.get("motion_limiter") or {}).get("enabled", False)):
        raise ValueError("baseline必须关闭motion limiter")
    if not bool((controlled_manifest.get("motion_limiter") or {}).get("enabled", False)):
        raise ValueError("controlled必须启用motion limiter")

    for label, rows in (("baseline", baseline_rows), ("controlled", controlled_rows)):
        exceptions = [key for key, row in rows.items() if row.get("failure_mode") == "control_exception" or row.get("error")]
        if exceptions:
            raise ValueError(f"{label}仍有控制异常或error，应先resume修复: {exceptions[:5]}")


def build_pairs(
    baseline_rows: dict[str, dict[str, Any]],
    controlled_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """按 rollout_key 生成有序配对行及成败转移类别。"""
    pairs: list[dict[str, Any]] = []
    for key in sorted(baseline_rows):
        baseline = baseline_rows[key]
        controlled = controlled_rows[key]
        for field in ("scene_seed", "policy_seed", "task_id", "prompt_type"):
            if baseline[field] != controlled[field]:
                raise ValueError(f"配对条件字段不一致: key={key}, field={field}")
        baseline_success = bool(baseline["success"])
        controlled_success = bool(controlled["success"])
        if not baseline_success and controlled_success:
            transition = "improved"
        elif baseline_success and not controlled_success:
            transition = "regressed"
        elif baseline_success:
            transition = "both_success"
        else:
            transition = "both_failure"
        pairs.append(
            {
                "rollout_key": key,
                "scene_seed": int(baseline["scene_seed"]),
                "policy_seed": int(baseline["policy_seed"]),
                "task_id": str(baseline["task_id"]),
                "prompt_type": str(baseline["prompt_type"]),
                "baseline_success": baseline_success,
                "controlled_success": controlled_success,
                "transition": transition,
                "baseline_failure_mode": str(baseline["failure_mode"]),
                "controlled_failure_mode": str(controlled["failure_mode"]),
                "baseline_steps": int(baseline["steps"]),
                "controlled_steps": int(controlled["steps"]),
            }
        )
    return pairs


def paired_scene_bootstrap_ci(
    pairs: list[dict[str, Any]],
    repeats: int = SCENE_BOOTSTRAP_REPEATS,
    bootstrap_seed: int = SCENE_BOOTSTRAP_SEED,
) -> list[float]:
    """按 scene 整组重采样，计算 controlled-baseline 成功率差值区间。"""
    if repeats <= 0:
        raise ValueError("Bootstrap repeats必须大于零")
    grouped: dict[int, list[float]] = defaultdict(list)
    for pair in pairs:
        difference = float(pair["controlled_success"]) - float(pair["baseline_success"])
        grouped[int(pair["scene_seed"])].append(difference)
    if not grouped:
        return [0.0, 0.0]
    scenes = np.asarray(sorted(grouped), dtype=np.int64)
    arrays = {scene: np.asarray(grouped[int(scene)], dtype=np.float64) for scene in scenes}
    rng = np.random.default_rng(bootstrap_seed)
    samples = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        chosen = rng.choice(scenes, size=len(scenes), replace=True)
        samples[index] = float(np.concatenate([arrays[int(scene)] for scene in chosen]).mean())
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def summarize_group(
    rows_by_key: dict[str, dict[str, Any]],
    repeats: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """计算单组总体、多 seed、分任务和失败模式统计。"""
    rows = list(rows_by_key.values())
    successes = sum(bool(row["success"]) for row in rows)
    per_seed = per_seed_stats(rows, repeats, bootstrap_seed)
    return {
        "rollouts": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "ci95_scene_bootstrap": scene_bootstrap_ci(rows, repeats, bootstrap_seed),
        "per_policy_seed": per_seed,
        "cross_seed": cross_seed_summary(per_seed),
        "seed_sensitivity": seed_sensitivity(rows),
        "by_task": attribute_rates(rows, "task_id"),
        "failure_modes": dict(sorted(Counter(str(row["failure_mode"]) for row in rows).items())),
        "steps_median": float(np.median([int(row["steps"]) for row in rows])),
    }


def _load_metric_rows(run_dir: Path) -> dict[str, dict[str, str]]:
    """读取评测自动生成的逐 rollout 运动指标。"""
    path = run_dir / "motion_metrics_by_rollout.csv"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return {str(row["rollout_key"]): row for row in rows}


def summarize_motion_metrics(
    baseline_dir: Path,
    controlled_dir: Path,
    keys: set[str],
) -> dict[str, Any]:
    """比较评测自动生成的运动指标中位数和逐条件配对差值。"""
    baseline = _load_metric_rows(baseline_dir)
    controlled = _load_metric_rows(controlled_dir)
    if set(baseline) != keys or set(controlled) != keys:
        return {"available": False, "reason": "motion_metrics_by_rollout.csv缺失或键集合不完整"}
    metrics: dict[str, Any] = {}
    for field in MOTION_FIELDS:
        baseline_values = np.asarray([float(baseline[key][field]) for key in sorted(keys)], dtype=np.float64)
        controlled_values = np.asarray([float(controlled[key][field]) for key in sorted(keys)], dtype=np.float64)
        metrics[field] = {
            "baseline_median": float(np.median(baseline_values)),
            "controlled_median": float(np.median(controlled_values)),
            "paired_difference_median": float(np.median(controlled_values - baseline_values)),
        }
    return {"available": True, "metrics": metrics}


def _trace_path(run_dir: Path, row: dict[str, Any]) -> Path:
    """从结果记录解析本机 action trace 路径，并兼容跨机器绝对路径。"""
    raw = str(row.get("action_trace_path", ""))
    name = re.split(r"[\\/]", raw)[-1]
    return run_dir / "action_traces" / name


def summarize_trace_metrics(run_dir: Path, rows_by_key: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """汇总轨迹抖动指标和 motion limiter 实际触发率。"""
    collected: dict[str, list[float]] = defaultdict(list)
    limited_steps = 0
    trace_steps = 0
    for row in rows_by_key.values():
        path = _trace_path(run_dir, row)
        if not path.is_file():
            return {"available": False, "reason": f"缺少action trace: {path.name}"}
        records = load_records(path)
        analyzed = analyze_records(records, 25)
        for field in TRACE_FIELDS:
            collected[field].append(float(analyzed[field]))
        for record in records:
            mask = list(record.get("motion_limited_mask", []))[:6]
            limited_steps += int(any(bool(value) for value in mask))
            trace_steps += 1
    return {
        "available": True,
        "rollouts": len(rows_by_key),
        "trace_steps": trace_steps,
        "motion_limited_steps": limited_steps,
        "motion_limited_step_rate": limited_steps / trace_steps if trace_steps else 0.0,
        "metric_medians": {field: float(np.median(values)) for field, values in collected.items()},
    }


def write_pairs_csv(path: Path, pairs: list[dict[str, Any]]) -> None:
    """写出逐条件配对明细 CSV。"""
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        writer.writerows(pairs)


def _fmt_rate(value: float) -> str:
    """把比例格式化为两位百分数。"""
    return f"{value:.2%}"


def write_report(path: Path, summary: dict[str, Any]) -> None:
    """生成中文配对实验 Markdown 报告。"""
    baseline = summary["groups"]["baseline"]
    controlled = summary["groups"]["controlled"]
    paired = summary["paired_comparison"]
    ci = paired["delta_ci95_scene_bootstrap"]
    conclusion = (
        "差值区间未跨越0，可报告联合控制组在本实验矩阵上的成功率存在明确变化。"
        if ci[0] > 0.0 or ci[1] < 0.0
        else "差值区间跨越0，只报告观察到的差值，不宣称联合控制带来确定提升。"
    )
    lines = [
        "# s12000 控制后处理双组配对实验报告",
        "",
        f"- Checkpoint SHA-256：`{summary['checkpoint_sha256']}`",
        f"- 配对条件：{summary['pairs']}（20 scene × 2 task × 3 policy seed）",
        f"- Bootstrap：B={summary['bootstrap']['repeats']}，rng={summary['bootstrap']['seed']}，聚类单元=scene_seed",
        "",
        "## 成功率",
        "",
        "| 组别 | 成功数/总数 | 成功率 | Scene Bootstrap 95% CI |",
        "| --- | ---: | ---: | ---: |",
        f"| A：K0、limiter关闭 | {baseline['successes']}/{baseline['rollouts']} | {_fmt_rate(baseline['success_rate'])} | [{_fmt_rate(baseline['ci95_scene_bootstrap'][0])}, {_fmt_rate(baseline['ci95_scene_bootstrap'][1])}] |",
        f"| B：K4、p99×1.1 limiter | {controlled['successes']}/{controlled['rollouts']} | {_fmt_rate(controlled['success_rate'])} | [{_fmt_rate(controlled['ci95_scene_bootstrap'][0])}, {_fmt_rate(controlled['ci95_scene_bootstrap'][1])}] |",
        "",
        "## 配对差值",
        "",
        f"- B − A：{_fmt_rate(paired['success_rate_delta'])}",
        f"- Scene配对Bootstrap 95% CI：[{_fmt_rate(ci[0])}, {_fmt_rate(ci[1])}]",
        f"- 改善（A失败、B成功）：{paired['transitions']['improved']}",
        f"- 退化（A成功、B失败）：{paired['transitions']['regressed']}",
        f"- 两组都成功：{paired['transitions']['both_success']}",
        f"- 两组都失败：{paired['transitions']['both_failure']}",
        f"- 判读：{conclusion}",
        "",
        "## 分 Policy Seed",
        "",
        "| Seed | A成功率 | B成功率 | B−A |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for seed in sorted(baseline["per_policy_seed"], key=int):
        a_rate = baseline["per_policy_seed"][seed]["success_rate"]
        b_rate = controlled["per_policy_seed"][seed]["success_rate"]
        lines.append(f"| {seed} | {_fmt_rate(a_rate)} | {_fmt_rate(b_rate)} | {_fmt_rate(b_rate - a_rate)} |")
    lines.extend(["", "## 轨迹与限制器", ""])
    for label, title in (("baseline", "A组"), ("controlled", "B组")):
        trace = summary["trace_metrics"][label]
        if not trace.get("available"):
            lines.append(f"- {title}轨迹指标不可用：{trace.get('reason', '未知原因')}")
            continue
        medians = trace["metric_medians"]
        lines.append(
            f"- {title}：exec jump ratio={medians['exec_jump_ratio']:.3f}，"
            f"边界方向翻转率={medians['boundary_dir_flip_fraction']:.3f}，"
            f"limiter触发步率={_fmt_rate(trace['motion_limited_step_rate'])}。"
        )
    lines.extend(
        [
            "",
            "## 结果边界",
            "",
            "结果衡量同一s12000 checkpoint在两种控制栈下的MuJoCo闭环表现。B组变化是K4与motion limiter的联合效果，不能从本实验拆分两者各自贡献，也不代表真实UR10e成功率。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_runs(
    baseline_dir: Path,
    controlled_dir: Path,
    output_dir: Path,
    expected_pairs: int = 120,
    repeats: int = SCENE_BOOTSTRAP_REPEATS,
    bootstrap_seed: int = SCENE_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """验证、比较两个评测目录并写出全部分析产物。"""
    baseline_dir = baseline_dir.resolve()
    controlled_dir = controlled_dir.resolve()
    output_dir = output_dir.resolve()
    baseline_manifest = load_manifest(baseline_dir)
    controlled_manifest = load_manifest(controlled_dir)
    baseline_rows = load_rollouts(baseline_dir)
    controlled_rows = load_rollouts(controlled_dir)
    validate_runs(
        baseline_manifest,
        controlled_manifest,
        baseline_rows,
        controlled_rows,
        expected_pairs,
    )
    pairs = build_pairs(baseline_rows, controlled_rows)
    baseline_summary = summarize_group(baseline_rows, repeats, bootstrap_seed)
    controlled_summary = summarize_group(controlled_rows, repeats, bootstrap_seed)
    transitions = Counter(str(pair["transition"]) for pair in pairs)
    paired_summary = {
        "success_rate_delta": controlled_summary["success_rate"] - baseline_summary["success_rate"],
        "delta_ci95_scene_bootstrap": paired_scene_bootstrap_ci(pairs, repeats, bootstrap_seed),
        "transitions": {
            name: int(transitions.get(name, 0))
            for name in ("improved", "regressed", "both_success", "both_failure")
        },
    }
    summary = {
        "schema_version": 1,
        "baseline_dir": str(baseline_dir),
        "controlled_dir": str(controlled_dir),
        "checkpoint_sha256": str(baseline_manifest["checkpoint_sha256"]),
        "pairs": len(pairs),
        "bootstrap": {"repeats": repeats, "seed": bootstrap_seed, "cluster": "scene_seed"},
        "groups": {"baseline": baseline_summary, "controlled": controlled_summary},
        "paired_comparison": paired_summary,
        "motion_metrics": summarize_motion_metrics(baseline_dir, controlled_dir, set(baseline_rows)),
        "trace_metrics": {
            "baseline": summarize_trace_metrics(baseline_dir, baseline_rows),
            "controlled": summarize_trace_metrics(controlled_dir, controlled_rows),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_pairs_csv(output_dir / "paired_rollouts.csv", pairs)
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(output_dir / "comparison_report.md", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """运行双组配对分析。"""
    args = build_parser().parse_args(argv)
    summary = compare_runs(
        args.baseline,
        args.controlled,
        args.output,
        expected_pairs=args.expected_pairs,
        repeats=args.repeats,
        bootstrap_seed=args.bootstrap_seed,
    )
    paired = summary["paired_comparison"]
    print(f"配对条件: {summary['pairs']}")
    print(f"B-A成功率差值: {paired['success_rate_delta']:.2%}")
    print(f"输出目录: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
