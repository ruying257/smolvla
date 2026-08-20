"""汇总 --chunk-blend 对照实验：跨 K 对比成功率与边界抖动指标。

读取 ``outputs/eval/chunk_blend`` 下每个 ``K*`` 评测目录的 ``rollouts.jsonl``
与 ``action_traces/``，逐 K 输出成功率和边界抖动指标（复用
``analyze_chunk_boundary_jitter`` 的逐轨迹计算），并生成对比 CSV 与中文报告。

用法::

    python scripts/summarize_chunk_blend.py \\
        [--chunk-blend-root outputs/eval/chunk_blend]

输出:
    summary_comparison.csv   每 K 一行的汇总表
    report.md                人工可读对照报告
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_chunk_boundary_jitter import analyze_records, load_manifest_horizon, load_records


def bootstrap_success_ci(scene_results: list[tuple[int, bool]], repeats: int = 10_000) -> list[float]:
    """按 scene 整组重采样计算成功率 95% 置信区间（与 evaluate 口径一致）。"""
    if not scene_results:
        return [0.0, 0.0]
    grouped: dict[int, list[bool]] = {}
    for scene, success in scene_results:
        grouped.setdefault(scene, []).append(success)
    scenes = np.asarray(sorted(grouped), dtype=np.int64)
    rng = np.random.default_rng(20260813)
    samples = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        chosen = rng.choice(scenes, size=len(scenes), replace=True)
        samples[index] = np.mean(
            [float(np.mean(grouped[int(scene)])) for scene in chosen]
        )
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def median(values: list[float]) -> float:
    arr = [float(v) for v in values]
    return float(np.median(arr)) if arr else 0.0


def collect_run(run_dir: Path) -> dict:
    """聚合一个 K 目录的成功率与边界抖动指标。"""
    summary: dict = {
        "run_dir": run_dir.name,
        "rollouts": 0,
        "success_rate": 0.0,
        "success_ci95": [0.0, 0.0],
        "median_steps": 0.0,
        "control_exceptions": 0,
        "gripper_excess_toggle": 0.0,
    }

    results = []
    scene_results: list[tuple[int, bool]] = []
    journal = run_dir / "rollouts.jsonl"
    if journal.is_file():
        for line in journal.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            scene_results.append((int(rec["scene_seed"]), bool(rec["success"])))
            if rec.get("failure_mode") == "control_exception":
                summary["control_exceptions"] += 1
            results.append(rec)
    summary["rollouts"] = len(results)
    if scene_results:
        summary["success_rate"] = float(np.mean([s for _, s in scene_results]))
        summary["success_ci95"] = bootstrap_success_ci(scene_results)
        steps = [int(r["steps"]) for r in results]
        summary["median_steps"] = float(np.median(steps)) if steps else 0.0

    # 边界抖动指标（逐 rollout）
    exec_ratio, model_ratio, dir_cos_ratio, dir_flip = [], [], [], []
    d_obs, p95_obs, gripper_excess = [], [], []
    horizon = load_manifest_horizon(run_dir)
    trace_dir = run_dir / "action_traces"
    if trace_dir.is_dir():
        for trace_path in sorted(trace_dir.glob("*.jsonl")):
            try:
                records = load_records(trace_path)
            except (OSError, ValueError):
                continue
            h = horizon or __import__(
                "scripts.analyze_chunk_boundary_jitter", fromlist=["infer_horizon"]
            ).infer_horizon(records)
            row = analyze_records(records, h)
            exec_ratio.append(float(row["exec_jump_ratio"]))
            model_ratio.append(float(row["model_jump_ratio"]))
            dir_cos_ratio.append(float(row["dir_cos_ratio"]))
            dir_flip.append(float(row["boundary_dir_flip_fraction"]))
            d_obs.append(float(row["mean_d_obs"]))
            p95_obs.append(float(row["p95_d_obs"]))
            # 夹爪多余切换（打开/闭合切换次数 - 2）
            gripper = np.asarray([float(r["executed_action"][6]) >= 0.5 for r in records])
            toggles = int(np.count_nonzero(gripper[1:] != gripper[:-1]))
            gripper_excess.append(float(max(0, toggles - 2)))

    summary["exec_jump_ratio"] = median(exec_ratio)
    summary["model_jump_ratio"] = median(model_ratio)
    summary["dir_cos_ratio"] = median(dir_cos_ratio)
    summary["boundary_dir_flip_fraction"] = median(dir_flip)
    summary["mean_d_obs"] = median(d_obs)
    summary["p95_d_obs"] = median(p95_obs)
    summary["gripper_excess_toggle"] = median(gripper_excess)
    summary["k"] = int(run_dir.name[1:]) if run_dir.name.startswith("K") and run_dir.name[1:].isdigit() else -1
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="汇总 chunk-blend 对照实验")
    parser.add_argument("--chunk-blend-root", type=Path, default=PROJECT_ROOT / "outputs" / "eval" / "chunk_blend")
    args = parser.parse_args(argv)
    root: Path = args.chunk_blend_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"chunk_blend 根目录不存在: {root}")

    runs = [
        collect_run(d)
        for d in sorted(root.iterdir())
        if d.is_dir() and d.name.startswith("K") and d.name[1:].isdigit()
    ]
    if not runs:
        raise RuntimeError(f"{root} 下没有找到 K* 评测目录")

    # 写 CSV
    fields = [
        "k", "run_dir", "rollouts", "success_rate", "success_ci95_low", "success_ci95_high",
        "median_steps", "exec_jump_ratio", "model_jump_ratio", "dir_cos_ratio",
        "boundary_dir_flip_fraction", "mean_d_obs", "p95_d_obs",
        "gripper_excess_toggle", "control_exceptions",
    ]
    with (root / "summary_comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(runs, key=lambda x: x["k"]):
            writer.writerow({
                "k": r["k"],
                "run_dir": r["run_dir"],
                "rollouts": r["rollouts"],
                "success_rate": f"{r['success_rate']:.4f}",
                "success_ci95_low": f"{r['success_ci95'][0]:.4f}",
                "success_ci95_high": f"{r['success_ci95'][1]:.4f}",
                "median_steps": f"{r['median_steps']:.1f}",
                "exec_jump_ratio": f"{r['exec_jump_ratio']:.4f}",
                "model_jump_ratio": f"{r['model_jump_ratio']:.4f}",
                "dir_cos_ratio": f"{r['dir_cos_ratio']:.4f}",
                "boundary_dir_flip_fraction": f"{r['boundary_dir_flip_fraction']:.4f}",
                "mean_d_obs": f"{r['mean_d_obs']:.6f}",
                "p95_d_obs": f"{r['p95_d_obs']:.6f}",
                "gripper_excess_toggle": f"{r['gripper_excess_toggle']:.1f}",
                "control_exceptions": r["control_exceptions"],
            })

    # 写报告
    lines = [
        "# Chunk 衔接平滑（--chunk-blend）对照实验结果",
        "",
        f"- 根目录：`{root.relative_to(PROJECT_ROOT)}`",
        "- 数据来源：各 `K*` 评测目录的 `rollouts.jsonl` 与 `action_traces/`",
        "",
        "## 成功率与完成步数",
        "",
        "| K | n | 成功率 | Bootstrap 95% CI | 完成步数中位数 | control_exception |",
        "| --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for r in sorted(runs, key=lambda x: x["k"]):
        lines.append(
            f"| K{r['k']} | {r['rollouts']} | {r['success_rate']:.3f} | "
            f"[{r['success_ci95'][0]:.3f}, {r['success_ci95'][1]:.3f}] | "
            f"{r['median_steps']:.0f} | {r['control_exceptions']} |"
        )

    lines += [
        "",
        "## 边界抖动指标（逐 rollout 中位数）",
        "",
        "| K | exec_jump_ratio | model_jump_ratio | dir_cos_ratio | boundary_dir_flip | mean_d_obs (rad/步) | p95_d_obs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in sorted(runs, key=lambda x: x["k"]):
        lines.append(
            f"| K{r['k']} | {r['exec_jump_ratio']:.3f} | {r['model_jump_ratio']:.3f} | "
            f"{r['dir_cos_ratio']:.3f} | {r['boundary_dir_flip_fraction']:.3f} | "
            f"{r['mean_d_obs']:.5f} | {r['p95_d_obs']:.5f} |"
        )

    lines += [
        "",
        "## 判读",
        "",
        "- `exec_jump_ratio` / `model_jump_ratio`：>1 表示边界跳变大于 chunk 内部；理想处理组应回落接近或低于 1.0。",
        "- `dir_cos_ratio`：边界与内部运动方向连续性比值；<1 表示边界方向被中断，处理组应回升接近 1.0。",
        "- `boundary_dir_flip_fraction`：边界处运动方向反转比例；K0 约 0.10，处理组应显著下降。",
        "- 成功率与完成步数用于监控 K 过大的副作用（动作被拖慢或偏离模型意图）。",
        "",
        "> 结论需结合完整 40 条矩阵；单条 rollout 或不足 12 条的目录不构成统计证据。",
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"汇总已写入: {root / 'summary_comparison.csv'} 与 {root / 'report.md'}")
    print(f"{'K':<4}{'n':<5}{'succ':>6}{'exec_r':>8}{'model_r':>8}{'dir_r':>8}{'flip':>7}")
    for r in sorted(runs, key=lambda x: x["k"]):
        print(
            f"K{r['k']:<3}{r['rollouts']:<5}{r['success_rate']:>6.3f}{r['exec_jump_ratio']:>8.2f}"
            f"{r['model_jump_ratio']:>8.2f}{r['dir_cos_ratio']:>8.2f}{r['boundary_dir_flip_fraction']:>7.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
