"""从现有评测 action_traces 分析 chunk 边界对轨迹抖动的影响。

背景：SmolVLA 是动作 chunk 策略，模型每 ``execution_horizon`` 步重新预测一段
动作。如果新旧 chunk 在衔接处不连续（重新预测的动作与旧 chunk 尾部动作发生
跳变），轨迹就会出现周期性抖动。本脚本不重跑评测，只读取已有 action_traces
JSONL，从三个层面量化 chunk 边界跳变：

1. 命令层 (executed_action)：发给 MuJoCo 的最终命令在 chunk 衔接处的跳变。
2. 模型层 (model_output)：策略原始输出的衔接跳变（排除 postprocessor）。
3. 物理层 (observation_state)：实际状态轨迹的步间位移（"看得见的抖动"）。

并给出 chunk 内位置分布（按 ``step % horizon`` 分组），直接展示跳变是否
集中在边界位置。

用法::

    python scripts/analyze_chunk_boundary_jitter.py [--eval-root outputs/eval]
        [--output-dir outputs/eval/chunk_boundary_analysis]

输出:
    summary_by_dir.csv     每个评测目录一行（各指标逐轨迹中位数）
    per_position.json      每个评测目录的 chunk 内位置分布
    example_timeline.csv   每个目录第一条成功轨迹的逐帧位移与边界标记
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_records(trace_path: Path) -> list[dict]:
    """读取一条 rollout 的 action trace。"""
    records = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"空 trace: {trace_path}")
    return records


def infer_horizon(records: list[dict]) -> int:
    """从 chunk_start 标记推断 execution_horizon。"""
    starts = [r["step"] for r in records if bool(r.get("chunk_start", False))]
    if not starts:
        return 0
    diffs = [b - a for a, b in zip(starts, starts[1:]) if b > a]
    return min(diffs) if diffs else starts[0]


def analyze_records(records: list[dict], horizon: int) -> dict:
    """计算一条 rollout 的 chunk 边界抖动指标。"""
    obs = np.asarray([r["observation_state"] for r in records], dtype=np.float64)
    exec_a = np.asarray([r["executed_action"] for r in records], dtype=np.float64)
    model = np.asarray([r["model_output"] for r in records], dtype=np.float64)
    n = len(records)
    chunk_start = np.asarray([bool(r.get("chunk_start", False)) for r in records])

    # 相邻步跳变；d[t] 对应 records[t] 与 records[t-1] 之差 (t=1..n-1)
    d_obs = np.linalg.norm(np.diff(obs, axis=0), axis=1)
    d_exec = np.linalg.norm(np.diff(exec_a, axis=0), axis=1)
    d_model = np.linalg.norm(np.diff(model, axis=0), axis=1)
    # 逐维平均绝对跳变（物理层）
    dim_obs = np.mean(np.abs(np.diff(obs, axis=0)), axis=0)  # (7,)

    # ---- 运动方向连续性 / 震荡 ----
    # 抖动本质是"方向突变/震荡"，位移幅度会被动力学过滤，但方向翻转不会。
    # 用相邻位移向量之间的方向余弦量化：cos 越接近 1 越平滑，为负说明方向反转。
    d_vec = np.diff(obs[:, :6], axis=0)                        # 位移向量 (n-1,6)
    norm = np.linalg.norm(d_vec, axis=1)
    unit = d_vec / (norm[:, None] + 1e-12)
    dir_cos = np.sum(unit[:-1] * unit[1:], axis=1)             # (n-2,), 相邻位移夹角
    # 边界衔接：跨 chunk 的方向突变 = 旧 chunk 尾动作位移 d[s-1] 与
    # 新 chunk 首动作位移 d[s] 的夹角（records[s] 是新 chunk 第一步）。
    start_indices = np.flatnonzero(chunk_start)
    boundary_mask = np.zeros(dir_cos.size, dtype=bool)
    for s in start_indices:
        if 1 <= s - 1 and s <= len(d_vec) - 1:
            boundary_mask[s - 1] = True
    bc, ic = dir_cos[boundary_mask], dir_cos[~boundary_mask]
    boundary_flip = float(np.mean(bc < 0)) if bc.size else 0.0
    interior_flip = float(np.mean(ic < 0)) if ic.size else 0.0

    t = np.arange(1, n)
    # 命令/模型衔接边界：records[t] 是新 chunk 第一步 (t>=1)
    cmd_boundary = chunk_start[1:]
    # 物理层定义 A（与 rollout.py 一致）：d_obs[t] 由执行旧 chunk 最后动作引起
    # （records[t-1] 是旧 chunk 最后一步，即 records[t] 为新 chunk 第一步）
    obs_boundary_A = chunk_start[1:]
    # 物理层定义 B：执行新 chunk 第一个动作的效果，即 d_obs[t] 中
    # records[t] 位于新 chunk 第二步（t-1 是 chunk_start）
    obs_boundary_B = chunk_start[:-1]

    def split(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return values[mask], values[~mask]

    def ratio(b: np.ndarray, i: np.ndarray) -> float:
        mb, mi = (float(np.mean(b)) if b.size else 0.0, float(np.mean(i)) if i.size else 0.0)
        return mb / mi if mi > 1e-12 else 0.0

    b_exec, i_exec = split(d_exec, cmd_boundary)
    b_model, i_model = split(d_model, cmd_boundary)
    b_obsA, i_obsA = split(d_obs, obs_boundary_A)
    b_obsB, i_obsB = split(d_obs, obs_boundary_B)

    # chunk 内位置分布：obs 按"执行动作在 chunk 内位置" (t-1)%horizon 分组，
    # 命令/模型按"衔接点位置" t%horizon 分组（0 为边界）。
    pos_obs: dict[str, float] = {}
    pos_exec: dict[str, float] = {}
    pos_model: dict[str, float] = {}
    if horizon > 0:
        for p in range(horizon):
            m_obs = (t - 1) % horizon == p
            m_cmd = t % horizon == p
            pos_obs[str(p)] = float(np.mean(d_obs[m_obs])) if np.any(m_obs) else 0.0
            pos_exec[str(p)] = float(np.mean(d_exec[m_cmd])) if np.any(m_cmd) else 0.0
            pos_model[str(p)] = float(np.mean(d_model[m_cmd])) if np.any(m_cmd) else 0.0

    return {
        "steps": n,
        "horizon": horizon,
        "boundary_count": int(np.count_nonzero(cmd_boundary)),
        "mean_d_obs": float(np.mean(d_obs)) if d_obs.size else 0.0,
        "p95_d_obs": float(np.percentile(d_obs, 95)) if d_obs.size else 0.0,
        "mean_d_exec": float(np.mean(d_exec)) if d_exec.size else 0.0,
        "p95_d_exec": float(np.percentile(d_exec, 95)) if d_exec.size else 0.0,
        "mean_d_model": float(np.mean(d_model)) if d_model.size else 0.0,
        # 命令层
        "mean_boundary_exec_jump": float(np.mean(b_exec)) if b_exec.size else 0.0,
        "mean_interior_exec_jump": float(np.mean(i_exec)) if i_exec.size else 0.0,
        "exec_jump_ratio": ratio(b_exec, i_exec),
        # 模型层
        "mean_boundary_model_jump": float(np.mean(b_model)) if b_model.size else 0.0,
        "mean_interior_model_jump": float(np.mean(i_model)) if i_model.size else 0.0,
        "model_jump_ratio": ratio(b_model, i_model),
        # 物理层 A（旧 chunk 最后动作的效果，与现有 motion_metrics 口径一致）
        "mean_boundary_obsA_jump": float(np.mean(b_obsA)) if b_obsA.size else 0.0,
        "mean_interior_obsA_jump": float(np.mean(i_obsA)) if i_obsA.size else 0.0,
        "obsA_jump_ratio": ratio(b_obsA, i_obsA),
        # 物理层 B（新 chunk 第一个动作的效果）
        "mean_boundary_obsB_jump": float(np.mean(b_obsB)) if b_obsB.size else 0.0,
        "mean_interior_obsB_jump": float(np.mean(i_obsB)) if i_obsB.size else 0.0,
        "obsB_jump_ratio": ratio(b_obsB, i_obsB),
        # 运动方向连续性 / 震荡
        "mean_dir_cos": float(np.mean(dir_cos)) if dir_cos.size else 0.0,
        "mean_boundary_dir_cos": float(np.mean(bc)) if bc.size else 0.0,
        "mean_interior_dir_cos": float(np.mean(ic)) if ic.size else 0.0,
        "dir_cos_ratio": (float(np.mean(bc) / np.mean(ic)) if bc.size and ic.size and float(np.mean(ic)) > 0.0 else 0.0),
        "boundary_dir_flip_fraction": boundary_flip,
        "interior_dir_flip_fraction": interior_flip,
        "dim_obs_mean_abs_delta": dim_obs.tolist(),
        "pos_obs": pos_obs,
        "pos_exec": pos_exec,
        "pos_model": pos_model,
        # 供示例时间线使用
        "d_obs": d_obs.tolist(),
        "chunk_start": chunk_start.tolist(),
    }


def load_manifest_horizon(dir_path: Path) -> int | None:
    """从 run_manifest.json 读取 execution_horizon。"""
    manifest_path = dir_path / "run_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return int(payload.get("execution_horizon", 0)) or None
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def load_rollout_success(run_dir: Path) -> dict[str, bool]:
    """从 rollouts.jsonl 读取 rollout_key 到 success 的映射。"""
    path = run_dir / "rollouts.jsonl"
    result: dict[str, bool] = {}
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        result[record.get("rollout_key", "")] = bool(record.get("success", False))
    return result


def wilcoxon_sign_test(differences: np.ndarray) -> tuple[float, int]:
    """对配对差值做 Wilcoxon 符号秩检验（纯 numpy 实现）。

    Returns:
        ``(p_value, n_nonzero)``；使用正态近似并做连续性修正。
    """
    d = np.asarray(differences, dtype=np.float64)
    d = d[np.isfinite(d) & (np.abs(d) > 1e-12)]
    n = d.size
    if n == 0:
        return 1.0, 0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1
    w = float(np.sum(ranks[d > 0]))
    mean_w = n * (n + 1) / 4.0
    std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (w - mean_w) / std_w
    p = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return p, n


def _normal_cdf(x: float) -> float:
    """标准正态累积分布函数近似。"""
    return 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))


def _erf(x: float) -> float:
    """误差函数近似（Abramowitz & Stegun 7.1.26）。"""
    sign = 1.0 if x >= 0.0 else -1.0
    ax = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    poly = t * (
        0.254829592
        + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
    )
    return sign * (1.0 - poly * np.exp(-ax * ax))


def per_dir_ratio_stats(per_rollout: list[dict], field: str) -> dict:
    """统计逐 rollout 边界/内部比值：>1 的比例、中位数和 Wilcoxon 符号秩检验。"""
    ratios = np.asarray([float(r[field]) for r in per_rollout if float(r[field]) > 0.0], dtype=np.float64)
    if ratios.size == 0:
        return {"n": 0, "ratio_gt1_fraction": 0.0, "median_ratio": 0.0, "wilcoxon_p": 1.0}
    boundary_field = {
        "exec_jump_ratio": "mean_boundary_exec_jump",
        "model_jump_ratio": "mean_boundary_model_jump",
        "obsA_jump_ratio": "mean_boundary_obsA_jump",
        "obsB_jump_ratio": "mean_boundary_obsB_jump",
    }[field]
    interior_field = boundary_field.replace("boundary", "interior")
    diffs = np.asarray(
        [float(r[boundary_field]) - float(r[interior_field]) for r in per_rollout],
        dtype=np.float64,
    )
    p_value, n_nonzero = wilcoxon_sign_test(diffs)
    return {
        "n": int(ratios.size),
        "ratio_gt1_fraction": float(np.mean(ratios > 1.0)),
        "median_ratio": float(np.median(ratios)),
        "wilcoxon_p": float(p_value),
        "wilcoxon_n_nonzero": n_nonzero,
    }



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="分析 chunk 边界对轨迹抖动的影响")
    parser.add_argument("--eval-root", type=Path, default=PROJECT_ROOT / "outputs" / "eval")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "eval" / "chunk_boundary_analysis")
    parser.add_argument("--min-rollouts", type=int, default=1, help="每个目录至少需要的 trace 数")
    args = parser.parse_args(argv)

    eval_root: Path = args.eval_root.resolve()
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dir_summaries: list[dict] = []
    per_position: dict[str, dict] = {}
    example_rows: list[dict] = []

    for trace_dir in sorted(eval_root.rglob("action_traces")):
        if not trace_dir.is_dir():
            continue
        run_dir = trace_dir.parent
        trace_files = sorted(trace_dir.glob("*.jsonl"))
        if not trace_files:
            continue
        manifest_horizon = load_manifest_horizon(run_dir)

        per_rollout: list[dict] = []
        horizons: list[int] = []
        for trace_path in trace_files:
            try:
                records = load_records(trace_path)
            except (OSError, ValueError) as exc:
                print(f"  [skip] {trace_path.name}: {exc}")
                continue
            horizon = manifest_horizon or infer_horizon(records)
            horizons.append(horizon)
            per_rollout.append(analyze_records(records, horizon))
        if len(per_rollout) < args.min_rollouts:
            continue

        # 目录级汇总：各指标逐轨迹中位数
        summary: dict = {
            "run_dir": str(run_dir.relative_to(eval_root)).replace("\\", "/"),
            "environment": "cube" if "cube" in str(run_dir) else ("mug" if "mug" in str(run_dir) else "?"),
            "rollouts": len(per_rollout),
            "horizon": int(np.median(horizons)) if horizons else 0,
        }
        fields = [
            "mean_d_obs", "p95_d_obs", "mean_d_exec", "p95_d_exec", "mean_d_model",
            "mean_boundary_exec_jump", "mean_interior_exec_jump", "exec_jump_ratio",
            "mean_boundary_model_jump", "mean_interior_model_jump", "model_jump_ratio",
            "mean_boundary_obsA_jump", "mean_interior_obsA_jump", "obsA_jump_ratio",
            "mean_boundary_obsB_jump", "mean_interior_obsB_jump", "obsB_jump_ratio",
            "mean_dir_cos", "mean_boundary_dir_cos", "mean_interior_dir_cos", "dir_cos_ratio",
            "boundary_dir_flip_fraction", "interior_dir_flip_fraction",
        ]
        for field in fields:
            values = [float(r[field]) for r in per_rollout]
            summary[field] = float(np.median(values)) if values else 0.0

        # 逐 rollout 边界/内部比值的统计检验
        success_map = load_rollout_success(run_dir)
        stats_fields = ["exec_jump_ratio", "model_jump_ratio", "obsA_jump_ratio", "obsB_jump_ratio"]
        for field in stats_fields:
            summary[f"{field}_stats"] = per_dir_ratio_stats(per_rollout, field)
        # 按成败分组的边界命令跳变（需要 rollout_key 关联）
        if success_map:
            grouped: dict[str, list[float]] = {"success": [], "failure": []}
            for trace_path, rec in zip(trace_files, per_rollout):
                key = load_records(trace_path)[0].get("rollout_key", "")
                bucket = "success" if success_map.get(key, False) else "failure"
                grouped[bucket].append(float(rec["mean_boundary_exec_jump"]))
            for bucket, values in grouped.items():
                summary[f"boundary_exec_jump_{bucket}_median"] = float(np.median(values)) if values else 0.0
                summary[f"boundary_exec_jump_{bucket}_n"] = len(values)

        # 位置分布：跨 rollout 平均
        pos_agg: dict[str, dict[str, float]] = {"obs": {}, "exec": {}, "model": {}}
        horizon = summary["horizon"]
        for layer in ("obs", "exec", "model"):
            for p in range(horizon):
                values = [float(r[f"pos_{layer}"][str(p)]) for r in per_rollout]
                pos_agg[layer][str(p)] = float(np.mean(values)) if values else 0.0
        per_position[summary["run_dir"]] = {"horizon": horizon, "rollouts": len(per_rollout), **pos_agg}

        # 示例时间线：第一个成功（或第一个）rollout
        sample = per_rollout[0]
        sample_records = load_records(trace_files[0])
        first_key = sample_records[0].get("rollout_key", trace_files[0].stem)
        for idx in range(1, len(sample["d_obs"])):
            example_rows.append({
                "run_dir": summary["run_dir"],
                "horizon": horizon,
                "rollout_key": first_key,
                "step": idx + 1,
                "chunk_start": bool(sample["chunk_start"][idx]),
                "d_obs": sample["d_obs"][idx],
            })
        dir_summaries.append(summary)
        print(f"  ok  {summary['run_dir']:<60} h={horizon:<3} rollouts={len(per_rollout)}")

    # 写出 CSV
    csv_path = output_dir / "summary_by_dir.csv"
    fieldnames = [
        "run_dir", "environment", "rollouts", "horizon",
        "mean_d_obs", "p95_d_obs", "mean_d_exec", "p95_d_exec", "mean_d_model",
        "mean_boundary_exec_jump", "mean_interior_exec_jump", "exec_jump_ratio",
        "mean_boundary_model_jump", "mean_interior_model_jump", "model_jump_ratio",
        "mean_boundary_obsA_jump", "mean_interior_obsA_jump", "obsA_jump_ratio",
        "mean_boundary_obsB_jump", "mean_interior_obsB_jump", "obsB_jump_ratio",
        "mean_dir_cos", "mean_boundary_dir_cos", "mean_interior_dir_cos", "dir_cos_ratio",
        "boundary_dir_flip_fraction", "interior_dir_flip_fraction",
    ]
    with (output_dir / "summary_by_dir.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(dir_summaries)

    with (output_dir / "per_position.json").open("w", encoding="utf-8") as f:
        json.dump(per_position, f, ensure_ascii=False, indent=2)
    with (output_dir / "example_timeline.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_dir", "horizon", "rollout_key", "step", "chunk_start", "d_obs"])
        writer.writeheader()
        writer.writerows(example_rows)

    # 统计检验与成败分组单独输出
    statistics: dict[str, dict] = {}
    for s in dir_summaries:
        stats_entry = {
            "environment": s["environment"],
            "rollouts": s["rollouts"],
            "horizon": s["horizon"],
        }
        for field in ("exec_jump_ratio", "model_jump_ratio", "obsA_jump_ratio", "obsB_jump_ratio"):
            stats_entry[field] = s.pop(f"{field}_stats", {})
        for key in list(s):
            if key.startswith("boundary_exec_jump_"):
                stats_entry[key] = s.pop(key)
        statistics[s["run_dir"]] = stats_entry
    with (output_dir / "hypothesis_stats.json").open("w", encoding="utf-8") as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)

    # 控制台摘要
    print("\n=== 目录级汇总（逐轨迹中位数）===")
    print(f"{'run_dir':<62}{'env':<5}{'n':<4}{'h':<4}{'mean_d_obs':>10}{'exec_ratio':>11}{'model_ratio':>12}{'obsA_ratio':>11}{'obsB_ratio':>11}")
    for s in sorted(dir_summaries, key=lambda x: (x["environment"], x["horizon"])):
        print(
            f"{s['run_dir']:<62}{s['environment']:<5}{s['rollouts']:<4}{s['horizon']:<4}"
            f"{s['mean_d_obs']:>10.5f}{s['exec_jump_ratio']:>11.3f}{s['model_jump_ratio']:>12.3f}"
            f"{s['obsA_jump_ratio']:>11.3f}{s['obsB_jump_ratio']:>11.3f}"
        )
    print(f"\n结果写入: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
