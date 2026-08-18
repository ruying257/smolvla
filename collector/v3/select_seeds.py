"""基于真实稳定杯子位置确定性筛选4×5覆盖seed。"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sim.mug_environment import (
    MUG_SAMPLE_X_RANGE,
    MUG_SAMPLE_Y_RANGE,
    MugTabletopEnv,
)


CANDIDATE_START = 0
CANDIDATE_STOP = 10_000
GRID_COLUMNS = 4
GRID_ROWS = 5
MIN_SELECTED_DISTANCE = 0.04
DEFAULT_JSON_PATH = Path("configs/mug_v3_seed_selection.json")
DEFAULT_CSV_PATH = Path("configs/mug_v3_seed_selection.csv")


@dataclass(frozen=True)
class SeedCandidate:
    """描述一个通过真实环境reset检查的候选seed。

    Attributes:
        seed: 传给 ``MugTabletopEnv.reset`` 的整数seed。
        x: 稳定后杯子中心世界坐标x。
        y: 稳定后杯子中心世界坐标y。
        column: 所属4列网格的零基索引。
        row: 所属5行网格的零基索引。
        distance_to_center: 到所属网格中心的欧氏距离。
        reason: 该候选通过合法性检查的机器可读原因。
    """

    seed: int
    x: float
    y: float
    column: int
    row: int
    distance_to_center: float
    reason: str


def _grid_index(value: float, limits: tuple[float, float], count: int) -> int:
    """把连续坐标稳定映射到有限网格索引。

    Args:
        value: 需要离散化的实际稳定坐标。
        limits: 工作区闭区间下界和上界。
        count: 等宽网格数量。

    Returns:
        位于 ``[0, count-1]`` 的网格索引。

    Raises:
        ValueError: 坐标不在工作区或网格数量无效时抛出。
    """
    low, high = limits
    if count <= 0 or not low <= value <= high:
        raise ValueError(f"坐标无法映射到网格: value={value}, limits={limits}, count={count}")
    normalized = (value - low) / (high - low)
    return min(count - 1, int(normalized * count))


def _cell_center(column: int, row: int) -> tuple[float, float]:
    """计算指定4×5网格单元的世界坐标中心。

    Args:
        column: 零基列索引。
        row: 零基行索引。

    Returns:
        ``(x, y)``形式的网格中心。
    """
    x_edges = np.linspace(*MUG_SAMPLE_X_RANGE, GRID_COLUMNS + 1)
    y_edges = np.linspace(*MUG_SAMPLE_Y_RANGE, GRID_ROWS + 1)
    return (
        float((x_edges[column] + x_edges[column + 1]) / 2.0),
        float((y_edges[row] + y_edges[row + 1]) / 2.0),
    )


def _legal_candidate(env: MugTabletopEnv, seed: int) -> tuple[SeedCandidate | None, str]:
    """重置真实环境并把合法稳定布局转换为候选记录。

    Args:
        env: 可重复使用的真实杯子MuJoCo环境。
        seed: 当前候选seed。

    Returns:
        合法时返回 ``(SeedCandidate, "")``，否则返回 ``(None, reason)``。
    """
    try:
        snapshot = env.reset(seed)
    except (RuntimeError, ValueError) as exc:
        return None, f"reset_failed:{type(exc).__name__}"
    pose = np.asarray(snapshot.mug_initial_pose, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        return None, "non_finite_pose"
    if not np.isfinite(env.get_state()).all() or not np.isfinite(env.data.qpos).all():
        return None, "non_finite_state"
    if env._has_mug_robot_contact():
        return None, "mug_robot_contact"
    if env._too_close_to_pad(pose[:2]):
        return None, "pad_overlap"
    x, y = float(pose[0]), float(pose[1])
    if not (MUG_SAMPLE_X_RANGE[0] <= x <= MUG_SAMPLE_X_RANGE[1]):
        return None, "x_out_of_workspace"
    if not (MUG_SAMPLE_Y_RANGE[0] <= y <= MUG_SAMPLE_Y_RANGE[1]):
        return None, "y_out_of_workspace"
    column = _grid_index(x, MUG_SAMPLE_X_RANGE, GRID_COLUMNS)
    row = _grid_index(y, MUG_SAMPLE_Y_RANGE, GRID_ROWS)
    center_x, center_y = _cell_center(column, row)
    return SeedCandidate(
        seed=seed,
        x=x,
        y=y,
        column=column,
        row=row,
        distance_to_center=float(np.hypot(x - center_x, y - center_y)),
        reason="legal_reset_nearest_center_with_pairwise_clearance",
    ), ""


def scan_candidates(
    start: int = CANDIDATE_START,
    stop: int = CANDIDATE_STOP,
) -> tuple[dict[tuple[int, int], list[SeedCandidate]], dict[str, int]]:
    """扫描固定候选池并按真实稳定位置归入4×5网格。

    Args:
        start: 闭区间起始seed，正式配置固定为0。
        stop: 开区间结束seed，正式配置固定为10000。

    Returns:
        每个网格单元的合法候选列表，以及拒绝原因计数。

    Raises:
        ValueError: 扫描范围为空或越出锁定候选池时抛出。
    """
    if not CANDIDATE_START <= start < stop <= CANDIDATE_STOP:
        raise ValueError(f"候选范围必须位于[0,10000)，实际为[{start},{stop})")
    cells = {(column, row): [] for row in range(GRID_ROWS) for column in range(GRID_COLUMNS)}
    rejected: dict[str, int] = {}
    with MugTabletopEnv() as env:
        for seed in range(start, stop):
            candidate, reason = _legal_candidate(env, seed)
            if candidate is None:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            cells[(candidate.column, candidate.row)].append(candidate)
    for candidates in cells.values():
        candidates.sort(key=lambda item: (item.distance_to_center, item.seed))
    return cells, rejected


def choose_grid_seeds(
    cells: dict[tuple[int, int], list[SeedCandidate]],
    minimum_distance: float = MIN_SELECTED_DISTANCE,
) -> list[SeedCandidate]:
    """按行优先顺序为每格选择最近且满足全局间距的seed。

    每个单元的候选已经按到中心距离和seed排序。选择时只接受与此前
    已选位置至少相隔4厘米的最近候选，因此结果完全确定且兼顾中心性。

    Args:
        cells: ``scan_candidates``产生的20格候选集合。
        minimum_distance: 最终杯子中心最小两两距离。

    Returns:
        按行优先、列次序排列的20个选中候选。

    Raises:
        RuntimeError: 任一网格没有满足全局间距的合法候选时抛出。
    """
    selected: list[SeedCandidate] = []
    for row in range(GRID_ROWS):
        for column in range(GRID_COLUMNS):
            candidates = cells.get((column, row), [])
            choice = next(
                (
                    candidate
                    for candidate in candidates
                    if all(
                        np.hypot(candidate.x - previous.x, candidate.y - previous.y)
                        >= minimum_distance
                        for previous in selected
                    )
                ),
                None,
            )
            if choice is None:
                raise RuntimeError(
                    f"网格({column},{row})没有满足{minimum_distance:.3f}m间距的候选"
                )
            selected.append(choice)
    return selected


def choose_pilot_seeds(selected: Sequence[SeedCandidate]) -> list[int]:
    """从最终布局中选择左右分离且非边界的两个中等难度pilot。

    选择固定内部单元 ``(1,1)`` 与 ``(2,3)``，使两个杯子分别处于y负侧
    和y正侧，同时避开最外行列与工作区极端边界。

    Args:
        selected: 已按网格顺序选出的20个候选。

    Returns:
        保持负y侧、正y侧顺序的两个pilot seed。

    Raises:
        RuntimeError: 最终候选缺少任一固定pilot单元时抛出。
    """
    by_cell = {(item.column, item.row): item for item in selected}
    try:
        return [by_cell[(1, 1)].seed, by_cell[(2, 3)].seed]
    except KeyError as exc:
        raise RuntimeError("最终20个seed缺少pilot所需内部网格") from exc


def write_selection_reports(
    selected: Sequence[SeedCandidate],
    rejected: dict[str, int],
    json_path: Path,
    csv_path: Path,
) -> None:
    """原子写出机器可读的seed筛选JSON和CSV报告。

    Args:
        selected: 最终20个候选。
        rejected: 扫描期间各拒绝原因数量。
        json_path: JSON报告路径。
        csv_path: CSV报告路径。
    """
    positions = np.asarray([(item.x, item.y) for item in selected], dtype=np.float64)
    distances = [
        float(np.linalg.norm(positions[left] - positions[right]))
        for left in range(len(positions))
        for right in range(left + 1, len(positions))
    ]
    payload: dict[str, Any] = {
        "candidate_pool": [CANDIDATE_START, CANDIDATE_STOP - 1],
        "grid": {"columns": GRID_COLUMNS, "rows": GRID_ROWS},
        "workspace": {"x": list(MUG_SAMPLE_X_RANGE), "y": list(MUG_SAMPLE_Y_RANGE)},
        "minimum_pairwise_distance": MIN_SELECTED_DISTANCE,
        "actual_minimum_pairwise_distance": min(distances),
        "selection_algorithm": "row_major_nearest_center_with_prior_pairwise_clearance",
        "scene_seeds": [item.seed for item in selected],
        "pilot_scene_seeds": choose_pilot_seeds(selected),
        "selected": [asdict(item) for item in selected],
        "rejected": rejected,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = json_path.with_suffix(json_path.suffix + ".tmp")
    temporary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_json.replace(json_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(selected[0])))
        writer.writeheader()
        writer.writerows(asdict(item) for item in selected)
    temporary_csv.replace(csv_path)


def verify_config_seeds(config_path: Path, selected: Sequence[SeedCandidate]) -> None:
    """验证锁定配置与重新筛选结果逐项一致。

    Args:
        config_path: 已固化V3配置路径。
        selected: 本次真实环境扫描得到的20个候选。

    Raises:
        ValueError: 配置不存在、结构错误或seed结果漂移时抛出。
    """
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected = [item.seed for item in selected]
    if config.get("scene_seeds") != expected:
        raise ValueError(f"配置scene_seeds与筛选结果不一致: expected={expected}")
    pilots = choose_pilot_seeds(selected)
    if config.get("pilot_scene_seeds") != pilots:
        raise ValueError(f"配置pilot_scene_seeds与筛选结果不一致: expected={pilots}")


def build_parser() -> argparse.ArgumentParser:
    """创建确定性seed筛选命令行解析器。

    Returns:
        包含报告输出与配置复核参数的解析器。
    """
    parser = argparse.ArgumentParser(description="筛选杯子V3的4×5固定scene seed")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--verify-config", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """扫描完整候选池、写出报告并可选复核锁定配置。

    Args:
        argv: 可选命令行参数；为空时读取当前进程参数。

    Returns:
        成功生成并验证结果时返回0。
    """
    args = build_parser().parse_args(argv)
    cells, rejected = scan_candidates()
    selected = choose_grid_seeds(cells)
    write_selection_reports(selected, rejected, args.output_json, args.output_csv)
    if args.verify_config is not None:
        verify_config_seeds(args.verify_config, selected)
    print(json.dumps({
        "scene_seeds": [item.seed for item in selected],
        "pilot_scene_seeds": choose_pilot_seeds(selected),
        "json": str(args.output_json),
        "csv": str(args.output_csv),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
