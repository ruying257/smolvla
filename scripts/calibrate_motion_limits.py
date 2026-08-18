"""从LeRobot专家数据标定UR10e关节速度和加速度限制。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np


JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="从LeRobot专家数据标定关节速度/加速度上限")
    parser.add_argument("--dataset-root", type=Path, required=True, help="LeRobot数据集根目录")
    parser.add_argument("--output", type=Path, required=True, help="锁定JSON输出路径")
    parser.add_argument("--quantile", type=float, default=0.99, help="绝对导数分位数，默认0.99")
    parser.add_argument("--margin", type=float, default=1.1, help="分位数安全裕量，默认1.1")
    return parser


def sha256_dataset(root: Path, parquet_paths: Sequence[Path]) -> str:
    """计算元数据和Parquet样本的稳定内容哈希。"""
    digest = hashlib.sha256()
    paths = [root / "meta" / "info.json", *parquet_paths]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"数据集标定所需文件不存在: {path}")
        digest.update(str(path.relative_to(root)).replace("\\", "/").encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_episode_actions(root: Path) -> tuple[int, list[np.ndarray], str, int]:
    """读取并按episode与帧序重建六维专家动作轨迹。"""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("标定需要pyarrow，请使用项目评测环境运行") from exc

    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"缺少LeRobot元数据: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    fps = int(info.get("fps", 0))
    if fps <= 0:
        raise ValueError("meta/info.json中的fps必须为正数")
    paths = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"数据集不存在Parquet样本: {root / 'data'}")
    episodes: dict[int, list[tuple[int, np.ndarray]]] = {}
    for path in paths:
        table = pq.read_table(path, columns=["action", "episode_index", "frame_index"])
        for action, episode_index, frame_index in zip(
            table.column("action").to_pylist(),
            table.column("episode_index").to_pylist(),
            table.column("frame_index").to_pylist(),
            strict=True,
        ):
            vector = np.asarray(action, dtype=np.float64)
            if vector.shape != (7,) or not np.isfinite(vector).all():
                raise ValueError(f"{path}包含非有限或非七维action")
            episodes.setdefault(int(episode_index), []).append((int(frame_index), vector[:6]))
    trajectories: list[np.ndarray] = []
    for episode_index in sorted(episodes):
        frames = sorted(episodes[episode_index], key=lambda item: item[0])
        indices = [item[0] for item in frames]
        if len(indices) != len(set(indices)) or indices != list(range(len(indices))):
            raise ValueError(f"episode {episode_index}的frame_index不连续或重复")
        trajectories.append(np.stack([item[1] for item in frames]))
    if not trajectories:
        raise ValueError("数据集中没有有效episode")
    return fps, trajectories, sha256_dataset(root, paths), len(paths)


def calibrate(trajectories: Sequence[np.ndarray], fps: int, quantile: float, margin: float) -> tuple[np.ndarray, np.ndarray]:
    """按episode内差分计算六关节速度和加速度限制。"""
    if not 0.0 < quantile < 1.0 or not np.isfinite(quantile):
        raise ValueError("quantile必须位于(0, 1)")
    if not margin > 0.0 or not np.isfinite(margin):
        raise ValueError("margin必须为有限正数")
    dt = 1.0 / fps
    velocities = [np.abs(np.diff(trajectory, axis=0) / dt) for trajectory in trajectories if len(trajectory) >= 2]
    accelerations = [
        np.abs((trajectory[2:] - 2.0 * trajectory[1:-1] + trajectory[:-2]) / (dt**2))
        for trajectory in trajectories
        if len(trajectory) >= 3
    ]
    if not velocities or not accelerations:
        raise ValueError("每个有效数据集至少需要一条长度不少于3帧的episode")
    velocity_limits = np.quantile(np.concatenate(velocities, axis=0), quantile, axis=0) * margin
    acceleration_limits = np.quantile(np.concatenate(accelerations, axis=0), quantile, axis=0) * margin
    if np.any(~np.isfinite(velocity_limits)) or np.any(~np.isfinite(acceleration_limits)):
        raise ValueError("标定得到非有限限制值")
    if np.any(velocity_limits <= 0.0) or np.any(acceleration_limits <= 0.0):
        raise ValueError("标定得到非正限制值；请检查专家动作是否变化")
    return velocity_limits, acceleration_limits


def main(argv: Sequence[str] | None = None) -> int:
    """执行标定并写出可被评测器锁定引用的JSON。"""
    args = build_parser().parse_args(argv)
    root = args.dataset_root.expanduser().resolve()
    fps, trajectories, dataset_sha256, parquet_count = load_episode_actions(root)
    velocity_limits, acceleration_limits = calibrate(trajectories, fps, args.quantile, args.margin)
    payload = {
        "schema_version": 1,
        "dataset_root": str(root),
        "dataset_sha256": dataset_sha256,
        "fps": fps,
        "quantile": float(args.quantile),
        "margin": float(args.margin),
        "joint_names": JOINT_NAMES,
        "episode_count": len(trajectories),
        "parquet_file_count": parquet_count,
        "velocity_limits_rad_s": velocity_limits.astype(float).tolist(),
        "acceleration_limits_rad_s2": acceleration_limits.astype(float).tolist(),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
