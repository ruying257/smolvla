"""不创建MuJoCo环境的杯子V3双相机episode回放入口。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from collector.v3.collection_plan import MUG_DATASET_VERSION, MUG_REPO_ID
from collector.v3.dataset_io import CONTRACT_FILENAME, DATASET_FPS, configure_hf_datasets_cache


def build_parser() -> argparse.ArgumentParser:
    """创建杯子V3数据回放参数解析器。

    Returns:
        要求数据集root并支持episode索引的解析器。
    """
    parser = argparse.ArgumentParser(description="回放杯子V3双相机专家episode")
    parser.add_argument("--root", type=Path, required=True, help="V3 LeRobot数据集或单分片目录")
    parser.add_argument("--episode-index", type=int, default=0, help="需要回放的episode编号")
    return parser


def _to_rgb_uint8(value: object) -> np.ndarray:
    """把LeRobot图像转换为HWC RGB uint8。

    Args:
        value: Torch Tensor或NumPy兼容图像。

    Returns:
        形状为 ``(256,256,3)`` 的RGB uint8数组。

    Raises:
        ValueError: 转换后的图像shape不正确时抛出。
    """
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * 255.0, 0, 255)
    result = array.astype(np.uint8)
    if result.shape != (256, 256, 3):
        raise ValueError(f"V3回放图像shape错误: {result.shape}")
    return result


def _to_vector(value: object) -> np.ndarray:
    """把Tensor或数组转换为一维NumPy向量。

    Args:
        value: LeRobot样本中的state或action字段。

    Returns:
        一维浮点数组。
    """
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    return np.asarray(array).reshape(-1)


def _compose_frame(
    agent_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    task_text: str,
    scene_seed: int,
    episode_index: int,
    frame_offset: int,
    frame_total: int,
    state: np.ndarray,
    action: np.ndarray,
) -> np.ndarray:
    """合成带任务、seed、帧号、状态和动作的双相机画面。

    Args:
        agent_rgb: ``agentview``来源RGB帧。
        wrist_rgb: ``d435i_rgb``来源RGB帧。
        task_text: canonical英文训练指令。
        scene_seed: 当前杯子布局seed。
        episode_index: 数据集episode索引。
        frame_offset: episode内部零基帧索引。
        frame_total: episode总帧数。
        state: 七维绝对机器人状态。
        action: 七维绝对目标动作。

    Returns:
        可直接交给OpenCV显示的BGR画布。
    """
    combined = np.concatenate([agent_rgb, wrist_rgb], axis=1)
    canvas = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    canvas = cv2.copyMakeBorder(canvas, 112, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    lines = (
        task_text,
        f"episode={episode_index} frame={frame_offset + 1}/{frame_total} seed={scene_seed}",
        "state: " + np.array2string(state, precision=3, suppress_small=True),
        "action: " + np.array2string(action, precision=3, suppress_small=True),
    )
    for index, line in enumerate(lines):
        cv2.putText(
            canvas, line, (8, 22 + 26 * index), cv2.FONT_HERSHEY_SIMPLEX,
            0.46, (235, 235, 235), 1, cv2.LINE_AA,
        )
    cv2.putText(canvas, "Agent View", (8, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(canvas, "Wrist View", (264, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return canvas


def run_replay(root: Path, episode_index: int) -> int:
    """以20 Hz回放指定V3 episode的已记录内容。

    本函数只读取LeRobot数据和视频，不创建MuJoCo环境，也不会重新执行动作。

    Args:
        root: 最终数据集或单episode分片目录。
        episode_index: 需要回放的episode索引。

    Returns:
        用户退出或关闭窗口时返回0。

    Raises:
        FileNotFoundError: 采集契约缺失时抛出。
        ValueError: 数据身份不属于杯子V3时抛出。
        IndexError: episode索引越界时抛出。
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    resolved = root.resolve()
    contract_path = resolved / "meta" / CONTRACT_FILENAME
    if not contract_path.is_file():
        raise FileNotFoundError(f"缺少杯子V3采集契约: {contract_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("repo_id") != MUG_REPO_ID or contract.get("dataset_version") != MUG_DATASET_VERSION:
        raise ValueError("回放目录不是杯子V3数据集")
    episodes = contract.get("episodes", [])
    if not 0 <= episode_index < len(episodes):
        raise IndexError(f"episode_index越界: {episode_index}, total={len(episodes)}")
    episode_meta = episodes[episode_index]
    configure_hf_datasets_cache(resolved.parent / ".hf-lerobot-cache")
    dataset = LeRobotDataset(MUG_REPO_ID, root=resolved, video_backend="pyav")
    episode = dataset.meta.episodes[episode_index]
    start = int(episode["dataset_from_index"])
    stop = int(episode["dataset_to_index"])
    frame_indices = list(range(start, stop))
    frame_offset = 0
    paused = False
    window_name = "SmolVLA Mug V3 Replay"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        while True:
            sample = dataset[frame_indices[frame_offset]]
            canvas = _compose_frame(
                _to_rgb_uint8(sample["observation.images.agent"]),
                _to_rgb_uint8(sample["observation.images.wrist"]),
                episode_meta["task"], int(episode_meta["scene_seed"]), episode_index,
                frame_offset, len(frame_indices), _to_vector(sample["observation.state"]),
                _to_vector(sample["action"]),
            )
            cv2.imshow(window_name, canvas)
            key = cv2.waitKeyEx(0 if paused else max(1, int(1000 / DATASET_FPS)))
            if key in (27, ord("q"), ord("Q")):
                break
            if key == 32:
                paused = not paused
            elif key in (81, 2424832):
                paused = True
                frame_offset = max(0, frame_offset - 1)
            elif key in (83, 2555904):
                paused = True
                frame_offset = min(len(frame_indices) - 1, frame_offset + 1)
            elif not paused:
                frame_offset = (frame_offset + 1) % len(frame_indices)
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cv2.destroyAllWindows()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """解析参数并启动杯子V3数据回放。

    Args:
        argv: 可选命令行参数；为空时读取当前进程参数。

    Returns:
        回放流程退出码。
    """
    args = build_parser().parse_args(argv)
    return run_replay(args.root, args.episode_index)


if __name__ == "__main__":
    raise SystemExit(main())
