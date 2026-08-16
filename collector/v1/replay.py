"""不创建MuJoCo环境的双相机episode视频回放入口。"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from collector.common.dataset_io import CONTRACT_FILENAME, DATASET_FPS, configure_hf_datasets_cache


def build_parser() -> argparse.ArgumentParser:
    """创建视频回放参数解析器。

    Returns:
        配置完成的命令行解析器。
    """
    parser = argparse.ArgumentParser(description="回放SmolVLA双相机专家episode")
    parser.add_argument("--root", type=Path, required=True, help="LeRobot数据集目录")
    parser.add_argument("--episode-index", type=int, default=0, help="需要回放的episode编号")
    return parser


def _to_rgb_uint8(value: object) -> np.ndarray:
    """把LeRobot返回的图像转换为HWC RGB uint8。

    Args:
        value: Torch Tensor或NumPy图像。

    Returns:
        HWC RGB uint8图像。
    """
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * 255.0, 0, 255)
    return array.astype(np.uint8)


def _to_vector(value: object) -> np.ndarray:
    """把Tensor或数组转换为一维NumPy向量。

    Args:
        value: LeRobot样本字段。

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
    """合成带任务、seed、状态和动作文本的双相机画面。

    Args:
        agent_rgb: 第三方RGB图像。
        wrist_rgb: 腕部RGB图像。
        task_text: 英文任务指令。
        scene_seed: 当前scene seed。
        episode_index: episode编号。
        frame_offset: episode内部帧编号。
        frame_total: episode总帧数。
        state: 七维当前状态。
        action: 七维目标动作。

    Returns:
        可交给OpenCV显示的BGR画面。
    """
    combined_rgb = np.concatenate([agent_rgb, wrist_rgb], axis=1)
    canvas = cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR)
    canvas = cv2.copyMakeBorder(canvas, 112, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    lines = (
        task_text,
        f"episode={episode_index} frame={frame_offset + 1}/{frame_total} seed={scene_seed}",
        "state: " + np.array2string(state, precision=3, suppress_small=True),
        "action: " + np.array2string(action, precision=3, suppress_small=True),
    )
    for line_index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (8, 22 + 26 * line_index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(canvas, "Agent View", (8, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(canvas, "Wrist View", (264, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return canvas


def run_replay(root: Path, episode_index: int) -> int:
    """加载指定episode并以20 Hz显示两路已记录视频。

    Args:
        root: LeRobot数据集根目录。
        episode_index: 需要回放的episode编号。

    Returns:
        正常关闭窗口时返回0。
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    contract_path = root.resolve() / "meta" / CONTRACT_FILENAME
    if not contract_path.is_file():
        raise FileNotFoundError(f"缺少采集契约: {contract_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    episodes = contract.get("episodes", [])
    if not 0 <= episode_index < len(episodes):
        raise IndexError(f"episode_index超出范围: {episode_index}, total={len(episodes)}")
    episode_meta = episodes[episode_index]
    repo_id = str(contract.get("repo_id", ""))
    if not repo_id:
        raise ValueError("采集契约缺少repo_id")
    configure_hf_datasets_cache(root.resolve().parent / ".hf-lerobot-cache")
    dataset = LeRobotDataset(repo_id, root=root.resolve(), video_backend="pyav")
    episode = dataset.meta.episodes[episode_index]
    start = int(episode["dataset_from_index"])
    stop = int(episode["dataset_to_index"])
    frame_indices = list(range(start, stop))
    frame_offset = 0
    paused = False
    cv2.namedWindow("SmolVLA Episode Replay", cv2.WINDOW_NORMAL)
    try:
        while True:
            sample = dataset[frame_indices[frame_offset]]
            canvas = _compose_frame(
                _to_rgb_uint8(sample["observation.images.agent"]),
                _to_rgb_uint8(sample["observation.images.wrist"]),
                episode_meta["task"],
                int(episode_meta["scene_seed"]),
                episode_index,
                frame_offset,
                len(frame_indices),
                _to_vector(sample["observation.state"]),
                _to_vector(sample["action"]),
            )
            cv2.imshow("SmolVLA Episode Replay", canvas)
            wait_ms = 0 if paused else max(1, int(1000 / DATASET_FPS))
            key = cv2.waitKeyEx(wait_ms)
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
                frame_offset += 1
                if frame_offset >= len(frame_indices):
                    frame_offset = 0
            if cv2.getWindowProperty("SmolVLA Episode Replay", cv2.WND_PROP_VISIBLE) < 1:
                break
            if not paused:
                time.sleep(0.0)
    finally:
        cv2.destroyAllWindows()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """解析参数并启动episode回放。

    Args:
        argv: 可选命令行参数；为空时读取进程参数。

    Returns:
        回放入口退出码。
    """
    args = build_parser().parse_args(argv)
    return run_replay(args.root, args.episode_index)


if __name__ == "__main__":
    raise SystemExit(main())
