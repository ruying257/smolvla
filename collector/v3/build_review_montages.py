"""为每个杯子scene生成蓝黄任务并排人工复核视频。"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable, Sequence
from pathlib import Path

import av
import cv2
import numpy as np

from collector.v3.collection_plan import (
    REVIEW_FILENAME,
    load_config,
    load_progress,
    plan_for_mode,
    validate_completed_shards,
)


TASK_LAYOUT = ("mug_on_blue", "mug_on_yellow")
VALID_REVIEW_VALUES = {
    "pending",
    "pass",
    "redo_mug_on_blue",
    "redo_mug_on_yellow",
}


def build_parser() -> argparse.ArgumentParser:
    """创建V3蒙太奇命令行解析器。

    Returns:
        支持pilot和2至4倍速参数的解析器。
    """
    parser = argparse.ArgumentParser(description="生成杯子V3蓝黄并排复核蒙太奇")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true", help="只生成两个pilot scene")
    parser.add_argument("--speed", type=float, default=None, help="回放倍速，范围2至4")
    return parser


def _decode_sampled_video(path: Path, speed: float) -> list[tuple[int, np.ndarray]]:
    """解码倍速复核需要的RGB帧。

    Args:
        path: 单路相机MP4路径。
        speed: 2至4倍的时间采样倍率。

    Returns:
        ``(原始帧索引, RGB图像)``列表。

    Raises:
        ValueError: 视频无法解码或没有帧时抛出。
    """
    selected: list[tuple[int, np.ndarray]] = []
    output_index = 0
    try:
        with av.open(str(path)) as container:
            for source_index, frame in enumerate(container.decode(video=0)):
                if source_index < int(output_index * speed):
                    continue
                selected.append((source_index, frame.to_ndarray(format="rgb24")))
                output_index += 1
    except Exception as exc:
        raise ValueError(f"复核视频无法解码: {path}: {exc}") from exc
    if not selected:
        raise ValueError(f"复核视频为空: {path}")
    return selected


def _single_video(shard: Path, feature: str) -> Path:
    """定位单分片指定feature的唯一MP4。

    Args:
        shard: 单episode分片根目录。
        feature: ``observation.images.agent``或``observation.images.wrist``。

    Returns:
        唯一MP4路径。

    Raises:
        ValueError: 视频缺失或存在多个文件时抛出。
    """
    paths = list((shard / "videos" / feature).glob("chunk-*/*.mp4"))
    if len(paths) != 1:
        raise ValueError(f"V3相机视频数量必须为1: {shard}, {feature}, actual={len(paths)}")
    return paths[0]


def _task_column(
    agent_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    scene_seed: int,
    task_id: str,
    prompt: str,
    frame_index: int,
    frame_total: int,
) -> np.ndarray:
    """生成一个任务的agent在上、wrist在下的256×512列。

    Args:
        agent_rgb: 第三方相机RGB帧。
        wrist_rgb: 腕部相机RGB帧。
        scene_seed: 当前共享scene seed。
        task_id: 蓝区或黄区任务标识。
        prompt: canonical英文指令。
        frame_index: 原episode帧索引。
        frame_total: 原episode总帧数。

    Returns:
        带任务、seed、prompt和帧号标注的BGR任务列。
    """
    agent = cv2.cvtColor(agent_rgb, cv2.COLOR_RGB2BGR)
    wrist = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR)
    column = np.concatenate([agent, wrist], axis=0)
    cv2.rectangle(column, (0, 0), (255, 59), (12, 12, 12), -1)
    lines = (
        f"scene={scene_seed} {task_id}",
        f"frame={frame_index + 1}/{frame_total}",
        prompt,
    )
    for index, line in enumerate(lines):
        cv2.putText(
            column, line, (4, 16 + index * 18), cv2.FONT_HERSHEY_SIMPLEX,
            0.34 if index < 2 else 0.28, (255, 255, 255) if index < 2 else (0, 230, 255),
            1, cv2.LINE_AA,
        )
    cv2.putText(column, "AGENT", (4, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(column, "WRIST", (4, 278), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return column


def _write_h264(path: Path, frames: Iterable[np.ndarray], fps: int) -> None:
    """把512×512 BGR复核帧编码为H.264 MP4。

    Args:
        path: 输出MP4路径。
        frames: 惰性产生BGR帧的可迭代对象。
        fps: 输出视频帧率。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width = 512
        stream.height = 512
        stream.pix_fmt = "yuv420p"
        for image in frames:
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _load_review_status(path: Path) -> dict[int, str]:
    """读取并校验已有逐scene人工复核状态。

    Args:
        path: ``review_status.csv``路径。

    Returns:
        scene seed到合法状态的映射；文件不存在时返回空映射。

    Raises:
        ValueError: 状态不属于pending、pass或两个redo值时抛出。
    """
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        result = {int(row["scene_seed"]): row["status"] for row in csv.DictReader(handle)}
    invalid = {seed: status for seed, status in result.items() if status not in VALID_REVIEW_VALUES}
    if invalid:
        raise ValueError(f"V3人工复核状态非法: {invalid}")
    return result


def _write_review_status(path: Path, seeds: Sequence[int], existing: dict[int, str]) -> None:
    """保留已有合法结果并补齐全部20个scene复核行。

    Args:
        path: 目标CSV路径。
        seeds: 锁定的20个scene seed。
        existing: 已有复核状态。
    """
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scene_seed", "status"])
        writer.writeheader()
        for seed in seeds:
            writer.writerow({"scene_seed": seed, "status": existing.get(seed, "pending")})


def build_montages(config_path: Path, pilot: bool, speed: float | None) -> list[Path]:
    """生成选定scene的蓝黄任务同步对照视频。

    Args:
        config_path: 锁定V3配置路径。
        pilot: 是否只处理两个pilot scene。
        speed: 可选2至4倍速；为空时使用配置默认值。

    Returns:
        按配置scene顺序生成的MP4路径列表。

    Raises:
        ValueError: 倍速越界或视频结构错误时抛出。
        RuntimeError: 任一scene的蓝黄任务尚未成对完成时抛出。
    """
    config = load_config(config_path)
    playback_speed = config.montage_speed if speed is None else speed
    if not 2.0 <= playback_speed <= 4.0:
        raise ValueError("--speed必须位于2至4之间")
    progress = load_progress(config)
    validate_completed_shards(config, progress)
    plan = plan_for_mode(config, pilot=pilot)
    by_pair = {(item.scene_seed, item.task_id): item for item in plan}
    records = progress["completed"]
    output_dir = config.work_root / "review_montages"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    scene_seeds = list(dict.fromkeys(item.scene_seed for item in plan))
    for seed in scene_seeds:
        missing = [
            task_id for task_id in TASK_LAYOUT
            if by_pair[(seed, task_id)].queue_key not in records
        ]
        if missing:
            raise RuntimeError(f"scene={seed}尚未完成蓝黄配对: {missing}")
        streams: dict[str, tuple[list[tuple[int, np.ndarray]], list[tuple[int, np.ndarray]], Any]] = {}
        for task_id in TASK_LAYOUT:
            item = by_pair[(seed, task_id)]
            shard = config.shard_root / records[item.queue_key]["shard_name"]
            streams[task_id] = (
                _decode_sampled_video(_single_video(shard, "observation.images.agent"), playback_speed),
                _decode_sampled_video(_single_video(shard, "observation.images.wrist"), playback_speed),
                item,
            )
        output_total = max(len(value[0]) for value in streams.values())

        def montage_frames() -> Iterable[np.ndarray]:
            """惰性合成当前scene的蓝黄并排帧。

            Yields:
                左蓝右黄、每列上agent下wrist的512×512 BGR帧。
            """
            for output_index in range(output_total):
                columns = []
                for task_id in TASK_LAYOUT:
                    agent, wrist, item = streams[task_id]
                    agent_index = min(output_index, len(agent) - 1)
                    wrist_index = min(output_index, len(wrist) - 1)
                    source_index, agent_image = agent[agent_index]
                    _, wrist_image = wrist[wrist_index]
                    columns.append(_task_column(
                        agent_image, wrist_image, seed, task_id, item.prompt,
                        source_index, int(records[item.queue_key]["frame_count"]),
                    ))
                yield np.concatenate(columns, axis=1)

        output = output_dir / f"scene_{seed}.mp4"
        _write_h264(output, montage_frames(), config.fps)
        outputs.append(output)
    review_path = config.work_root / REVIEW_FILENAME
    _write_review_status(review_path, config.scene_seeds, _load_review_status(review_path))
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    """解析参数、生成复核视频并提示用户填写状态表。

    Args:
        argv: 可选命令行参数；为空时读取当前进程参数。

    Returns:
        全部目标蒙太奇生成成功时返回0。
    """
    args = build_parser().parse_args(argv)
    outputs = build_montages(args.config, args.pilot, args.speed)
    print(f"已生成{len(outputs)}个杯子V3复核蒙太奇；人工检查后填写review_status.csv。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
