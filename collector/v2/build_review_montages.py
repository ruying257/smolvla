"""为每个scene生成四任务2×2同步人工复核视频。"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable, Sequence
from pathlib import Path

import av
import cv2
import numpy as np

from collector.v2.collection_plan import (
    REVIEW_FILENAME,
    load_config,
    load_progress,
    plan_for_mode,
    validate_completed_shards,
)


LAYOUT = (
    ("red_on_blue", "green_on_blue"),
    ("red_on_yellow", "green_on_yellow"),
)
VALID_REVIEW_VALUES = {
    "pending",
    "pass",
    "redo_red_on_blue",
    "redo_green_on_blue",
    "redo_red_on_yellow",
    "redo_green_on_yellow",
}


def build_parser() -> argparse.ArgumentParser:
    """创建复核蒙太奇命令行解析器。"""
    parser = argparse.ArgumentParser(description="生成Grounding v2四任务复核蒙太奇")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true", help="只生成前两个scene")
    parser.add_argument("--speed", type=float, default=None, help="回放倍速，范围2至4")
    return parser


def _decode_sampled_video(
    path: Path,
    speed: float,
    resize: tuple[int, int] | None = None,
) -> list[tuple[int, np.ndarray]]:
    """解码倍速回放实际需要的RGB帧，控制蒙太奇内存占用。"""
    selected: list[tuple[int, np.ndarray]] = []
    output_index = 0
    with av.open(str(path)) as container:
        for source_index, frame in enumerate(container.decode(video=0)):
            if source_index < int(output_index * speed):
                continue
            image = frame.to_ndarray(format="rgb24")
            if resize is not None:
                image = cv2.resize(image, resize)
            selected.append((source_index, image))
            output_index += 1
    return selected


def _single_video(shard: Path, camera: str) -> Path:
    """返回指定相机唯一视频，缺失或重复时失败。"""
    paths = list((shard / "videos" / camera).glob("chunk-*/*.mp4"))
    if len(paths) != 1:
        raise ValueError(f"相机视频数量必须为1: shard={shard}, camera={camera}, actual={len(paths)}")
    return paths[0]


def _task_cell(
    agent_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    scene_seed: int,
    task_id: str,
    prompt: str,
    frame_index: int,
    frame_total: int,
) -> np.ndarray:
    """生成带腕部画中画和文本标注的256像素任务单元。"""
    cell = cv2.cvtColor(agent_rgb, cv2.COLOR_RGB2BGR)
    wrist = cv2.resize(cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR), (82, 82))
    cell[-86:-4, -86:-4] = wrist
    cv2.rectangle(cell, (0, 0), (255, 49), (12, 12, 12), -1)
    cv2.putText(
        cell, f"scene={scene_seed} {task_id}", (4, 15), cv2.FONT_HERSHEY_SIMPLEX,
        0.36, (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        cell, f"frame={frame_index + 1}/{frame_total}", (4, 29),
        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (220, 220, 220), 1, cv2.LINE_AA,
    )
    cv2.putText(
        cell, prompt, (4, 44), cv2.FONT_HERSHEY_SIMPLEX,
        0.27, (0, 230, 255), 1, cv2.LINE_AA,
    )
    return cell


def _write_h264(path: Path, frames: Iterable[np.ndarray], fps: int) -> None:
    """使用PyAV把BGR蒙太奇帧编码为H.264视频。"""
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
    """读取已有人工复核结果，保留合法状态。"""
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        result = {int(row["scene_seed"]): row["status"] for row in rows}
    invalid = {seed: status for seed, status in result.items() if status not in VALID_REVIEW_VALUES}
    if invalid:
        raise ValueError(f"人工复核状态非法: {invalid}")
    return result


def _write_review_status(path: Path, seeds: list[int], existing: dict[int, str]) -> None:
    """写出用户需要逐scene填写的复核状态表。"""
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scene_seed", "status"])
        writer.writeheader()
        for seed in seeds:
            writer.writerow({"scene_seed": seed, "status": existing.get(seed, "pending")})


def build_montages(config_path: Path, pilot: bool, speed: float | None) -> list[Path]:
    """生成选定scene的全部2×2复核视频并初始化复核表。"""
    config = load_config(config_path)
    playback_speed = config.montage_speed if speed is None else speed
    if not 2.0 <= playback_speed <= 4.0:
        raise ValueError("--speed必须位于2至4之间")
    progress = load_progress(config)
    validate_completed_shards(config, progress)
    plan = plan_for_mode(config, pilot=pilot)
    by_scene_task = {(item.scene_seed, item.task_id): item for item in plan}
    records = progress["completed"]
    output_dir = config.work_root / "review_montages"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    scene_seeds = list(dict.fromkeys(item.scene_seed for item in plan))
    for scene_seed in scene_seeds:
        missing = [
            task for row in LAYOUT for task in row
            if by_scene_task[(scene_seed, task)].queue_key not in records
        ]
        if missing:
            raise RuntimeError(f"scene={scene_seed}尚未完成四任务: {missing}")
        streams = {}
        for row in LAYOUT:
            for task_id in row:
                item = by_scene_task[(scene_seed, task_id)]
                shard = config.shard_root / records[item.queue_key]["shard_name"]
                streams[task_id] = (
                    _decode_sampled_video(
                        _single_video(shard, "observation.images.agent"), playback_speed,
                    ),
                    _decode_sampled_video(
                        _single_video(shard, "observation.images.wrist"), playback_speed, (82, 82),
                    ),
                    item,
                )
        output_total = max(len(value[0]) for value in streams.values())

        def montage_frames() -> Iterable[np.ndarray]:
            """逐帧合成当前scene，避免同时缓存全部512像素输出帧。"""
            for output_index in range(output_total):
                rows = []
                for layout_row in LAYOUT:
                    cells = []
                    for task_id in layout_row:
                        agent, wrist, item = streams[task_id]
                        agent_index = min(output_index, len(agent) - 1)
                        wrist_index = min(output_index, len(wrist) - 1)
                        source_index, agent_image = agent[agent_index]
                        _, wrist_image = wrist[wrist_index]
                        cells.append(
                            _task_cell(
                                agent_image, wrist_image, scene_seed, task_id,
                                item.prompt, source_index, int(records[item.queue_key]["frame_count"]),
                            )
                        )
                    rows.append(np.concatenate(cells, axis=1))
                yield np.concatenate(rows, axis=0)

        output = output_dir / f"scene_{scene_seed}.mp4"
        _write_h264(output, montage_frames(), config.fps)
        outputs.append(output)
    review_path = config.work_root / REVIEW_FILENAME
    existing = _load_review_status(review_path)
    all_seeds = list(config.scene_seeds)
    _write_review_status(review_path, all_seeds, existing)
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    """解析参数并生成复核蒙太奇。"""
    args = build_parser().parse_args(argv)
    outputs = build_montages(args.config, args.pilot, args.speed)
    print(f"已生成{len(outputs)}个复核蒙太奇；请填写review_status.csv后再运行验收。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
