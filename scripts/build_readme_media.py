"""从项目数据集与评测结果生成 README 可视化素材。

该脚本只读取 LeRobot 数据集、闭环评测视频与汇总 JSON，并将专家示范、
轨迹优化对照、物理一致域随机化示范和鲁棒性统计图写入 ``assets/readme``。
生成过程不会修改原始数据或评测结果。

用法::

    python -m scripts.build_readme_media
    python -m scripts.build_readme_media --font-path C:\\Windows\\Fonts\\msyh.ttc
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import av
import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERT_DATASET = PROJECT_ROOT / "smolvla-data" / "smolvla_ur10e_mug_v1"
DEFAULT_DR_DATASET = PROJECT_ROOT / "smolvla-data" / "smolvla_ur10e_mug_dr"
DEFAULT_BASELINE_EVAL = (
    PROJECT_ROOT / "outputs" / "eval" / "mug_v1_s12000_h25_unseen_multiseed_baseline"
)
DEFAULT_CONTROLLED_EVAL = (
    PROJECT_ROOT / "outputs" / "eval" / "mug_v1_s12000_h25_unseen_multiseed_k4_limiter"
)
DEFAULT_ROBUSTNESS_EVAL = (
    PROJECT_ROOT
    / "outputs"
    / "eval"
    / "robustness"
    / "mug_color_ood_dr_s18000"
    / "mug_color_ood_dr_s18000"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "assets" / "readme"

EXPERT_EPISODES = (0, 2, 4, 6)
DR_SOURCE_EPISODES = (0, 1, 3, 4)


@dataclass(frozen=True)
class TrajectoryCase:
    """描述一组轨迹优化前后的配对 rollout。"""

    scene_seed: int
    task_id: str
    policy_seed: int

    @property
    def video_stem(self) -> str:
        """返回评测器生成的视频文件名。"""

        return (
            f"scene_{self.scene_seed}_policy_{self.policy_seed}_"
            f"{self.task_id}_canonical.mp4"
        )

    @property
    def rollout_key(self) -> str:
        """返回评测结果使用的稳定 rollout 键。"""

        return (
            f"scene={self.scene_seed}|task={self.task_id}|"
            f"prompt=canonical|policy={self.policy_seed}"
        )


TRAJECTORY_CASES = (
    TrajectoryCase(7149, "mug_on_yellow", 20261),
    TrajectoryCase(4175, "mug_on_blue", 20260),
    TrajectoryCase(1323, "mug_on_yellow", 20262),
    TrajectoryCase(8848, "mug_on_blue", 20262),
)

EXPERT_GIF = "expert_episodes.gif"
TRAJECTORY_GIF = "trajectory_optimization.gif"
DR_PAIRS_GIF = "domain_randomization_pairs.gif"
ROBUSTNESS_PNG = "dr_robustness_results.png"

DEFAULT_GIF_FPS = 8
DEFAULT_MAX_GIF_MB = 8.0
SOURCE_FPS = 20.0
TRAJECTORY_PLAYBACK_SPEED = 2.0
LABEL_BG = (12, 18, 28, 215)
WHITE = (255, 255, 255, 255)


@dataclass(frozen=True)
class EpisodeVideoRange:
    """描述 LeRobot 合并视频中的一个 episode 片段。

    Args:
        episode_index: 数据集内 episode 编号。
        start_seconds: 片段在合并视频中的起始时间。
        end_seconds: 片段在合并视频中的结束时间。
        length: episode 原始帧数。
    """

    episode_index: int
    start_seconds: float
    end_seconds: float
    length: int


@dataclass(frozen=True)
class DomainEpisode:
    """描述域随机化数据集中的一条轨迹及其来源。"""

    episode_index: int
    source_episode: int
    variant: str
    texture: str
    lighting: str
    scene_seed: int
    task_id: str
    frame_count: int
    max_state_dev: float


def build_parser() -> argparse.ArgumentParser:
    """创建 README 素材生成命令行解析器。

    Returns:
        带数据路径、字体和 GIF 参数的解析器。
    """

    parser = argparse.ArgumentParser(description="生成 SmolVLA README 可视化素材")
    parser.add_argument(
        "--expert-dataset",
        type=Path,
        default=DEFAULT_EXPERT_DATASET,
        help="原始专家 LeRobot 数据集目录",
    )
    parser.add_argument(
        "--dr-dataset",
        type=Path,
        default=DEFAULT_DR_DATASET,
        help="包含原始域与随机化域配对的 LeRobot 数据集目录",
    )
    parser.add_argument(
        "--baseline-eval",
        type=Path,
        default=DEFAULT_BASELINE_EVAL,
        help="轨迹优化基线评测目录",
    )
    parser.add_argument(
        "--controlled-eval",
        type=Path,
        default=DEFAULT_CONTROLLED_EVAL,
        help="K4 ChunkBlend 与限制器联合评测目录",
    )
    parser.add_argument(
        "--robustness-eval",
        type=Path,
        default=DEFAULT_ROBUSTNESS_EVAL,
        help="包含 summary.json 的 DR 鲁棒性评测目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="生成素材的输出目录",
    )
    parser.add_argument("--font-path", type=Path, default=None, help="可选中文字体文件")
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=DEFAULT_GIF_FPS,
        help="GIF 播放帧率，默认 8",
    )
    parser.add_argument(
        "--max-gif-mb",
        type=float,
        default=DEFAULT_MAX_GIF_MB,
        help="单个 GIF 的最大体积，默认 8 MB",
    )
    return parser


def resolve_chinese_font(explicit_path: Path | None = None) -> Path:
    """解析可用于中文标签的字体。

    Args:
        explicit_path: 用户显式指定的字体路径；指定后不再自动回退。

    Returns:
        存在的 TrueType 或 OpenType 字体路径。

    Raises:
        FileNotFoundError: 显式路径无效或系统中没有候选中文字体。
    """

    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"指定的中文字体不存在: {path}")
        return path

    windows_root = Path(os.environ.get("WINDIR", "C:/Windows"))
    candidates = (
        windows_root / "Fonts" / "msyh.ttc",
        windows_root / "Fonts" / "msyhbd.ttc",
        windows_root / "Fonts" / "simhei.ttf",
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "未找到可用中文字体；请通过 --font-path 指定微软雅黑、黑体或 Noto Sans CJK"
    )


def _load_font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    """按指定字号加载中文字体。"""

    return ImageFont.truetype(str(path), size=size)


def _ensure_file(path: Path, label: str) -> Path:
    """校验输入文件存在并返回绝对路径。"""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"缺少{label}: {resolved}")
    return resolved


def _dataset_video(dataset_root: Path, feature: str = "observation.images.agent") -> Path:
    """定位 LeRobot 数据集指定相机的唯一合并视频。"""

    candidates = sorted((dataset_root / "videos" / feature).glob("chunk-*/*.mp4"))
    if len(candidates) != 1:
        raise ValueError(
            f"{dataset_root} 的 {feature} 应包含一个合并视频，实际为 {len(candidates)}"
        )
    return candidates[0]


def load_episode_ranges(dataset_root: Path) -> dict[int, EpisodeVideoRange]:
    """读取 LeRobot episode 元数据并建立视频时间范围索引。

    Args:
        dataset_root: LeRobot 数据集根目录。

    Returns:
        episode 编号到合并视频时间范围的映射。
    """

    parquet_files = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"缺少 episode 元数据: {dataset_root / 'meta' / 'episodes'}")
    columns = [
        "episode_index",
        "length",
        "videos/observation.images.agent/from_timestamp",
        "videos/observation.images.agent/to_timestamp",
    ]
    output: dict[int, EpisodeVideoRange] = {}
    for parquet_path in parquet_files:
        table = pq.read_table(parquet_path, columns=columns)
        for row in table.to_pylist():
            episode_index = int(row["episode_index"])
            output[episode_index] = EpisodeVideoRange(
                episode_index=episode_index,
                start_seconds=float(row["videos/observation.images.agent/from_timestamp"]),
                end_seconds=float(row["videos/observation.images.agent/to_timestamp"]),
                length=int(row["length"]),
            )
    return output


def load_domain_manifest(dataset_root: Path) -> list[DomainEpisode]:
    """读取并校验域随机化 episode 清单。"""

    path = _ensure_file(dataset_root / "meta" / "domain_manifest.json", "域随机化清单")
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError(f"domain_manifest.json 缺少 episodes 列表: {path}")
    return [
        DomainEpisode(
            episode_index=int(item["episode_index"]),
            source_episode=int(item["source_episode"]),
            variant=str(item["variant"]),
            texture=str(item["texture"]),
            lighting=str(item["lighting"]),
            scene_seed=int(item["scene_seed"]),
            task_id=str(item["task_id"]),
            frame_count=int(item["frame_count"]),
            max_state_dev=float(item["max_state_dev"]),
        )
        for item in episodes
    ]


def select_domain_pairs(
    episodes: Sequence[DomainEpisode], source_episodes: Sequence[int]
) -> list[tuple[DomainEpisode, DomainEpisode]]:
    """为指定源 episode 选择原始域与随机化域配对。

    Raises:
        ValueError: 任一源 episode 不存在唯一的 original/dr0 配对。
    """

    output: list[tuple[DomainEpisode, DomainEpisode]] = []
    for source_episode in source_episodes:
        rows = [item for item in episodes if item.source_episode == source_episode]
        originals = [item for item in rows if item.variant == "original"]
        randomized = [item for item in rows if item.variant == "dr0"]
        if len(originals) != 1 or len(randomized) != 1:
            raise ValueError(
                f"source_episode={source_episode} 必须有唯一 original/dr0 配对，"
                f"实际为 original={len(originals)}、dr0={len(randomized)}"
            )
        if originals[0].frame_count != randomized[0].frame_count:
            raise ValueError(f"source_episode={source_episode} 原始与随机化帧数不一致")
        output.append((originals[0], randomized[0]))
    return output


def decode_video_segment(
    path: Path,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
) -> list[np.ndarray]:
    """按时间范围解码视频为 RGB 帧。

    Args:
        path: 视频文件路径。
        start_seconds: 包含的起始时间。
        end_seconds: 不包含的结束时间；为空时解码到文件末尾。

    Returns:
        按时间排序的 RGB uint8 图像列表。
    """

    video_path = _ensure_file(path, "视频")
    frames: list[np.ndarray] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        if start_seconds > 0:
            seek_pts = max(0, int(start_seconds / float(stream.time_base)))
            container.seek(seek_pts, stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            timestamp = float(frame.pts * frame.time_base)
            if timestamp + 1e-6 < start_seconds:
                continue
            if end_seconds is not None and timestamp >= end_seconds - 1e-6:
                break
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise ValueError(f"视频片段没有可用帧: {video_path} [{start_seconds}, {end_seconds})")
    return frames


def _resize_rgb(frame: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """将 RGB 数组缩放为指定宽高的 Pillow 图像。"""

    interpolation = cv2.INTER_AREA if frame.shape[1] > size[0] else cv2.INTER_CUBIC
    resized = cv2.resize(frame, size, interpolation=interpolation)
    return Image.fromarray(resized)


def _draw_label(
    image: Image.Image,
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    position: tuple[int, int] = (8, 6),
    background: tuple[int, int, int, int] = LABEL_BG,
    foreground: tuple[int, int, int, int] = WHITE,
) -> None:
    """在图像上绘制带半透明底色的中文标签。"""

    if image.mode != "RGBA":
        raise ValueError("_draw_label 要求 RGBA 图像")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    box = draw.textbbox(position, text, font=font)
    padding = 5
    draw.rounded_rectangle(
        (box[0] - padding, box[1] - padding, box[2] + padding, box[3] + padding),
        radius=4,
        fill=background,
    )
    draw.text(position, text, font=font, fill=foreground)
    image.alpha_composite(overlay)


def _progress_index(frame_count: int, progress: float) -> int:
    """把零到一的轨迹进度映射到合法帧索引。"""

    if frame_count <= 0:
        raise ValueError("frame_count 必须大于零")
    return min(frame_count - 1, max(0, int(round(progress * (frame_count - 1)))))


def _compose_grid(images: Sequence[Image.Image], columns: int, background=(20, 24, 32)) -> Image.Image:
    """把等尺寸图像排成规则宫格。"""

    if not images or columns <= 0:
        raise ValueError("宫格至少需要一张图且 columns 必须大于零")
    width, height = images[0].size
    if any(image.size != (width, height) for image in images):
        raise ValueError("宫格中的图像尺寸必须一致")
    rows = math.ceil(len(images) / columns)
    canvas = Image.new("RGB", (columns * width, rows * height), background)
    for index, image in enumerate(images):
        canvas.paste(image.convert("RGB"), ((index % columns) * width, (index // columns) * height))
    return canvas


def _encode_gif_once(frames: Sequence[Image.Image], path: Path, fps: int, colors: int = 96) -> None:
    """使用固定调色板参数编码一版循环 GIF。"""

    if not frames:
        raise ValueError("GIF 至少需要一帧")
    duration_ms = max(1, int(round(1000 / fps)))
    quantized = [
        frame.convert("RGB").quantize(
            colors=colors,
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.FLOYDSTEINBERG,
        )
        for frame in frames
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(
        path,
        save_all=True,
        append_images=quantized[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )


def save_gif_with_budget(
    frames: Sequence[Image.Image],
    path: Path,
    fps: int,
    max_megabytes: float,
    *,
    minimum_width: int = 420,
) -> Path:
    """编码 GIF，并通过渐进缩放把文件控制在体积上限内。

    Args:
        frames: RGB 或 RGBA 动画帧。
        path: 输出 GIF 路径。
        fps: 播放帧率。
        max_megabytes: 单文件体积上限。
        minimum_width: 自动缩放允许的最小宽度。

    Returns:
        已满足体积上限的 GIF 路径。

    Raises:
        ValueError: 参数无效或缩放至最小宽度后仍超出上限。
    """

    if fps <= 0 or max_megabytes <= 0:
        raise ValueError("fps 与 max_megabytes 必须大于零")
    working = [frame.convert("RGB") for frame in frames]
    max_bytes = int(max_megabytes * 1024 * 1024)
    while True:
        _encode_gif_once(working, path, fps)
        if path.stat().st_size <= max_bytes:
            return path
        width, height = working[0].size
        next_width = int(width * 0.88)
        if next_width < minimum_width:
            raise ValueError(
                f"GIF 在最小宽度 {minimum_width}px 时仍超过 {max_megabytes:.1f} MB: {path}"
            )
        next_height = max(1, int(height * next_width / width))
        working = [frame.resize((next_width, next_height), Image.Resampling.LANCZOS) for frame in working]


def _sample_count(sequences: Sequence[Sequence[np.ndarray]], fps: int, speed: float) -> int:
    """根据最长轨迹计算按进度同步后的 GIF 帧数。"""

    longest_seconds = max(len(sequence) / SOURCE_FPS for sequence in sequences)
    return max(2, int(math.ceil(longest_seconds / speed * fps)))


def build_expert_episode_gif(
    expert_dataset: Path,
    dr_dataset: Path,
    output_path: Path,
    font_path: Path,
    fps: int,
    max_megabytes: float,
) -> Path:
    """生成四条专家示范的 2×2 同步宫格 GIF。"""

    ranges = load_episode_ranges(expert_dataset)
    manifest = load_domain_manifest(dr_dataset)
    metadata = {
        item.source_episode: item for item in manifest if item.variant == "original"
    }
    video_path = _dataset_video(expert_dataset)
    sequences: list[list[np.ndarray]] = []
    labels: list[str] = []
    for episode_index in EXPERT_EPISODES:
        if episode_index not in ranges or episode_index not in metadata:
            raise ValueError(f"专家示范缺少 episode={episode_index} 的元数据")
        episode_range = ranges[episode_index]
        sequences.append(
            decode_video_segment(
                video_path, episode_range.start_seconds, episode_range.end_seconds
            )
        )
        item = metadata[episode_index]
        target = "蓝色区域" if item.task_id == "mug_on_blue" else "黄色区域"
        labels.append(f"Episode {episode_index} · 场景 {item.scene_seed} · 放置到{target}")

    label_font = _load_font(font_path, 17)
    output_frames: list[Image.Image] = []
    frame_count = _sample_count(sequences, fps, speed=2.0)
    for output_index in range(frame_count):
        progress = output_index / max(1, frame_count - 1)
        tiles: list[Image.Image] = []
        for sequence, label in zip(sequences, labels):
            frame = sequence[_progress_index(len(sequence), progress)]
            tile = _resize_rgb(frame, (320, 320)).convert("RGBA")
            _draw_label(tile, label, label_font)
            tiles.append(tile)
        output_frames.append(_compose_grid(tiles, columns=2))
    return save_gif_with_budget(output_frames, output_path, fps, max_megabytes)


def _domain_condition_label(item: DomainEpisode) -> str:
    """把域随机化纹理与光照标识转换为中文画面标签。"""

    texture = {
        "original": "原始纹理",
        "green_white": "绿白纹理",
        "changed": "变更纹理",
    }.get(item.texture, item.texture)
    lighting = {"default": "默认光照", "alt": "增强光照"}.get(
        item.lighting, item.lighting
    )
    return f"{texture} + {lighting}"


def build_domain_pair_gif(
    dr_dataset: Path,
    output_path: Path,
    font_path: Path,
    fps: int,
    max_megabytes: float,
) -> Path:
    """生成四组原始域与随机化域同步重放对照 GIF。"""

    ranges = load_episode_ranges(dr_dataset)
    pairs = select_domain_pairs(load_domain_manifest(dr_dataset), DR_SOURCE_EPISODES)
    video_path = _dataset_video(dr_dataset)
    decoded: list[tuple[list[np.ndarray], list[np.ndarray], DomainEpisode]] = []
    for original, randomized in pairs:
        original_range = ranges[original.episode_index]
        randomized_range = ranges[randomized.episode_index]
        original_frames = decode_video_segment(
            video_path, original_range.start_seconds, original_range.end_seconds
        )
        randomized_frames = decode_video_segment(
            video_path, randomized_range.start_seconds, randomized_range.end_seconds
        )
        if abs(len(original_frames) - len(randomized_frames)) > 1:
            raise ValueError(f"source_episode={original.source_episode} 解码后帧数不一致")
        decoded.append((original_frames, randomized_frames, randomized))

    label_font = _load_font(font_path, 15)
    small_font = _load_font(font_path, 13)
    sequences = [item for pair in decoded for item in pair[:2]]
    frame_count = _sample_count(sequences, fps, speed=2.0)
    output_frames: list[Image.Image] = []
    for output_index in range(frame_count):
        progress = output_index / max(1, frame_count - 1)
        pair_tiles: list[Image.Image] = []
        for original_frames, randomized_frames, item in decoded:
            source_index = _progress_index(min(len(original_frames), len(randomized_frames)), progress)
            original = _resize_rgb(original_frames[source_index], (200, 200)).convert("RGBA")
            randomized = _resize_rgb(randomized_frames[source_index], (200, 200)).convert("RGBA")
            _draw_label(original, "原始域", label_font)
            _draw_label(randomized, "随机化域", label_font)
            pair_image = Image.new("RGBA", (400, 238), (20, 24, 32, 255))
            pair_image.paste(original, (0, 38))
            pair_image.paste(randomized, (200, 38))
            header = (
                f"源 Episode {item.source_episode} · {_domain_condition_label(item)} · "
                f"最大状态偏差 {item.max_state_dev:.6f} rad"
            )
            _draw_label(pair_image, header, small_font, position=(7, 7))
            pair_tiles.append(pair_image)
        output_frames.append(_compose_grid(pair_tiles, columns=2))
    return save_gif_with_budget(output_frames, output_path, fps, max_megabytes)


def _load_motion_metrics(eval_root: Path) -> dict[str, dict[str, str]]:
    """读取逐 rollout 轨迹指标，并按稳定实验键建立索引。"""

    path = _ensure_file(eval_root / "motion_metrics_by_rollout.csv", "逐轨迹指标")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "rollout_key" not in rows[0]:
        raise ValueError(f"逐轨迹指标缺少 rollout_key: {path}")
    return {row["rollout_key"]: row for row in rows}


def _agent_view(frame: np.ndarray) -> np.ndarray:
    """从评测拼接视频中截取左侧第三视角画面。"""

    height, width = frame.shape[:2]
    return frame[:, :height] if width >= height * 2 else frame


def build_trajectory_comparison_gif(
    baseline_eval: Path,
    controlled_eval: Path,
    output_path: Path,
    font_path: Path,
    fps: int,
    max_megabytes: float,
) -> Path:
    """生成四组 rollout 的轨迹优化前后同步对照 GIF。"""

    baseline_metrics = _load_motion_metrics(baseline_eval)
    controlled_metrics = _load_motion_metrics(controlled_eval)
    decoded: list[dict[str, Any]] = []
    for case in TRAJECTORY_CASES:
        if case.rollout_key not in baseline_metrics or case.rollout_key not in controlled_metrics:
            raise ValueError(f"轨迹指标缺少配对 rollout: {case.rollout_key}")
        before = baseline_metrics[case.rollout_key]
        after = controlled_metrics[case.rollout_key]
        if before["success"].lower() != "true" or after["success"].lower() != "true":
            raise ValueError(f"用于 README 的轨迹对照必须两组均成功: {case.rollout_key}")
        jerk_before = float(before["p95_ee_jerk_m_s3"])
        jerk_after = float(after["p95_ee_jerk_m_s3"])
        jump_before = float(before["p95_chunk_boundary_jump_rad"])
        jump_after = float(after["p95_chunk_boundary_jump_rad"])
        decoded.append(
            {
                "case": case,
                "baseline": decode_video_segment(
                    baseline_eval / "videos" / case.video_stem
                ),
                "controlled": decode_video_segment(
                    controlled_eval / "videos" / case.video_stem
                ),
                "jerk_drop": (jerk_before - jerk_after) / jerk_before,
                "jump_drop": (jump_before - jump_after) / jump_before,
            }
        )

    total_steps = max(
        max(len(item["baseline"]), len(item["controlled"])) for item in decoded
    )
    output_count = max(
        2,
        int(
            math.ceil(
                total_steps / SOURCE_FPS / TRAJECTORY_PLAYBACK_SPEED * fps
            )
        ),
    )
    label_font = _load_font(font_path, 14)
    header_font = _load_font(font_path, 13)
    output_frames: list[Image.Image] = []
    for output_index in range(output_count):
        panels: list[Image.Image] = []
        for item in decoded:
            case = item["case"]
            baseline_frames = item["baseline"]
            controlled_frames = item["controlled"]
            pair_steps = max(len(baseline_frames), len(controlled_frames))
            source_step = min(
                pair_steps,
                int(
                    round(
                        output_index
                        * SOURCE_FPS
                        * TRAJECTORY_PLAYBACK_SPEED
                        / fps
                    )
                )
                + 1,
            )
            source_index = source_step - 1
            left_frame = baseline_frames[min(source_index, len(baseline_frames) - 1)]
            right_frame = controlled_frames[
                min(source_index, len(controlled_frames) - 1)
            ]
            left = _resize_rgb(_agent_view(left_frame), (200, 200)).convert("RGBA")
            right = _resize_rgb(_agent_view(right_frame), (200, 200)).convert("RGBA")
            _draw_label(left, "基线", label_font)
            _draw_label(right, "K4 + Limiter", label_font)

            panel = Image.new("RGBA", (400, 246), (20, 24, 32, 255))
            panel.paste(left, (0, 46))
            panel.paste(right, (200, 46))
            target = "蓝色" if case.task_id == "mug_on_blue" else "黄色"
            header = (
                f"场景 {case.scene_seed} · {target}任务 · Seed {case.policy_seed}\n"
                f"Jerk ↓{item['jerk_drop']:.1%} · 边界跳变 ↓{item['jump_drop']:.1%}"
            )
            _draw_label(
                panel,
                header,
                header_font,
                position=(6, 3),
            )
            panels.append(panel)
        output_frames.append(_compose_grid(panels, columns=2))
    return save_gif_with_budget(output_frames, output_path, fps, max_megabytes)


def _load_robustness_rows(robustness_eval: Path) -> list[dict[str, Any]]:
    """读取并校验颜色 OOD 鲁棒性汇总行。"""

    path = _ensure_file(robustness_eval / "summary.json", "鲁棒性汇总")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("condition_count", 0)) != 576:
        raise ValueError(f"鲁棒性评测条件数不是 576: {path}")
    rows = payload.get("aggregate_rows")
    if not isinstance(rows, list):
        raise ValueError(f"鲁棒性汇总缺少 aggregate_rows: {path}")
    return rows


def build_robustness_chart(
    robustness_eval: Path,
    output_path: Path,
    font_path: Path,
) -> Path:
    """绘制 DR-s18000 视觉鲁棒性成功率及场景 Bootstrap 区间。"""

    rows = _load_robustness_rows(robustness_eval)
    by_condition = {str(row["intensity"]): row for row in rows}
    groups = [
        ("原始外观", ("original@default", "original@new_light")),
        ("训练域 changed", ("changed@alt",)),
        ("未见灰色", ("holdout_gray@default", "holdout_gray@new_light")),
        ("未见紫色", ("holdout_purple@default", "holdout_purple@new_light")),
        ("未见橙色", ("holdout_orange@default", "holdout_orange@new_light")),
    ]
    required = {condition for _, conditions in groups for condition in conditions}
    missing = sorted(required - set(by_condition))
    if missing:
        raise ValueError(f"鲁棒性汇总缺少条件: {missing}")

    width, height = 1200, 720
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _load_font(font_path, 34)
    subtitle_font = _load_font(font_path, 19)
    axis_font = _load_font(font_path, 18)
    value_font = _load_font(font_path, 16)
    draw.text((60, 34), "DR-s18000 视觉鲁棒性（描述性评测）", font=title_font, fill=(20, 33, 55))
    draw.text(
        (60, 84),
        "16个未见场景 × 2任务 × 2策略种子 × 9视觉条件 = 576次闭环 rollout",
        font=subtitle_font,
        fill=(75, 85, 100),
    )

    plot_left, plot_top, plot_right, plot_bottom = 105, 150, 1145, 600
    for percent in range(0, 101, 20):
        y = plot_bottom - int((plot_bottom - plot_top) * percent / 100)
        draw.line((plot_left, y, plot_right, y), fill=(222, 227, 234), width=1)
        draw.text((48, y - 10), f"{percent}%", font=axis_font, fill=(65, 75, 90))
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=(55, 65, 80), width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=(55, 65, 80), width=2)

    lighting_colors = {
        "default": (59, 130, 246),
        "alt": (35, 166, 100),
        "new_light": (242, 142, 43),
    }
    group_width = (plot_right - plot_left) / len(groups)
    bar_width = 56
    for group_index, (group_label, conditions) in enumerate(groups):
        center = plot_left + group_width * (group_index + 0.5)
        offsets = [0.0] if len(conditions) == 1 else [-36.0, 36.0]
        for offset, condition in zip(offsets, conditions):
            row = by_condition[condition]
            lighting = condition.split("@", maxsplit=1)[1]
            rate = float(row["success_rate"])
            ci_low = float(row["ci_low"])
            ci_high = float(row["ci_high"])
            x = int(center + offset)
            y = plot_bottom - int((plot_bottom - plot_top) * rate)
            draw.rounded_rectangle(
                (x - bar_width // 2, y, x + bar_width // 2, plot_bottom),
                radius=6,
                fill=lighting_colors[lighting],
            )
            ci_y_low = plot_bottom - int((plot_bottom - plot_top) * ci_low)
            ci_y_high = plot_bottom - int((plot_bottom - plot_top) * ci_high)
            draw.line((x, ci_y_low, x, ci_y_high), fill=(30, 35, 45), width=3)
            draw.line((x - 9, ci_y_low, x + 9, ci_y_low), fill=(30, 35, 45), width=3)
            draw.line((x - 9, ci_y_high, x + 9, ci_y_high), fill=(30, 35, 45), width=3)
            value = f"{rate * 100:.1f}%"
            value_box = draw.textbbox((0, 0), value, font=value_font)
            draw.text(
                (x - (value_box[2] - value_box[0]) / 2, max(plot_top - 2, ci_y_high - 27)),
                value,
                font=value_font,
                fill=(25, 35, 50),
            )
        label_box = draw.textbbox((0, 0), group_label, font=axis_font)
        draw.text(
            (center - (label_box[2] - label_box[0]) / 2, plot_bottom + 16),
            group_label,
            font=axis_font,
            fill=(45, 55, 70),
        )

    legend = (("默认光照", "default"), ("训练光照 alt", "alt"), ("未见光照 new_light", "new_light"))
    legend_x = 260
    for label, key in legend:
        draw.rounded_rectangle((legend_x, 662, legend_x + 24, 686), radius=4, fill=lighting_colors[key])
        draw.text((legend_x + 33, 660), label, font=axis_font, fill=(45, 55, 70))
        legend_x += 245 if key != "new_light" else 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, optimize=True)
    return output_path


def build_all_media(args: argparse.Namespace) -> list[Path]:
    """按固定顺序生成 README 的全部四项素材。"""

    if args.gif_fps <= 0:
        raise ValueError("--gif-fps 必须大于零")
    if args.max_gif_mb <= 0:
        raise ValueError("--max-gif-mb 必须大于零")
    font_path = resolve_chinese_font(args.font_path)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        build_expert_episode_gif(
            args.expert_dataset,
            args.dr_dataset,
            output_dir / EXPERT_GIF,
            font_path,
            args.gif_fps,
            args.max_gif_mb,
        ),
        build_trajectory_comparison_gif(
            args.baseline_eval,
            args.controlled_eval,
            output_dir / TRAJECTORY_GIF,
            font_path,
            args.gif_fps,
            args.max_gif_mb,
        ),
        build_domain_pair_gif(
            args.dr_dataset,
            output_dir / DR_PAIRS_GIF,
            font_path,
            args.gif_fps,
            args.max_gif_mb,
        ),
        build_robustness_chart(
            args.robustness_eval,
            output_dir / ROBUSTNESS_PNG,
            font_path,
        ),
    ]
    total_megabytes = sum(path.stat().st_size for path in outputs) / (1024 * 1024)
    if total_megabytes > 25.0:
        raise ValueError(f"README 新增素材总量超过 25 MB: {total_megabytes:.2f} MB")
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    """解析参数、生成素材并打印文件体积。"""

    args = build_parser().parse_args(argv)
    outputs = build_all_media(args)
    for path in outputs:
        print(f"{path}: {path.stat().st_size / (1024 * 1024):.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
