"""检查云端 GPU、依赖、公开模型下载和 MuJoCo EGL 渲染。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Sequence

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")


def build_parser() -> argparse.ArgumentParser:
    """创建云端环境检查参数。"""
    parser = argparse.ArgumentParser(description="检查 SmolVLA 云端运行环境")
    parser.add_argument("--model-id", default="lerobot/smolvla_base", help="公开 SmolVLA 模型 ID")
    parser.add_argument("--skip-model-download", action="store_true", help="跳过模型快照下载")
    parser.add_argument("--minimum-vram-gb", type=float, default=14.0, help="最低可接受 GPU 显存")
    return parser


def check_environment(model_id: str, skip_model_download: bool, minimum_vram_gb: float) -> dict[str, object]:
    """执行会在正式训练前暴露环境问题的检查。

    Args:
        model_id: Hugging Face 公开模型 ID。
        skip_model_download: 是否跳过模型快照下载。
        minimum_vram_gb: 最低显存门槛，单位 GiB。

    Returns:
        可序列化的环境报告。

    Raises:
        RuntimeError: Python、GPU、依赖、模型或 EGL 检查失败。
    """
    if sys.version_info[:2] not in {(3, 10), (3, 11)}:
        raise RuntimeError(f"要求 Python 3.10 或 3.11，实际为 {platform.python_version()}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("找不到 ffmpeg；请使用 bootstrap_cloud.sh --install-system-packages")

    import mujoco
    import torch
    from huggingface_hub import snapshot_download

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch 无法使用 CUDA，请检查 NVIDIA 驱动和 cu126 wheel")
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    vram_gb = properties.total_memory / (1024**3)
    if vram_gb < minimum_vram_gb:
        raise RuntimeError(f"GPU 显存不足: {vram_gb:.1f} GiB < {minimum_vram_gb:.1f} GiB")

    model_path = ""
    if not skip_model_download:
        model_path = snapshot_download(
            repo_id=model_id,
            allow_patterns=[
                "config.json",
                "model.safetensors",
                "policy_preprocessor.json",
                "policy_postprocessor.json",
                "policy_*_processor.safetensors",
            ],
        )

    from sim.environment import CleanTabletopEnv

    with CleanTabletopEnv() as env:
        env.reset(0)
        images = env.capture_training_images()
        if set(images) != {"agent", "wrist"}:
            raise RuntimeError(f"EGL 相机输出不完整: {set(images)}")
        if any(image.shape != (256, 256, 3) for image in images.values()):
            raise RuntimeError("EGL 相机输出 shape 错误")

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": True,
        "gpu_name": properties.name,
        "gpu_vram_gb": round(vram_gb, 2),
        "mujoco": mujoco.__version__,
        "lerobot": importlib.metadata.version("lerobot"),
        "mujoco_gl": os.environ["MUJOCO_GL"],
        "headless_images": {key: list(value.shape) for key, value in images.items()},
        "model_id": model_id,
        "model_cache": model_path,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """运行检查并在成功时输出 JSON 报告。"""
    args = build_parser().parse_args(argv)
    report = check_environment(args.model_id, args.skip_model_download, args.minimum_vram_gb)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
