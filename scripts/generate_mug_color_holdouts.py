"""生成仅用于评测的 Mug 未见纯色纹理及可复现清单。

本工具以原始红色 Mug 纹理为模板，只替换高饱和红色杯身像素。白色 UV
区域、黑色标识、透明度和图像尺寸保持不变，避免把几何或图案变化混入
颜色泛化实验。

典型用法：

    python -m scripts.generate_mug_color_holdouts
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VISUAL_DIR = PROJECT_ROOT / "assets" / "mujoco" / "mug_5" / "visual"
SOURCE_FILENAME = "image0.png"
MANIFEST_FILENAME = "holdout_color_manifest.json"
HOLDOUT_COLORS = {
    "holdout_gray": (0x73, 0x73, 0x73),
    "holdout_purple": (0x7B, 0x2C, 0xBF),
    "holdout_orange": (0xE6, 0x7E, 0x22),
}


def sha256_file(path: Path) -> str:
    """计算文件 SHA-256。

    Args:
        path: 待读取文件。

    Returns:
        小写十六进制 SHA-256。
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_body_mask(rgb: np.ndarray) -> np.ndarray:
    """定位原始纹理中的高饱和红色杯身像素。

    Args:
        rgb: ``(H, W, 3)`` uint8 原始纹理。

    Returns:
        ``(H, W)`` 布尔掩码。

    Raises:
        ValueError: 输入格式错误或未找到足够的杯身像素时抛出。
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"原始纹理必须是(H,W,3) uint8，实际={rgb.shape}/{rgb.dtype}")
    red = rgb[..., 0].astype(np.int16)
    green = rgb[..., 1].astype(np.int16)
    blue = rgb[..., 2].astype(np.int16)
    mask = (red >= 128) & (red - green >= 64) & (red - blue >= 64)
    if int(np.count_nonzero(mask)) < rgb.shape[0] * rgb.shape[1] // 4:
        raise ValueError("原始纹理中的红色杯身掩码过小，拒绝生成")
    return mask


def recolor_texture(source: np.ndarray, target_rgb: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """只替换杯身颜色并返回结果与变更掩码。

    Args:
        source: ``(H, W, 3|4)`` uint8 原始纹理。
        target_rgb: 目标纯色 RGB。

    Returns:
        ``(重着色纹理, 二维变更掩码)``。
    """
    if source.ndim != 3 or source.shape[2] not in (3, 4) or source.dtype != np.uint8:
        raise ValueError(f"纹理必须是(H,W,3|4) uint8，实际={source.shape}/{source.dtype}")
    output = source.copy()
    mask = build_body_mask(source[..., :3])
    output[mask, :3] = np.asarray(target_rgb, dtype=np.uint8)
    return output, mask


def generate_holdout_textures(
    visual_dir: Path = DEFAULT_VISUAL_DIR,
    manifest_path: Path | None = None,
) -> dict[str, object]:
    """生成全部 holdout 纹理并写入确定性 manifest。

    Args:
        visual_dir: 原始纹理与输出纹理所在目录。
        manifest_path: manifest 输出路径；省略时写入 ``visual_dir``。

    Returns:
        可直接序列化的 manifest 字典。
    """
    visual_dir = visual_dir.resolve()
    source_path = visual_dir / SOURCE_FILENAME
    if not source_path.is_file():
        raise FileNotFoundError(f"原始 Mug 纹理不存在: {source_path}")
    manifest_path = (manifest_path or visual_dir / MANIFEST_FILENAME).resolve()

    with Image.open(source_path) as image:
        mode = image.mode
        if mode not in {"RGB", "RGBA"}:
            image = image.convert("RGBA" if "A" in mode else "RGB")
            mode = image.mode
        source = np.asarray(image, dtype=np.uint8).copy()

    outputs: dict[str, dict[str, object]] = {}
    reference_mask: np.ndarray | None = None
    for variant, target_rgb in HOLDOUT_COLORS.items():
        recolored, mask = recolor_texture(source, target_rgb)
        if reference_mask is None:
            reference_mask = mask
        elif not np.array_equal(mask, reference_mask):
            raise RuntimeError("不同目标颜色生成了不一致的杯身掩码")
        output_path = visual_dir / f"image0_{variant}.png"
        Image.fromarray(recolored, mode=mode).save(output_path, format="PNG", compress_level=9)
        outputs[variant] = {
            "filename": output_path.name,
            "rgb": list(target_rgb),
            "hex": "#" + "".join(f"{value:02X}" for value in target_rgb),
            "sha256": sha256_file(output_path),
        }

    assert reference_mask is not None
    manifest: dict[str, object] = {
        "schema_version": 1,
        "purpose": "evaluation_only_unseen_mug_colors",
        "source": {
            "filename": source_path.name,
            "sha256": sha256_file(source_path),
            "mode": mode,
            "size": [int(source.shape[1]), int(source.shape[0])],
        },
        "mask": {
            "rule": "r>=128 and r-g>=64 and r-b>=64",
            "changed_pixels": int(np.count_nonzero(reference_mask)),
        },
        "outputs": outputs,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="生成 Mug 未见纯色评测纹理")
    parser.add_argument("--visual-dir", type=Path, default=DEFAULT_VISUAL_DIR)
    parser.add_argument("--manifest", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """生成纹理并打印 manifest。"""
    args = build_parser().parse_args(argv)
    manifest = generate_holdout_textures(args.visual_dir, args.manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
