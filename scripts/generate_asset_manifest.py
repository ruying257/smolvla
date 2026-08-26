"""生成 ACT 场景迁移资源的 SHA-256 清单。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    """分块计算文件 SHA-256。

    Args:
        path: 需要计算摘要的文件。

    Returns:
        小写十六进制 SHA-256 字符串。
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_manifest(asset_root: Path, source_root: Path, output_path: Path) -> None:
    """生成迁移资源与 ACT 源资源的逐文件摘要。

    Args:
        asset_root: SmolVLA 中的 ``assets/mujoco`` 目录。
        source_root: ACT 项目的 ``mode`` 目录。
        output_path: JSON 清单输出路径。
    """
    records = []
    for target_path in sorted(path for path in asset_root.rglob("*") if path.is_file()):
        relative_name = target_path.relative_to(asset_root).as_posix()
        # 两个主场景都以ACT的demo_scene.xml为结构来源；杯子兼容XML则以
        # model_new.xml为来源。其余资源保持相同的相对路径映射。
        source_overrides = {
            "scene.xml": "demo_scene.xml",
            "mug_scene.xml": "demo_scene.xml",
            "mug_5/model_smolvla.xml": "mug_5/model_new.xml",
            "mug_5/visual/image0_green_white.png": "mug_5/visual/image0.png",
            "mug_5/visual/image0_holdout_gray.png": "mug_5/visual/image0.png",
            "mug_5/visual/image0_holdout_purple.png": "mug_5/visual/image0.png",
            "mug_5/visual/image0_holdout_orange.png": "mug_5/visual/image0.png",
        }
        source_relative_name = source_overrides.get(relative_name, relative_name)
        source_path = source_root / source_relative_name
        target_hash = _sha256(target_path)
        source_hash = _sha256(source_path) if source_path.is_file() else None
        records.append(
            {
                "path": relative_name,
                "source_path": source_relative_name if source_path.is_file() else None,
                "size": target_path.stat().st_size,
                "sha256": target_hash,
                "source_sha256": source_hash,
                "status": "copied" if source_hash == target_hash else "modified",
            }
        )

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "mujoco-act-robotics/mode",
        "file_count": len(records),
        "total_bytes": sum(record["size"] for record in records),
        "files": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    """创建资源清单脚本的参数解析器。

    Returns:
        已配置目标、源和输出路径的解析器。
    """
    parser = argparse.ArgumentParser(description="生成 ACT MuJoCo 资源迁移清单")
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """解析参数并生成资源清单。

    Args:
        argv: 可选命令行参数；为空时读取当前进程参数。

    Returns:
        成功时返回0。
    """
    args = build_parser().parse_args(argv)
    generate_manifest(args.asset_root.resolve(), args.source_root.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
