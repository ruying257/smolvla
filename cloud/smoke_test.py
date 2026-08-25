"""编排一轮真实数据云端训练并检查 checkpoint。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from cloud.common import PROJECT_ROOT, resolve_path


def build_parser() -> argparse.ArgumentParser:
    """创建 smoke test 参数。"""
    parser = argparse.ArgumentParser(description="执行 SmolVLA 云端训练 smoke test")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "smolvla-data" / "smolvla_ur10e_mug_v1",
        help="包含 1 至 4 条示范的数据集目录，默认使用项目内 smolvla-data/smolvla_ur10e_mug_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "smoke",
        help="smoke 产物根目录，调用前不得存在",
    )
    parser.add_argument("--skip-bootstrap-check", action="store_true", help="跳过云端 GPU、模型和 EGL 环境检查")
    return parser


def run_checked(command: list[str]) -> None:
    """运行子命令并保留原始退出状态。"""
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main(argv: Sequence[str] | None = None) -> int:
    """串联环境检查、单步训练和 checkpoint 验收。"""
    args = build_parser().parse_args(argv)
    dataset_root = resolve_path(args.dataset_root)
    output_root = resolve_path(args.output_dir)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"数据集目录不存在: {dataset_root}")
    if output_root.exists():
        raise FileExistsError(f"smoke 输出目录已存在，避免混用旧产物: {output_root}")

    python = sys.executable
    if not args.skip_bootstrap_check:
        run_checked([python, "-m", "cloud.bootstrap_check"])

    train_output = output_root / "train"
    run_checked(
        [
            python,
            "-m",
            "cloud.train",
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(train_output),
            "--job-name",
            "smolvla_ur10e_smoke",
            "--smoke",
        ]
    )
    required = ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json")
    numeric_checkpoints = sorted(
        (path for path in (train_output / "checkpoints").glob("*") if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
        reverse=True,
    )
    candidates = (
        train_output / "pretrained_model",
        train_output / "checkpoints" / "last" / "pretrained_model",
        *(checkpoint / "pretrained_model" for checkpoint in numeric_checkpoints),
    )
    checkpoint = next(
        (candidate for candidate in candidates if all((candidate / filename).is_file() for filename in required)),
        None,
    )
    if checkpoint is None:
        raise RuntimeError(f"单步训练未生成完整 checkpoint: {train_output}")
    print(f"P4 云端训练 smoke test 通过: checkpoint={checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
