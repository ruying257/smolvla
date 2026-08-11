"""编排一轮真实数据训练、checkpoint 重载和 EGL rollout。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from cloud.common import PROJECT_ROOT, find_pretrained_model, resolve_path


def build_parser() -> argparse.ArgumentParser:
    """创建 smoke test 参数。"""
    parser = argparse.ArgumentParser(description="执行 SmolVLA 云端端到端 smoke test")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "smolvla-data" / "smolvla_ur10e",
        help="包含 1 至 4 条示范的数据集目录，默认使用项目内 smolvla-data/smolvla_ur10e",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "smoke",
        help="smoke 产物根目录，调用前不得存在",
    )
    parser.add_argument("--skip-bootstrap-check", action="store_true", help="跳过 GPU、模型和 EGL 环境检查")
    return parser


def run_checked(command: list[str]) -> None:
    """运行子命令并保留原始退出状态。"""
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main(argv: Sequence[str] | None = None) -> int:
    """串联 P4 的六项 smoke test 验收动作。"""
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
    checkpoint = find_pretrained_model(train_output)
    eval_output = output_root / "eval"
    run_checked(
        [
            python,
            "-m",
            "cloud.rollout",
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(eval_output),
            "--max-rollouts",
            "1",
            "--max-steps",
            "2",
        ]
    )
    required_outputs = (eval_output / "rollouts.csv", eval_output / "summary.json")
    missing = [path for path in required_outputs if not path.is_file() or path.stat().st_size == 0]
    videos = list((eval_output / "videos").glob("*.mp4"))
    if missing or not videos or any(path.stat().st_size == 0 for path in videos):
        raise RuntimeError(f"smoke 输出不完整: missing={missing}, videos={videos}")
    print(f"P4 smoke test 通过: checkpoint={checkpoint}, eval={eval_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
