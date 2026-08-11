"""根据项目 YAML 配置调用 LeRobot SmolVLA 训练入口。"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Sequence

from cloud.common import PROJECT_ROOT, format_command, load_yaml_config, resolve_path, run_logged


def build_parser() -> argparse.ArgumentParser:
    """创建训练命令行解析器。"""
    parser = argparse.ArgumentParser(description="启动 SmolVLA 云端训练")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "smolvla-data" / "smolvla_ur10e",
        help="完整 LeRobot 数据集目录，默认使用项目内 smolvla-data/smolvla_ur10e",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "cloud_train.yaml",
        help="训练 YAML 配置",
    )
    parser.add_argument("--output-dir", type=Path, help="覆盖配置中的训练输出目录")
    parser.add_argument("--job-name", help="覆盖配置中的任务名称")
    parser.add_argument("--smoke", action="store_true", help="强制执行 1 step、batch=1 的 smoke 训练")
    parser.add_argument("--dry-run", action="store_true", help="只打印最终 lerobot-train 命令")
    return parser


def build_train_command(
    config: dict[str, Any],
    dataset_root: Path,
    output_dir: Path,
    job_name: str,
    smoke: bool = False,
    executable: str = "lerobot-train",
) -> list[str]:
    """构造不经过 shell 插值的 LeRobot 训练参数列表。

    Args:
        config: ``cloud_train.yaml`` 配置。
        dataset_root: 完整 LeRobot 数据集目录。
        output_dir: 本次训练输出目录，调用前不得存在。
        job_name: LeRobot 任务名称。
        smoke: 是否覆盖为单步 smoke 训练。
        executable: LeRobot CLI 可执行文件名。

    Returns:
        可传给 ``subprocess`` 的参数列表。
    """
    train = _mapping(config, "train")
    policy = _mapping(config, "policy")
    dataset = _mapping(config, "dataset")
    steps = 1 if smoke else int(train.get("steps", 20_000))
    batch_size = 1 if smoke else int(train.get("batch_size", 1))
    save_freq = 1 if smoke else int(train.get("save_freq", steps))
    return [
        executable,
        f"--policy.path={policy.get('model_id', 'lerobot/smolvla_base')}",
        "--policy.input_features=null",
        "--policy.output_features=null",
        f"--policy.empty_cameras={int(policy.get('empty_cameras', 1))}",
        f"--policy.device={policy.get('device', 'cuda')}",
        f"--policy.use_amp={_bool_text(policy.get('use_amp', True))}",
        "--policy.push_to_hub=false",
        f"--dataset.repo_id={dataset.get('repo_id', 'smolvla_ur10e')}",
        f"--dataset.root={dataset_root}",
        f"--dataset.video_backend={dataset.get('video_backend', 'pyav')}",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        f"--num_workers={int(train.get('num_workers', 2))}",
        f"--seed={int(train.get('seed', 1000))}",
        f"--log_freq={1 if smoke else int(train.get('log_freq', 20))}",
        f"--save_freq={save_freq}",
        "--save_checkpoint=true",
        "--eval_freq=0",
        "--wandb.enable=false",
        f"--output_dir={output_dir}",
        f"--job_name={job_name}",
    ]


def _mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    """读取必需映射配置段并给出明确错误。"""
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"配置段 {key!r} 必须是映射")
    return value


def _bool_text(value: Any) -> str:
    """把配置布尔值转换为 LeRobot CLI 文本。"""
    if not isinstance(value, bool):
        raise ValueError(f"期望布尔值，实际为 {value!r}")
    return "true" if value else "false"


def main(argv: Sequence[str] | None = None) -> int:
    """校验最小入口条件并启动训练。

    Args:
        argv: 测试时可注入的命令行参数。

    Returns:
        训练子进程退出码。
    """
    args = build_parser().parse_args(argv)
    config = load_yaml_config(args.config)
    dataset_root = resolve_path(args.dataset_root)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"数据集目录不存在: {dataset_root}")

    train = _mapping(config, "train")
    output_dir = resolve_path(args.output_dir or train.get("output_dir", "outputs/train/smolvla"))
    job_name = args.job_name or str(train.get("job_name", "smolvla_ur10e"))
    if output_dir.exists() and not args.dry_run:
        raise FileExistsError(f"训练输出目录已存在，避免覆盖: {output_dir}")

    executable = shutil.which("lerobot-train") or "lerobot-train"
    command = build_train_command(config, dataset_root, output_dir, job_name, args.smoke, executable)
    print("训练命令:")
    print(format_command(command))
    if args.dry_run:
        return 0
    log_path = output_dir.parent / f"{output_dir.name}.train.log"
    return run_logged(command, log_path)


if __name__ == "__main__":
    raise SystemExit(main())
