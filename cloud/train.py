"""根据项目 YAML 配置调用 LeRobot SmolVLA 训练入口。"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

from cloud.common import PROJECT_ROOT, format_command, load_yaml_config, resolve_path, run_logged


def build_parser() -> argparse.ArgumentParser:
    """创建训练命令行解析器。"""
    parser = argparse.ArgumentParser(description="启动 SmolVLA 云端训练 v2")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "smolvla-data" / "smolvla_ur10e_grounding_v2",
        help="完整 LeRobot 数据集目录，默认使用项目内 smolvla-data/smolvla_ur10e_grounding_v2",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "cloud_train_Tencent.yaml",
        help="训练 YAML 配置",
    )
    parser.add_argument("--output-dir", type=Path, help="覆盖配置中的训练输出目录")
    parser.add_argument("--job-name", help="覆盖配置中的任务名称")
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="从已有训练输出的 checkpoint 目录（如 outputs/train/xxx/checkpoints/last）恢复训练；"
        "使用 --resume=true 完整恢复 step、优化器与调度器状态",
    )
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
    resume_from: Path | None = None,
) -> list[str]:
    """构造不经过 shell 插值的 LeRobot 训练参数列表。

    Args:
        config: ``cloud_train_Tencent.yaml`` 配置。
        dataset_root: 完整 LeRobot 数据集目录。
        output_dir: 本次训练输出目录，调用前不得存在。
        job_name: LeRobot 任务名称。
        smoke: 是否覆盖为单步 smoke 训练（与 ``resume_from`` 互斥）。
        executable: LeRobot CLI 可执行文件名。
        resume_from: 非空时从该 checkpoint 目录（如 ``.../checkpoints/last``）恢复训练；
            此时 ``train.steps`` 表示总步数（训练循环从恢复的 step 执行到 ``steps``）。

    Returns:
        可传给 ``subprocess`` 的参数列表。
    """
    train = _mapping(config, "train")
    dataset = _mapping(config, "dataset")

    if resume_from is not None and smoke:
        raise ValueError(
            "--smoke 与 --resume-from 不能同时使用：resume 恢复的 step>0，smoke 的 steps=1 无法生效"
        )
    if resume_from is not None:
        return _build_resume_command(
            train, dataset, config.get("resume", {}), dataset_root, output_dir, job_name, resume_from, executable
        )

    policy = _mapping(config, "policy")
    steps = 1 if smoke else int(train.get("steps", 20_000))
    batch_size = 1 if smoke else int(train.get("batch_size", 1))
    save_freq = 1 if smoke else int(train.get("save_freq", steps))
    command = [
        executable,
        f"--policy.path={policy.get('model_id', 'lerobot/smolvla_base')}",
        "--policy.input_features=null",
        "--policy.output_features=null",
        f"--policy.empty_cameras={int(policy.get('empty_cameras', 1))}",
        f"--policy.device={policy.get('device', 'cuda')}",
        f"--policy.use_amp={_bool_text(policy.get('use_amp', True))}",
        "--policy.push_to_hub=false",
        f"--dataset.repo_id={dataset.get('repo_id', 'smolvla_ur10e_grounding_v2')}",
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
        "--wandb.enable=true",
        f"--output_dir={output_dir}",
        f"--job_name={job_name}",
    ]
    _append_image_transforms_args(command, dataset)
    return command


def _build_resume_command(
    train: dict[str, Any],
    dataset: dict[str, Any],
    resume_cfg: dict[str, Any],
    dataset_root: Path,
    output_dir: Path,
    job_name: str,
    resume_from: Path,
    executable: str,
) -> list[str]:
    """构造从已有 checkpoint 恢复训练的 LeRobot 参数列表。

    恢复机制：``--resume=true`` + ``--config_path=<checkpoint>/pretrained_model/train_config.json``。
    ``TrainPipelineConfig.from_pretrained`` 从 checkpoint 配置加载完整训练配置（含策略超参、
    优化器与调度器段），其余 CLI 参数作为覆盖项；``load_training_state`` 恢复 step、优化器、
    调度器与 RNG 状态。因此此处不传任何 ``--policy.*`` 参数，且 ``train.steps`` 为总步数。

    注意：调度器会按新总步数重建后再套用已保存状态，若总步数改变其自动缩放会导致恢复点
    学习率跳升；必须用 ``resume_cfg`` 的 ``scheduler_num_warmup_steps`` /
    ``scheduler_num_decay_steps`` 复刻原有效调度（如原运行 10000 步配置自动缩放为
    warmup 333 / decay 10000），保证学习率平滑衔接。

    Args:
        train: 配置的 ``train`` 段。
        dataset: 配置的 ``dataset`` 段。
        resume_cfg: 配置的 ``resume`` 段。
        dataset_root: 完整 LeRobot 数据集目录。
        output_dir: 本次训练输出目录（resume 时可为新目录，训练状态从 checkpoint 恢复）。
        job_name: LeRobot 任务名称。
        resume_from: 已有训练输出的 checkpoint 目录，必须包含
            ``pretrained_model/train_config.json``。
        executable: LeRobot CLI 可执行文件名。

    Returns:
        可传给 ``subprocess`` 的参数列表。
    """
    steps = int(train.get("steps", 20_000))
    train_config = resume_from / "pretrained_model" / "train_config.json"
    command = [
        executable,
        "--resume=true",
        f"--config_path={train_config}",
        f"--dataset.repo_id={dataset.get('repo_id', 'smolvla_ur10e_grounding_v2')}",
        f"--dataset.root={dataset_root}",
        f"--dataset.video_backend={dataset.get('video_backend', 'pyav')}",
        f"--batch_size={int(train.get('batch_size', 8))}",
        f"--steps={steps}",
        f"--num_workers={int(train.get('num_workers', 2))}",
        f"--seed={int(train.get('seed', 1000))}",
        f"--log_freq={int(train.get('log_freq', 20))}",
        f"--save_freq={int(train.get('save_freq', steps))}",
        "--save_checkpoint=true",
        "--eval_freq=0",
        "--wandb.enable=true",
        f"--output_dir={output_dir}",
        f"--job_name={job_name}",
    ]
    warmup = resume_cfg.get("scheduler_num_warmup_steps")
    decay = resume_cfg.get("scheduler_num_decay_steps")
    if warmup is not None:
        command.append(f"--scheduler.num_warmup_steps={int(warmup)}")
    if decay is not None:
        command.append(f"--scheduler.num_decay_steps={int(decay)}")
    _append_image_transforms_args(command, dataset)
    return command


def _append_image_transforms_args(command: list[str], dataset: dict[str, Any]) -> None:
    """把配置中的 ``dataset.image_transforms`` 段序列化为 LeRobot CLI 参数。

    域随机化训练增强：``--dataset.image_transforms.enable=true`` 后，数据集每次采样
    都会对每帧施加随机子集增强（``RandomSubsetApply``）。

    注意：draccus 不支持嵌套的 ``--dataset.image_transforms.tfs.<name>.*`` 参数，
    整个 ``tfs`` 字典必须作为单个 JSON 字符串传入 ``--dataset.image_transforms.tfs=<json>``
    （已实测可被正确解析为 ``dict[str, ImageTransformConfig]``）。

    Args:
        command: 正在构造的命令列表，原地追加参数。
        dataset: 配置的 ``dataset`` 段。
    """
    transforms = dataset.get("image_transforms")
    if not transforms:
        return
    if not isinstance(transforms, dict):
        raise ValueError("配置段 dataset.image_transforms 必须是映射")

    enable = transforms.get("enable", True)
    if not isinstance(enable, bool):
        raise ValueError("dataset.image_transforms.enable 必须是布尔值")
    command.append(f"--dataset.image_transforms.enable={'true' if enable else 'false'}")

    max_num = transforms.get("max_num_transforms")
    if max_num is not None:
        command.append(f"--dataset.image_transforms.max_num_transforms={int(max_num)}")

    random_order = transforms.get("random_order")
    if random_order is not None:
        if not isinstance(random_order, bool):
            raise ValueError("dataset.image_transforms.random_order 必须是布尔值")
        command.append(f"--dataset.image_transforms.random_order={'true' if random_order else 'false'}")

    tfs = transforms.get("tfs")
    if tfs is not None:
        if not isinstance(tfs, dict):
            raise ValueError("dataset.image_transforms.tfs 必须是映射")
        command.append(f"--dataset.image_transforms.tfs={json.dumps(tfs, ensure_ascii=False)}")


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
    resume_from = resolve_path(args.resume_from) if args.resume_from else None
    if resume_from is not None and not args.dry_run:
        resume_train_config = resume_from / "pretrained_model" / "train_config.json"
        if not resume_train_config.is_file():
            raise FileNotFoundError(
                f"resume checkpoint 缺少 train_config.json: {resume_train_config}"
            )

    train = _mapping(config, "train")
    output_dir = resolve_path(args.output_dir or train.get("output_dir", "outputs/train/smolvla_ur10e_grounding_v2"))
    job_name = args.job_name or str(train.get("job_name", "smolvla_ur10e_grounding_v2"))    
    if output_dir.exists() and not args.dry_run:
        raise FileExistsError(f"训练输出目录已存在，避免覆盖: {output_dir}")

    executable = shutil.which("lerobot-train") or "lerobot-train"
    command = build_train_command(
        config, dataset_root, output_dir, job_name, args.smoke, executable, resume_from
    )
    print("训练命令:")
    print(format_command(command))
    if args.dry_run:
        return 0
    log_path = output_dir.parent / f"{output_dir.name}.train.log"
    return run_logged(command, log_path)


if __name__ == "__main__":
    raise SystemExit(main())
