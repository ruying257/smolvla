"""本机模型评测使用的路径、动作和结果工具。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from numpy.typing import NDArray


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml_config(path: Path) -> dict[str, Any]:
    """读取 YAML 配置并要求根节点为映射。

    Args:
        path: YAML 配置文件路径。

    Returns:
        配置字典。
    """
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"配置文件不存在: {resolved}")
    content = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError(f"配置根节点必须是映射: {resolved}")
    return content


def resolve_path(value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    """把相对路径解析为相对于项目根目录的绝对路径。

    Args:
        value: 绝对路径、相对路径或包含 ``~`` 的路径。
        base: 相对路径的解析基准。

    Returns:
        解析后的绝对路径。
    """
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


@dataclass(frozen=True)
class SafeAction:
    """策略动作经过形状检查和仿真限位后的结果。"""

    command: NDArray[np.float32]
    clipped: bool


def convert_policy_action(action: Any, arm_ctrlrange: NDArray[np.floating]) -> SafeAction:
    """把策略输出转换为受限的七维 UR10e 目标命令。

    Args:
        action: Tensor、数组或七维序列，可带一个 batch 维。
        arm_ctrlrange: 六个机械臂 actuator 的 ``(6, 2)`` 控制范围。

    Returns:
        可执行动作及是否发生裁剪。
    """
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    vector = np.asarray(action, dtype=np.float64).squeeze()
    ranges = np.asarray(arm_ctrlrange, dtype=np.float64)
    if vector.shape != (7,):
        raise ValueError(f"策略动作必须是七维，实际 shape={vector.shape}")
    if ranges.shape != (6, 2) or not np.isfinite(ranges).all():
        raise ValueError(f"机械臂控制范围必须是有限 (6, 2)，实际 shape={ranges.shape}")
    if not np.isfinite(vector).all():
        raise ValueError("策略动作包含 NaN 或无穷值")

    limited = vector.copy()
    limited[:6] = np.clip(limited[:6], ranges[:, 0], ranges[:, 1])
    limited[6] = np.clip(limited[6], 0.0, 1.0)
    clipped = not np.allclose(vector, limited, rtol=0.0, atol=1e-7)
    return SafeAction(limited.astype(np.float32), clipped)


def find_pretrained_model(path: Path) -> Path:
    """从模型目录、step checkpoint 或训练输出中定位完整模型目录。

    Args:
        path: 候选 pretrained_model、checkpoint 或训练输出目录。

    Returns:
        包含配置、权重和策略处理器的目录。
    """
    root = path.expanduser().resolve()
    direct_candidates = (root, root / "pretrained_model", root / "checkpoints" / "last" / "pretrained_model")
    for candidate in direct_candidates:
        if _is_complete_pretrained_model(candidate):
            return candidate.resolve()

    checkpoints = root / "checkpoints"
    if checkpoints.is_dir():
        numeric = sorted(
            (item for item in checkpoints.iterdir() if item.is_dir() and item.name.isdigit()),
            key=lambda item: int(item.name),
            reverse=True,
        )
        for checkpoint in numeric:
            candidate = checkpoint / "pretrained_model"
            if _is_complete_pretrained_model(candidate):
                return candidate.resolve()
    raise FileNotFoundError(
        "找不到完整 pretrained_model；必须包含 config.json、model.safetensors、"
        "policy_preprocessor.json 和 policy_postprocessor.json: "
        f"{root}"
    )


def _is_complete_pretrained_model(path: Path) -> bool:
    """检查目录是否包含闭环推理所需的核心 checkpoint 文件。"""
    required = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    )
    return path.is_dir() and all((path / name).is_file() for name in required)


def percentile(values: Iterable[float], q: float) -> float:
    """计算有限样本百分位数，空输入返回零。"""
    sequence = [float(value) for value in values if math.isfinite(float(value))]
    return 0.0 if not sequence else float(np.percentile(sequence, q))


def write_json(path: Path, value: Any) -> None:
    """以稳定 UTF-8 格式写入 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
