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
    clipped_mask: NDArray[np.bool_]
    clip_amount: NDArray[np.float32]


@dataclass(frozen=True)
class MotionLimits:
    """六关节参考轨迹的速度与加速度上限。"""

    velocity_limits_rad_s: NDArray[np.float64]
    acceleration_limits_rad_s2: NDArray[np.float64]


class JointMotionLimiter:
    """把绝对关节目标转换为满足二阶约束的参考轨迹。

    该限制器只作用于前六个连续机械臂关节。第七维夹爪指令原样透传，
    并且每条 rollout 必须以实际关节状态调用 ``reset`` 初始化。
    """

    def __init__(self, limits: MotionLimits, dt: float, arm_ctrlrange: NDArray[np.floating]) -> None:
        """初始化限制器并校验固定控制约束。"""
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("motion limiter的dt必须为有限正数")
        velocity = np.asarray(limits.velocity_limits_rad_s, dtype=np.float64)
        acceleration = np.asarray(limits.acceleration_limits_rad_s2, dtype=np.float64)
        ranges = np.asarray(arm_ctrlrange, dtype=np.float64)
        if velocity.shape != (6,) or acceleration.shape != (6,):
            raise ValueError("motion limiter速度和加速度上限必须均为六维")
        if not np.isfinite(velocity).all() or not np.isfinite(acceleration).all():
            raise ValueError("motion limiter上限必须为有限数")
        if np.any(velocity <= 0.0) or np.any(acceleration <= 0.0):
            raise ValueError("motion limiter上限必须为正数")
        if ranges.shape != (6, 2) or not np.isfinite(ranges).all() or np.any(ranges[:, 0] >= ranges[:, 1]):
            raise ValueError("motion limiter机械臂范围必须为有限(6, 2)区间")
        self._velocity_limits = velocity
        self._acceleration_limits = acceleration
        self._dt = float(dt)
        self._ranges = ranges
        self._reference_position: NDArray[np.float64] | None = None
        self._reference_velocity: NDArray[np.float64] | None = None

    @property
    def reference_velocity(self) -> NDArray[np.float64]:
        """返回当前六关节参考速度的副本。"""
        if self._reference_velocity is None:
            raise RuntimeError("motion limiter尚未reset")
        return self._reference_velocity.copy()

    def reset(self, actual_arm_qpos: NDArray[np.floating]) -> None:
        """以当前实际关节位置和零速度初始化一条新轨迹。"""
        position = np.asarray(actual_arm_qpos, dtype=np.float64)
        if position.shape != (6,) or not np.isfinite(position).all():
            raise ValueError("motion limiter初始关节位置必须为有限六维")
        self._reference_position = np.clip(position, self._ranges[:, 0], self._ranges[:, 1])
        self._reference_velocity = np.zeros(6, dtype=np.float64)

    def limit(self, safe_action: NDArray[np.floating]) -> tuple[NDArray[np.float32], NDArray[np.bool_], NDArray[np.float32]]:
        """限制一个已通过关节范围裁剪的七维动作。

        Returns:
            ``(最终动作, 发生二阶限制的掩码, 原目标减最终目标)``。
        """
        if self._reference_position is None or self._reference_velocity is None:
            raise RuntimeError("motion limiter必须在limit前reset")
        action = np.asarray(safe_action, dtype=np.float64)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("motion limiter输入必须为有限七维动作")
        target = np.clip(action[:6], self._ranges[:, 0], self._ranges[:, 1])
        desired_velocity = (target - self._reference_position) / self._dt
        desired_acceleration = (desired_velocity - self._reference_velocity) / self._dt
        acceleration = np.clip(
            desired_acceleration,
            -self._acceleration_limits,
            self._acceleration_limits,
        )
        velocity = np.clip(
            self._reference_velocity + acceleration * self._dt,
            -self._velocity_limits,
            self._velocity_limits,
        )
        next_position = self._reference_position + velocity * self._dt

        # 防止积分后的参考点跨越目标，跨越后精确停在当前模型目标。
        crossed = (target - self._reference_position) * (target - next_position) <= 0.0
        next_position = np.where(crossed, target, next_position)
        velocity = np.where(crossed, 0.0, velocity)
        next_position = np.clip(next_position, self._ranges[:, 0], self._ranges[:, 1])
        limited = action.copy()
        limited[:6] = next_position
        self._reference_position = next_position
        self._reference_velocity = velocity
        amount = action - limited
        mask = np.zeros(7, dtype=np.bool_)
        mask[:6] = ~np.isclose(action[:6], limited[:6], rtol=0.0, atol=1e-7)
        return limited.astype(np.float32), mask, amount.astype(np.float32)


def load_motion_limits(path: Path, expected_fps: int) -> MotionLimits:
    """读取并校验由专家动作标定脚本生成的锁定限制文件。"""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"motion limiter限制文件不存在: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"motion limiter限制文件不是有效JSON: {resolved}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("motion limiter限制文件schema_version必须为1")
    if int(payload.get("fps", -1)) != expected_fps:
        raise ValueError(
            f"motion limiter限制文件fps与评测不一致: "
            f"limits={payload.get('fps')}, evaluation={expected_fps}"
        )
    names = payload.get("joint_names")
    if names != ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]:
        raise ValueError("motion limiter限制文件关节顺序不兼容")
    velocity = np.asarray(payload.get("velocity_limits_rad_s"), dtype=np.float64)
    acceleration = np.asarray(payload.get("acceleration_limits_rad_s2"), dtype=np.float64)
    if velocity.shape != (6,) or acceleration.shape != (6,):
        raise ValueError("motion limiter限制文件速度和加速度上限必须均为六维")
    if not np.isfinite(velocity).all() or not np.isfinite(acceleration).all():
        raise ValueError("motion limiter限制文件上限必须为有限数")
    if np.any(velocity <= 0.0) or np.any(acceleration <= 0.0):
        raise ValueError("motion limiter限制文件上限必须为正数")
    return MotionLimits(velocity_limits_rad_s=velocity, acceleration_limits_rad_s2=acceleration)


def action_to_vector(action: Any) -> NDArray[np.float64]:
    """把Tensor或数组动作转换为有限七维向量。

    Args:
        action: Tensor、数组或七维序列，可带一个batch维。

    Returns:
        七维float64动作向量。

    Raises:
        ValueError: 动作shape错误或包含非有限值时抛出。
    """
    if hasattr(action, "detach"):
        action = action.detach().float().cpu().numpy()
    vector = np.asarray(action, dtype=np.float64).squeeze()
    if vector.shape != (7,):
        raise ValueError(f"策略动作必须是七维，实际 shape={vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError("策略动作包含 NaN 或无穷值")
    return vector


def convert_policy_action(action: Any, arm_ctrlrange: NDArray[np.floating]) -> SafeAction:
    """把策略输出转换为受限的七维 UR10e 目标命令。

    Args:
        action: Tensor、数组或七维序列，可带一个 batch 维。
        arm_ctrlrange: 六个机械臂 actuator 的 ``(6, 2)`` 控制范围。

    Returns:
        可执行动作及是否发生裁剪。
    """
    vector = action_to_vector(action)
    ranges = np.asarray(arm_ctrlrange, dtype=np.float64)
    if ranges.shape != (6, 2) or not np.isfinite(ranges).all():
        raise ValueError(f"机械臂控制范围必须是有限 (6, 2)，实际 shape={ranges.shape}")

    limited = vector.copy()
    limited[:6] = np.clip(limited[:6], ranges[:, 0], ranges[:, 1])
    limited[6] = np.clip(limited[6], 0.0, 1.0)
    clipped_mask = ~np.isclose(vector, limited, rtol=0.0, atol=1e-7)
    clip_amount = vector - limited
    return SafeAction(
        command=limited.astype(np.float32),
        clipped=bool(clipped_mask.any()),
        clipped_mask=clipped_mask,
        clip_amount=clip_amount.astype(np.float32),
    )


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


def write_pillow_line_plot(
    path: Path,
    values: np.ndarray,
    title: str,
    marker_index: int | None,
) -> None:
    """在无 Matplotlib 环境中用 Pillow 生成可读折线图。

    Args:
        path: PNG 输出路径。
        values: 一维有限数值序列。
        title: 图标题。
        marker_index: 可选的竖线索引，例如前 10 步边界使用 9。
    """
    from PIL import Image, ImageDraw

    sequence = np.asarray(values, dtype=np.float64).reshape(-1)
    if sequence.size == 0 or not np.isfinite(sequence).all():
        raise ValueError("Pillow 折线图需要非空有限数值")
    width, height = 1000, 500
    left, right, top, bottom = 85, 35, 65, 65
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((left, 20), title, fill="black")
    draw.line((left, top, left, height - bottom), fill="black", width=2)
    draw.line((left, height - bottom, width - right, height - bottom), fill="black", width=2)
    lower = float(np.min(sequence))
    upper = float(np.max(sequence))
    span = upper - lower if upper > lower else 1.0
    x_span = width - left - right
    y_span = height - top - bottom
    points = []
    for index, value in enumerate(sequence):
        ratio = index / max(1, sequence.size - 1)
        x = left + int(round(ratio * x_span))
        y = height - bottom - int(round((float(value) - lower) / span * y_span))
        points.append((x, y))
    if len(points) > 1:
        draw.line(points, fill=(31, 119, 180), width=3)
    else:
        x, y = points[0]
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(31, 119, 180))
    if marker_index is not None and 0 <= marker_index < sequence.size:
        marker_x = points[marker_index][0]
        draw.line((marker_x, top, marker_x, height - bottom), fill=(214, 39, 40), width=2)
    draw.text((10, top), f"max={upper:.5g}", fill="black")
    draw.text((10, height - bottom - 12), f"min={lower:.5g}", fill="black")
    draw.text((left, height - bottom + 20), "1", fill="black")
    draw.text((width - right - 30, height - bottom + 20), str(sequence.size), fill="black")
    image.save(path, format="PNG")
