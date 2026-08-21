"""在本机 MuJoCo 环境中执行可复现、可恢复的 SmolVLA 闭环评测。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import imageio.v2 as imageio
import numpy as np

from collector.common import task_spec as task_spec_module
from evaluate.common import (
    JointMotionLimiter,
    MotionLimits,
    PROJECT_ROOT,
    action_to_vector,
    convert_policy_action,
    find_pretrained_model,
    load_motion_limits,
    load_yaml_config,
    percentile,
    resolve_path,
    write_json,
)
from sim.environment import CleanTabletopEnv
from sim.mug_environment import MUG_TASK_IDS, MugTabletopEnv, resolve_mug_texture_path


TASKS = task_spec_module.TASKS
MUG_PROMPTS = {
    "mug_on_blue": "Put the mug on the blue pad.",
    "mug_on_yellow": "Put the mug on the yellow pad.",
}
UNSEEN_TEMPLATE = "Move the {cube_color} cube to the {pad_color} pad."
RESULTS_JSONL = "rollouts.jsonl"
MANIFEST_JSON = "run_manifest.json"
UPDATE_JSON = "rollout_update.json"
ACTION_TRACE_DIR = "action_traces"
ACTION_CLIPPING_JSON = "action_clipping_summary.json"
ACTION_CLIPPING_CSV = "action_clipping_by_dimension.csv"
MOTION_METRICS_CSV = "motion_metrics_by_rollout.csv"
MOTION_METRICS_JSON = "motion_metrics_summary.json"
MOTION_METRICS_REPORT = "motion_metrics_report.md"
ACTION_DIMENSIONS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
    "gripper",
)


@dataclass(frozen=True)
class RolloutSpec:
    """一条评测轨迹的不可变实验条件。"""

    scene_seed: int
    task_id: str
    prompt_type: str
    policy_seed: int

    @property
    def key(self) -> str:
        """返回可用于恢复和去重的稳定实验键。"""
        return f"scene={self.scene_seed}|task={self.task_id}|prompt={self.prompt_type}|policy={self.policy_seed}"


@dataclass
class RolloutResult:
    """一条闭环评测记录。"""

    rollout_key: str
    scene_seed: int
    policy_seed: int
    task_id: str
    task: str
    prompt_type: str
    success: bool
    failure_mode: str
    steps: int
    elapsed_seconds: float
    latency_mean_ms: float
    latency_p95_ms: float
    clipped_action_steps: int
    clipped_action_rate: float
    action_trace_path: str
    checkpoint_sha256: str
    video_path: str
    video_retained: bool
    error: str
    completed_at: str


def build_parser() -> argparse.ArgumentParser:
    """创建本机闭环评测命令行解析器。"""
    parser = argparse.ArgumentParser(description="执行 SmolVLA UR10e 本机闭环评测")
    parser.add_argument("--checkpoint", type=Path, required=True, help="模型、checkpoint 或训练输出目录")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "eval.yaml",
        help="评测 YAML 配置",
    )
    parser.add_argument("--output-dir", type=Path, help="覆盖配置中的评测输出目录")
    parser.add_argument("--device", default="cuda", help="策略推理设备")
    parser.add_argument(
        "--scene-seed",
        type=int,
        help="只评测配置中指定的一个scene seed；保留该配置的全部任务、措辞和policy seed",
    )
    parser.add_argument("--max-rollouts", type=int, help="只执行前 N 条 rollout，用于 smoke test")
    parser.add_argument("--max-steps", type=int, help="覆盖每条 rollout 的最大控制步数")
    parser.add_argument(
        "--execution-horizon",
        type=int,
        help="每个动作chunk实际执行的步数；例如10表示预测50步但只执行前10步后重规划",
    )
    parser.add_argument(
        "--chunk-blend",
        type=int,
        default=0,
        help="模型输出层chunk衔接平滑帧数K（默认0关闭）。重预测时用旧chunk尾帧作锚点，"
        "对新chunk前K帧做带角度回卷的线性插值；夹爪维度透传不插值。用于消除重预测边界的方向突变/抖动。",
    )
    parser.add_argument("--resume", action="store_true", help="校验manifest后续跑未完成或异常轨迹")
    parser.add_argument(
        "--rerun-failures",
        action="store_true",
        help="覆盖更新已有结果中的全部失败rollout；不可与--resume、--scene-seed或--max-rollouts组合",
    )
    parser.add_argument("--prune-videos", action="store_true", help="裁剪视频：仅保留全部失败视频及每个任务措辞组合的首条成功视频（默认保留全部视频）")
    return parser


def build_prompt(task_id: str, prompt_type: str) -> str:
    """构造训练已见或训练未见的任务文本。

    Args:
        task_id: 四类搬运任务之一。
        prompt_type: ``canonical``、``synonym`` 或 ``unseen``。

    Returns:
        英文任务指令。
    """
    if task_id in MUG_PROMPTS:
        if prompt_type != "canonical":
            raise ValueError("杯子评测目前只支持训练使用的canonical措辞")
        return MUG_PROMPTS[task_id]
    task = TASKS[task_id]
    if prompt_type in {"canonical", "synonym"}:
        return task.prompt(prompt_type)
    if prompt_type == "unseen":
        return UNSEEN_TEMPLATE.format(cube_color=task.cube_color, pad_color=task.pad_color)
    raise ValueError(f"未知 prompt_type={prompt_type!r}")


def set_policy_seed(policy_seed: int) -> None:
    """固定 Python、NumPy、PyTorch 和 CUDA 的模型采样随机种子。

    Args:
        policy_seed: 当前轨迹使用的模型采样种子。
    """
    import torch

    random.seed(policy_seed)
    np.random.seed(policy_seed % (2**32))
    torch.manual_seed(policy_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(policy_seed)


def make_policy_observation(images: dict[str, np.ndarray], state: np.ndarray, task: str) -> dict[str, Any]:
    """把 MuJoCo 观测转换为 LeRobot 策略的 batch 输入。

    Args:
        images: 包含 ``agent`` 和 ``wrist`` 的 HWC RGB uint8 图像。
        state: 七维当前状态。
        task: 英文任务文本。

    Returns:
        带 batch 维的Tensor字典和任务列表。
    """
    import torch

    if set(images) != {"agent", "wrist"}:
        raise ValueError(f"策略相机键必须为 agent/wrist，实际为 {set(images)}")
    observation: dict[str, Any] = {"task": [task]}
    for key, image in images.items():
        array = np.asarray(image)
        if array.shape != (256, 256, 3) or array.dtype != np.uint8:
            raise ValueError(f"{key} 图像必须是 256x256x3 uint8")
        observation[f"observation.images.{key}"] = (
            torch.from_numpy(array.copy()).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        )
    state_array = np.asarray(state, dtype=np.float32)
    if state_array.shape != (7,) or not np.isfinite(state_array).all():
        raise ValueError("状态必须是有限七维向量")
    observation["observation.state"] = torch.from_numpy(state_array).unsqueeze(0)
    return observation


def resolve_execution_horizon(
    checkpoint: Path,
    evaluation: dict[str, Any],
    cli_value: int | None,
) -> tuple[int, int]:
    """确定动作chunk长度和本次实际执行步数。

    Args:
        checkpoint: 完整pretrained_model目录。
        evaluation: YAML中的evaluation配置段。
        cli_value: 命令行覆盖值，未提供时为None。

    Returns:
        ``(execution_horizon, chunk_size)``。

    Raises:
        ValueError: 步数不是正数或超过模型chunk长度时抛出。
    """
    policy_config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    chunk_size = int(policy_config.get("chunk_size", 1))
    checkpoint_horizon = int(policy_config.get("n_action_steps", chunk_size))
    configured = evaluation.get("execution_horizon", checkpoint_horizon)
    execution_horizon = int(cli_value if cli_value is not None else configured)
    if execution_horizon <= 0:
        raise ValueError("execution-horizon必须大于零")
    if execution_horizon > chunk_size:
        raise ValueError(
            f"execution-horizon不能超过模型chunk_size: {execution_horizon} > {chunk_size}"
        )
    return execution_horizon, chunk_size


def resolve_motion_limiter(evaluation: dict[str, Any], fps: int) -> dict[str, Any]:
    """解析评测配置中的可选二阶关节限制器。"""
    section = evaluation.get("motion_limiter", {})
    if section is None:
        section = {}
    if not isinstance(section, dict):
        raise ValueError("motion_limiter配置必须是映射")
    enabled = bool(section.get("enabled", False))
    if not enabled:
        return {"enabled": False}
    raw_path = section.get("limits_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("启用motion_limiter时必须提供limits_path")
    path = resolve_path(raw_path)
    limits = load_motion_limits(path, fps)
    # 在此处提前触发构造校验，避免运行到第一条轨迹才发现标定文件非法。
    MotionLimits(limits.velocity_limits_rad_s, limits.acceleration_limits_rad_s2)
    return {
        "enabled": True,
        "limits_path": str(path),
        "limits_sha256": sha256_file(path),
        "velocity_limits_rad_s": limits.velocity_limits_rad_s.astype(float).tolist(),
        "acceleration_limits_rad_s2": limits.acceleration_limits_rad_s2.astype(float).tolist(),
    }


def load_policy_bundle(
    checkpoint: Path,
    device: str,
    execution_horizon: int,
) -> tuple[Any, Callable[[Any], Any], Callable[[Any], Any]]:
    """加载checkpoint策略并覆盖实际执行的动作步数。

    Args:
        checkpoint: 完整pretrained_model目录。
        device: 策略推理设备。
        execution_horizon: 每次动作chunk实际放入执行队列的步数。

    Returns:
        策略、输入预处理器和动作后处理器。
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    config = PreTrainedConfig.from_pretrained(
        str(checkpoint),
        cli_overrides=[
            f"--device={device}",
            "--push_to_hub=false",
            f"--n_action_steps={execution_horizon}",
        ],
    )
    policy_class = get_policy_class(config.type)
    policy = policy_class.from_pretrained(str(checkpoint), config=config)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor


def _wrap_angle(angle: NDArray[np.floating]) -> NDArray[np.floating]:
    """把角度差回卷到 [-pi, pi)，避免线性插值跨 ±pi 绕远路。

    关节角是循环量：对跨越 π 边界的两个角度直接线性插值会走一整圈的
    远路（例如 +3.0 与 -3.0 直插得到 0，实际只差 0.28 rad）。回卷后走
    最短弧。对差值本来就落在 [-pi, pi) 的情况是 no-op。
    """
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _tensor_to_numpy_action_chunk(chunk: Any) -> NDArray[np.float64]:
    """把策略预测的动作 chunk 张量转换为 ``(steps, 7)`` float64 数组。"""
    if hasattr(chunk, "detach"):
        chunk = chunk.detach().cpu()
    array = np.asarray(chunk)
    if array.ndim == 3:  # (batch, steps, dim) -> (steps, dim)
        array = array[0]
    return np.asarray(array, dtype=np.float64)


class ChunkBlendPolicy:
    """在策略重预测时对动作 chunk 前 K 帧做衔接平滑（模型输出层）。

    动机：SmolVLA 按 ``n_action_steps`` 预测一段动作 chunk 并逐帧执行，
    队列耗尽时重新预测。新 chunk 的起点与旧 chunk 尾部在输出空间不连续，
    造成周期性方向突变/抖动（前序分析：边界模型输出跳变是 chunk 内部的
    1.3–2.1 倍）。本包装器在每次重预测时，用旧 chunk 的尾帧作为锚点，
    对新 chunk 前 K 帧做带角度回卷的线性插值，使衔接连续。

    两个保护：
    - **角度回卷**：前 6 个关节角是循环量，插值前把角度差回卷到 [-pi, pi)，
      避免跨 π 时多转一整圈。
    - **夹爪保护**：第 7 维夹爪是离散语义（开/合），不参与插值，直接透传
      新 chunk 的值，避免产生"半开半合"的无意义中间夹持力。

    ``blend_frames=0`` 时退化为直接透传底层策略，行为完全等同未包装。

    通过标准 ``predict_action_chunk`` 接口预测整段 chunk 并自行管理动作
    队列，因此不依赖 SmolVLA 内部的私有队列结构，对其他 chunking 策略
    （ACT 等）同样适用。
    """

    def __init__(self, policy: Any, blend_frames: int) -> None:
        """包一层策略，并在重预测时对 chunk 前 K 帧做衔接平滑。"""
        self._policy = policy
        self._k = max(0, int(blend_frames))
        self._queue: list[NDArray[np.float64]] = []
        self._prev_chunk: NDArray[np.float64] | None = None
        self._out_device: Any = "cpu"
        self._out_dtype: Any = None

    @property
    def config(self) -> Any:
        """转发到底层策略配置，便于运行入口覆盖 ``n_action_steps``。"""
        return self._policy.config

    def reset(self) -> None:
        """重置底层策略与包装器的队列和衔接状态。"""
        if hasattr(self._policy, "reset"):
            self._policy.reset()
        self._queue = []
        self._prev_chunk = None

    def select_action(self, batch: Any, **kwargs: Any) -> Any:
        """返回单帧动作；队列耗尽时预测整段新 chunk 并做衔接平滑。

        返回值与底层 ``select_action`` 保持一致（torch Tensor），以便策略
        后处理器（postprocessor）能正常解析；blend 计算在 numpy 侧完成。
        """
        if not self._queue:
            import torch

            chunk = self._policy.predict_action_chunk(batch, **kwargs)
            if torch.is_tensor(chunk):
                self._out_device = chunk.device
                self._out_dtype = chunk.dtype
            else:
                self._out_device = "cpu"
                self._out_dtype = torch.float32
            chunk_np = _tensor_to_numpy_action_chunk(chunk)
            horizon = int(getattr(self._policy.config, "n_action_steps", len(chunk_np)))
            chunk_np = chunk_np[:horizon]
            chunk_np = self._blend_chunk(chunk_np)
            self._prev_chunk = chunk_np.copy()
            self._queue = [chunk_np[i] for i in range(len(chunk_np))]
        import torch

        frame = self._queue.pop(0)
        dtype = self._out_dtype if self._out_dtype is not None else torch.float32
        return torch.as_tensor(frame, device=self._out_device, dtype=dtype)

    def _blend_chunk(self, chunk: NDArray[np.float64]) -> NDArray[np.float64]:
        """用旧 chunk 尾帧作锚点，对新 chunk 前 K 帧做带回卷的线性插值。"""
        if self._k <= 0 or self._prev_chunk is None:
            return chunk
        anchor = self._prev_chunk[-1]  # 旧 chunk 尾帧，边界时实际到达的目标
        horizon = chunk.shape[0]
        k = min(self._k, horizon)
        blended = chunk.copy()
        for t in range(k):
            if k == 1:
                weight = 0.5
            else:
                weight = (t + 1) / k  # t=0 -> 1/k 小幅前进，t=k-1 -> 1 贴合新 chunk
            for i in range(6):  # 关节角：回卷后向锚点靠拢
                delta = _wrap_angle(chunk[t, i] - anchor[i])
                blended[t, i] = anchor[i] + weight * delta
            blended[t, 6] = chunk[t, 6]  # 夹爪：透传新 chunk 值，不插值
        return blended


def rollout_artifact_stem(spec: RolloutSpec) -> str:
    """生成视频和动作日志共用的稳定文件名。"""
    return f"scene_{spec.scene_seed}_policy_{spec.policy_seed}_{spec.task_id}_{spec.prompt_type}"


def write_action_trace_record(file: Any, record: dict[str, Any]) -> None:
    """立即追加并刷新一条动作诊断记录。

    Args:
        file: 已打开的JSONL文本文件。
        record: 当前控制步的动作、限位和裁剪信息。
    """
    file.write(json.dumps(record, ensure_ascii=False) + "\n")
    file.flush()


def run_single_rollout(
    policy: Any,
    preprocessor: Callable[[Any], Any],
    postprocessor: Callable[[Any], Any],
    spec: RolloutSpec,
    output_dir: Path,
    fps: int,
    max_steps: int,
    device: str,
    checkpoint_sha256: str,
    execution_horizon: int = 50,
    environment: str = "cube",
    appearance_variant: str = "original",
    motion_limiter_settings: dict[str, Any] | None = None,
    chunk_blend: int = 0,
) -> RolloutResult:
    """执行一条固定场景和模型随机种子的闭环rollout。

    Args:
        policy: 已加载的LeRobot策略。
        preprocessor: 策略输入预处理器。
        postprocessor: 策略动作后处理器。
        spec: 当前轨迹的完整实验条件。
        output_dir: 评测产物目录。
        fps: 控制与输出视频帧率。
        max_steps: 单条rollout最大控制步数。
        device: 策略推理设备。
        checkpoint_sha256: 模型权重SHA-256。
        execution_horizon: 每个预测chunk实际执行的步数。
        environment: ``cube``或``mug``仿真环境。
        appearance_variant: 杯子环境使用的视觉外观变体。
        motion_limiter_settings: 已解析的限制器配置；未启用时为``None``或``enabled=false``。

    Returns:
        单条闭环评测结果。
    """
    import torch

    task_text = build_prompt(spec.task_id, spec.prompt_type)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_stem = rollout_artifact_stem(spec)
    video_path = output_dir / "videos" / f"{artifact_stem}.mp4"
    action_trace_path = output_dir / ACTION_TRACE_DIR / f"{artifact_stem}.jsonl"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    action_trace_path.parent.mkdir(parents=True, exist_ok=True)
    latencies: list[float] = []
    clipped_steps = 0
    failure_mode = "timeout"
    error = ""
    completed_steps = 0
    rollout_start = time.perf_counter()

    try:
        set_policy_seed(spec.policy_seed)
        if execution_horizon <= 0:
            raise ValueError("execution_horizon必须大于零")
        if hasattr(policy.config, "n_action_steps"):
            chunk_size = int(getattr(policy.config, "chunk_size", execution_horizon))
            if execution_horizon > chunk_size:
                raise ValueError(
                    f"execution_horizon不能超过模型chunk_size: {execution_horizon} > {chunk_size}"
                )
            policy.config.n_action_steps = execution_horizon
        env_context = (
            MugTabletopEnv(appearance_variant=appearance_variant)
            if environment == "mug"
            else CleanTabletopEnv()
        )
        with env_context as env, imageio.get_writer(
            video_path,
            fps=fps,
            codec="libx264",
            quality=7,
            macro_block_size=None,
        ) as writer, action_trace_path.open("w", encoding="utf-8", newline="\n") as trace_file:
            env.reset(spec.scene_seed)
            policy.reset()
            physics_steps = max(1, round((1.0 / fps) / float(env.model.opt.timestep)))
            action_lower = np.concatenate([env.model.actuator_ctrlrange[:6, 0], [0.0]])
            action_upper = np.concatenate([env.model.actuator_ctrlrange[:6, 1], [1.0]])
            limiter: JointMotionLimiter | None = None
            limiter_enabled = bool(motion_limiter_settings and motion_limiter_settings.get("enabled", False))
            if limiter_enabled:
                limits = MotionLimits(
                    np.asarray(motion_limiter_settings["velocity_limits_rad_s"], dtype=np.float64),
                    np.asarray(motion_limiter_settings["acceleration_limits_rad_s2"], dtype=np.float64),
                )
                limiter = JointMotionLimiter(limits, 1.0 / fps, env.model.actuator_ctrlrange[:6])
                limiter.reset(np.asarray(env.get_state()[:6], dtype=np.float64))
            for step_index in range(max_steps):
                images = env.capture_training_images()
                writer.append_data(np.concatenate([images["agent"], images["wrist"]], axis=1))
                observation_state = env.get_state()
                observation = make_policy_observation(images, observation_state, task_text)
                started = time.perf_counter()
                processed = preprocessor(observation)
                autocast_context = (
                    torch.autocast(device_type="cuda")
                    if device.startswith("cuda") and bool(getattr(policy.config, "use_amp", False))
                    else nullcontext()
                )
                with torch.inference_mode(), autocast_context:
                    model_output = policy.select_action(processed)
                model_output_vector = action_to_vector(model_output)
                physical_action = postprocessor(model_output)
                latencies.append((time.perf_counter() - started) * 1000.0)
                physical_action_vector = action_to_vector(physical_action)
                safe_action = convert_policy_action(physical_action_vector, env.model.actuator_ctrlrange[:6])
                clipped_steps += int(safe_action.clipped)
                if limiter is None:
                    executed_action = safe_action.command
                    motion_limited_mask = np.zeros(7, dtype=np.bool_)
                    motion_limit_amount = np.zeros(7, dtype=np.float32)
                    reference_velocity = np.zeros(6, dtype=np.float64)
                else:
                    executed_action, motion_limited_mask, motion_limit_amount = limiter.limit(safe_action.command)
                    reference_velocity = limiter.reference_velocity
                env.apply_joint_action(executed_action, physics_steps=physics_steps)
                actual_state_after = env.get_state()
                write_action_trace_record(
                    trace_file,
                    {
                        "rollout_key": spec.key,
                        "step": step_index + 1,
                        "chunk_start": step_index % execution_horizon == 0,
                        "chunk_blend": chunk_blend,
                        "model_output": model_output_vector.tolist(),
                        "physical_action": physical_action_vector.tolist(),
                        "range_safe_action": safe_action.command.astype(np.float64).tolist(),
                        "executed_action": executed_action.astype(np.float64).tolist(),
                        "action_lower": action_lower.astype(np.float64).tolist(),
                        "action_upper": action_upper.astype(np.float64).tolist(),
                        "clipped_mask": safe_action.clipped_mask.tolist(),
                        "clip_amount": safe_action.clip_amount.astype(np.float64).tolist(),
                        "any_clipped": safe_action.clipped,
                        "observation_state": np.asarray(observation_state, dtype=np.float64).tolist(),
                        "motion_limiter_enabled": limiter_enabled,
                        "motion_limited_mask": motion_limited_mask.tolist(),
                        "motion_limit_amount": motion_limit_amount.astype(np.float64).tolist(),
                        "reference_velocity_rad_s": reference_velocity.astype(np.float64).tolist(),
                        "actual_qpos_after": np.asarray(actual_state_after[:6], dtype=np.float64).tolist(),
                        "actual_qvel_after": env.get_arm_qvel().astype(np.float64).tolist(),
                        "ee_position_after": env.get_end_effector_position().astype(np.float64).tolist(),
                    },
                )
                completed_steps = step_index + 1
                evaluation = env.evaluate_task(spec.task_id, completed_steps / fps, max_steps / fps)
                failure_mode = evaluation.failure_mode
                if evaluation.success or failure_mode in {
                    "wrong_cube",
                    "wrong_pad",
                    "dropped_or_out_of_bounds",
                    "control_exception",
                }:
                    break
            success = failure_mode == "success"
    except Exception as exc:
        success = False
        failure_mode = "control_exception"
        error = f"{type(exc).__name__}: {exc}"

    elapsed = time.perf_counter() - rollout_start
    return RolloutResult(
        rollout_key=spec.key,
        scene_seed=spec.scene_seed,
        policy_seed=spec.policy_seed,
        task_id=spec.task_id,
        task=task_text,
        prompt_type=spec.prompt_type,
        success=success,
        failure_mode=failure_mode,
        steps=completed_steps,
        elapsed_seconds=elapsed,
        latency_mean_ms=float(np.mean(latencies)) if latencies else 0.0,
        latency_p95_ms=percentile(latencies, 95),
        clipped_action_steps=clipped_steps,
        clipped_action_rate=clipped_steps / completed_steps if completed_steps else 0.0,
        action_trace_path=str(action_trace_path.resolve()),
        checkpoint_sha256=checkpoint_sha256,
        video_path=str(video_path.resolve()),
        video_retained=True,
        error=error,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )


def build_specs(evaluation: dict[str, Any], max_rollouts: int | None = None) -> list[RolloutSpec]:
    """从配置构造有序、唯一的完整实验矩阵。

    Args:
        evaluation: YAML中的evaluation映射。
        max_rollouts: 冒烟测试需要保留的前N条。

    Returns:
        有序实验条件列表。
    """
    scene_seeds = [int(value) for value in evaluation.get("scene_seeds", [10000])]
    environment = str(evaluation.get("environment", "cube"))
    if environment not in {"cube", "mug"}:
        raise ValueError(f"未知评测环境: {environment!r}，可选值为cube或mug")
    default_tasks = MUG_TASK_IDS if environment == "mug" else tuple(TASKS)
    task_ids = [str(value) for value in evaluation.get("task_ids", default_tasks)]
    prompt_types = [str(value) for value in evaluation.get("prompt_types", ["canonical"])]
    policy_seeds = [int(value) for value in evaluation.get("policy_seeds", [20260])]
    valid_tasks = set(MUG_TASK_IDS if environment == "mug" else TASKS)
    if any(task_id not in valid_tasks for task_id in task_ids):
        raise ValueError(f"评测包含未知任务: {task_ids}")
    if environment == "mug" and any(prompt != "canonical" for prompt in prompt_types):
        raise ValueError("杯子环境目前只支持canonical措辞")
    if any(prompt not in {"canonical", "synonym", "unseen"} for prompt in prompt_types):
        raise ValueError(f"评测包含未知措辞: {prompt_types}")
    dimensions = (scene_seeds, task_ids, prompt_types, policy_seeds)
    if any(not values for values in dimensions):
        raise ValueError("scene_seeds、task_ids、prompt_types和policy_seeds均不得为空")
    specs = [
        RolloutSpec(scene, task_id, prompt, policy_seed)
        for scene in scene_seeds
        for task_id in task_ids
        for prompt in prompt_types
        for policy_seed in policy_seeds
    ]
    if len({spec.key for spec in specs}) != len(specs):
        raise ValueError("评测配置产生重复实验键，请检查各维度是否包含重复值")
    if max_rollouts is not None:
        if max_rollouts <= 0:
            raise ValueError("max-rollouts必须大于零")
        specs = specs[:max_rollouts]
    return specs


def select_scene_specs(specs: Sequence[RolloutSpec], scene_seed: int | None) -> list[RolloutSpec]:
    """按命令行指定的单个场景筛选评测组合。

    Args:
        specs: 已由配置展开且完成唯一性校验的评测组合。
        scene_seed: 需要保留的场景种子；为None时保持全部组合。

    Returns:
        筛选后的评测组合。指定seed时保留该seed下全部任务、措辞和policy seed。

    Raises:
        ValueError: 指定的seed不属于配置中的scene_seeds时抛出。
    """
    if scene_seed is None:
        return list(specs)
    selected = [spec for spec in specs if spec.scene_seed == scene_seed]
    if not selected:
        available = sorted({spec.scene_seed for spec in specs})
        raise ValueError(f"scene-seed={scene_seed}不在配置场景中，可选值: {available}")
    return selected


def sha256_file(path: Path) -> str:
    """流式计算大文件SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_sha256() -> str:
    """计算影响评测行为的项目源码身份。"""
    digest = hashlib.sha256()
    task_spec_path = Path(task_spec_module.__file__).resolve()
    paths = sorted((PROJECT_ROOT / "evaluate").glob("*.py")) + [
        PROJECT_ROOT / "sim" / "environment.py",
        PROJECT_ROOT / "sim" / "mug_environment.py",
        task_spec_path,
        PROJECT_ROOT / "collector" / "v3" / "task_spec.py",
    ]
    for path in paths:
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_identity() -> dict[str, Any]:
    """读取当前Git提交和工作区状态，失败时返回明确占位值。"""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, encoding="utf-8"
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True, encoding="utf-8"
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unavailable", "dirty": None}


def package_versions(device: str) -> dict[str, Any]:
    """收集可影响评测复现的软件和GPU版本。"""
    import importlib.metadata
    import mujoco
    import torch

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "lerobot": importlib.metadata.version("lerobot"),
        "mujoco": mujoco.__version__,
        "gpu_name": gpu_name,
        "device": device,
    }


def build_manifest(
    checkpoint: Path,
    checkpoint_sha256: str,
    config: dict[str, Any],
    specs: list[RolloutSpec],
    device: str,
    fps: int,
    max_steps: int,
    keep_all_videos: bool,
    execution_horizon: int,
    chunk_size: int,
    appearance_variant: str,
    appearance_texture_sha256: str,
    motion_limiter: dict[str, Any],
    chunk_blend: int = 0,
) -> dict[str, Any]:
    """构造用于恢复校验的完整运行清单。"""
    return {
        "schema_version": 4,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "source_sha256": source_sha256(),
        "git": git_identity(),
        "environment": package_versions(device),
        "amp_enabled": bool(json.loads((checkpoint / "config.json").read_text(encoding="utf-8")).get("use_amp")),
        "fps": fps,
        "max_steps": max_steps,
        "chunk_size": chunk_size,
        "execution_horizon": execution_horizon,
        "appearance_variant": appearance_variant,
        "appearance_texture_sha256": appearance_texture_sha256,
        "motion_limiter": motion_limiter,
        "chunk_blend": chunk_blend,
        "keep_all_videos": keep_all_videos,
        "config": config,
        "rollout_count": len(specs),
        "rollout_keys": [spec.key for spec in specs],
    }


def manifest_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    """提取恢复时必须完全一致的manifest字段。

    ``config``、``rollout_count`` 和 ``rollout_keys`` 不参与校验——续跑
    的主要场景就是在相同checkpoint和运行参数下扩展评测矩阵(增加scene seed
    或 task)，此时只有rollout集合会变化，关键运行参数已通过其余字段校验。
    """
    keys = (
        "schema_version",
        "checkpoint_path",
        "checkpoint_sha256",
        "source_sha256",
        "environment",
        "amp_enabled",
        "fps",
        "max_steps",
        "chunk_size",
        "execution_horizon",
        "appearance_variant",
        "appearance_texture_sha256",
        "motion_limiter",
        "chunk_blend",
        "keep_all_videos",
    )
    identity = {key: manifest.get(key) for key in keys}
    # schema_version=4及更早版本尚未记录chunk_blend。缺失字段与当前默认
    # 关闭(0)完全等价，避免为默认值升级错误拒绝续跑。
    if identity["chunk_blend"] is None:
        identity["chunk_blend"] = 0
    return identity


def prepare_run(output_dir: Path, manifest: dict[str, Any], resume: bool) -> list[RolloutResult]:
    """创建新运行或校验旧运行，并返回可安全恢复的有效结果。"""
    manifest_path = output_dir / MANIFEST_JSON
    journal_path = output_dir / RESULTS_JSONL
    if not resume:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"输出目录已存在且非空；请更换目录或使用--resume: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(manifest_path, manifest)
        return []
    if not manifest_path.is_file():
        raise FileNotFoundError(f"续跑缺少{MANIFEST_JSON}: {output_dir}")
    existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest_identity(existing_manifest) != manifest_identity(manifest):
        raise ValueError("续跑manifest与当前checkpoint、代码、配置或环境不一致")
    results = load_jsonl(journal_path)
    retryable = [result for result in results if result.failure_mode == "control_exception" or result.error]
    valid = [result for result in results if result not in retryable]
    validate_completed_results(valid)
    if retryable:
        rewrite_jsonl(journal_path, valid)
    return valid


def artifact_paths(output_dir: Path, spec: RolloutSpec) -> tuple[Path, Path]:
    """返回一条rollout在当前输出目录中的视频和动作日志路径。"""
    stem = rollout_artifact_stem(spec)
    return (
        output_dir / "videos" / f"{stem}.mp4",
        output_dir / ACTION_TRACE_DIR / f"{stem}.jsonl",
    )


def update_manifest_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    """提取失败项重跑时必须保持一致的运行身份。

    成功判据修订会自然改变源码哈希，因此更新模式故意不比较
    ``source_sha256``；其余checkpoint、环境、配置和执行参数必须不变。
    """
    keys = (
        "checkpoint_path",
        "checkpoint_sha256",
        "environment",
        "amp_enabled",
        "fps",
        "max_steps",
        "chunk_size",
        "execution_horizon",
        "appearance_variant",
        "appearance_texture_sha256",
        "motion_limiter",
        "chunk_blend",
        "keep_all_videos",
        "config",
    )
    identity = {key: manifest.get(key) for key in keys}
    # schema_version=3及更早版本尚未记录motion_limiter。缺失字段与当前
    # 配置未启用限制器完全等价，避免为默认值升级错误拒绝失败项重跑。
    if identity["motion_limiter"] is None:
        identity["motion_limiter"] = {"enabled": False}
    # schema_version=4及更早版本尚未记录chunk_blend，缺失等价于默认关闭(0)。
    if identity["chunk_blend"] is None:
        identity["chunk_blend"] = 0
    return identity


def prepare_failure_update(
    output_dir: Path,
    manifest: dict[str, Any],
    specs: Sequence[RolloutSpec],
    checkpoint_sha256: str,
) -> tuple[list[RolloutResult], list[RolloutResult], bool]:
    """加载并校验待覆盖更新的历史结果。

    Args:
        output_dir: 已有评测结果目录。
        manifest: 当前命令构造的运行清单。
        specs: 当前配置完整展开后的实验矩阵。
        checkpoint_sha256: 当前checkpoint权重哈希。

    Returns:
        ``(全部旧结果, 旧失败结果, legacy_without_manifest)``。

    Raises:
        FileNotFoundError: 目录或结果日志不存在时抛出。
        ValueError: 实验键、checkpoint或已有manifest身份不一致时抛出。
    """
    journal_path = output_dir / RESULTS_JSONL
    if not output_dir.is_dir() or not journal_path.is_file():
        raise FileNotFoundError(f"失败项更新需要已有{RESULTS_JSONL}: {output_dir}")
    existing_results = load_jsonl(journal_path)
    expected_keys = {spec.key for spec in specs}
    existing_keys = {result.rollout_key for result in existing_results}
    if existing_keys != expected_keys:
        missing = sorted(expected_keys - existing_keys)
        extra = sorted(existing_keys - expected_keys)
        raise ValueError(f"已有结果与当前配置实验键不一致: missing={missing}, extra={extra}")
    checkpoint_hashes = {result.checkpoint_sha256 for result in existing_results}
    if checkpoint_hashes != {checkpoint_sha256}:
        raise ValueError(
            f"已有结果checkpoint哈希与当前输入不一致: existing={sorted(checkpoint_hashes)}, "
            f"current={checkpoint_sha256}"
        )

    manifest_path = output_dir / MANIFEST_JSON
    legacy_without_manifest = not manifest_path.is_file()
    if not legacy_without_manifest:
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if update_manifest_identity(existing_manifest) != update_manifest_identity(manifest):
            raise ValueError("已有manifest与当前checkpoint、配置、execution horizon或环境不一致")
    failures = [result for result in existing_results if not result.success]
    return existing_results, failures, legacy_without_manifest


def repair_legacy_artifact_paths(output_dir: Path, results: Sequence[RolloutResult]) -> None:
    """把历史结果中指向已迁移目录的产物路径修正到当前目录。

    只在当前目录存在同名产物、而历史路径缺失时修正，不会触碰外部目录。
    """
    for result in results:
        spec = RolloutSpec(result.scene_seed, result.task_id, result.prompt_type, result.policy_seed)
        video, trace = artifact_paths(output_dir, spec)
        if result.action_trace_path and not Path(result.action_trace_path).is_file() and trace.is_file():
            result.action_trace_path = str(trace.resolve())
        if result.video_retained and result.video_path and not Path(result.video_path).is_file() and video.is_file():
            result.video_path = str(video.resolve())


def remove_failure_artifacts(output_dir: Path, failures: Sequence[RolloutResult]) -> None:
    """删除待重跑失败项在当前结果目录中的旧视频和动作日志。"""
    for result in failures:
        spec = RolloutSpec(result.scene_seed, result.task_id, result.prompt_type, result.policy_seed)
        for path in artifact_paths(output_dir, spec):
            if path.is_file():
                path.unlink()


def load_jsonl(path: Path) -> list[RolloutResult]:
    """严格读取JSONL日志并拒绝损坏行和重复实验键。"""
    if not path.exists():
        return []
    results: list[RolloutResult] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            result = RolloutResult(**json.loads(line))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path.name}第{line_number}行损坏: {exc}") from exc
        if result.rollout_key in seen:
            raise ValueError(f"{path.name}包含重复实验键: {result.rollout_key}")
        seen.add(result.rollout_key)
        results.append(result)
    return results


def append_jsonl(path: Path, result: RolloutResult) -> None:
    """在每条轨迹结束后立即持久化结果。"""
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def rewrite_jsonl(path: Path, results: list[RolloutResult]) -> None:
    """在清理视频或移除可重试异常后原子重写JSONL。"""
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        for result in results:
            file.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
    temporary.replace(path)


def validate_completed_results(results: list[RolloutResult]) -> None:
    """验证已完成结果的键唯一性和需要保留的视频完整性。"""
    keys = [result.rollout_key for result in results]
    if len(keys) != len(set(keys)):
        raise ValueError("已完成结果包含重复实验键")
    for result in results:
        if result.video_retained:
            video = Path(result.video_path)
            if not video.is_file() or video.stat().st_size == 0:
                raise ValueError(f"已完成结果的视频缺失或为空: {result.rollout_key}")
        if result.action_trace_path:
            trace = Path(result.action_trace_path)
            if result.steps > 0 and (not trace.is_file() or trace.stat().st_size == 0):
                raise ValueError(f"已完成结果的动作诊断日志缺失或为空: {result.rollout_key}")


def bootstrap_success_ci(results: list[RolloutResult], repeats: int = 10_000) -> list[float]:
    """按scene seed整组重采样计算成功率95%置信区间。"""
    if not results:
        return [0.0, 0.0]
    grouped = {
        scene: np.asarray([float(result.success) for result in results if result.scene_seed == scene])
        for scene in sorted({result.scene_seed for result in results})
    }
    scenes = list(grouped)
    rng = np.random.default_rng(20260813)
    samples = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        chosen = rng.choice(scenes, size=len(scenes), replace=True)
        samples[index] = np.concatenate([grouped[int(scene)] for scene in chosen]).mean()
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def grouped_rates(results: list[RolloutResult], attribute: str) -> dict[str, dict[str, float | int]]:
    """按指定结果字段计算成功数、总数和成功率。"""
    output: dict[str, dict[str, float | int]] = {}
    for value in sorted({getattr(result, attribute) for result in results}, key=str):
        group = [result for result in results if getattr(result, attribute) == value]
        successes = sum(result.success for result in group)
        output[str(value)] = {"successes": successes, "rollouts": len(group), "success_rate": successes / len(group)}
    return output


def summarize_action_clipping(results: list[RolloutResult]) -> dict[str, Any]:
    """读取逐步动作日志并计算七个维度的裁剪统计。

    Args:
        results: 已完成且通过完整性检查的rollout结果。

    Returns:
        总体裁剪率、最常裁剪维度及逐维统计。
    """
    clipped_counts = np.zeros(len(ACTION_DIMENSIONS), dtype=np.int64)
    absolute_excess_sums = np.zeros(len(ACTION_DIMENSIONS), dtype=np.float64)
    absolute_excess_max = np.zeros(len(ACTION_DIMENSIONS), dtype=np.float64)
    normalized_excess_sums = np.zeros(len(ACTION_DIMENSIONS), dtype=np.float64)
    normalized_excess_max = np.zeros(len(ACTION_DIMENSIONS), dtype=np.float64)
    trace_steps = 0
    clipped_trace_steps = 0
    trace_rollouts = 0

    for result in results:
        if not result.action_trace_path:
            continue
        trace_path = Path(result.action_trace_path)
        if not trace_path.is_file():
            raise FileNotFoundError(f"动作诊断日志缺失: {result.rollout_key}: {trace_path}")
        trace_rollouts += 1
        rollout_trace_steps = 0
        for line_number, line in enumerate(trace_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                mask = np.asarray(record["clipped_mask"], dtype=np.bool_)
                amount = np.abs(np.asarray(record["clip_amount"], dtype=np.float64))
                lower = np.asarray(record["action_lower"], dtype=np.float64)
                upper = np.asarray(record["action_upper"], dtype=np.float64)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{trace_path.name}第{line_number}行动作记录损坏: {exc}") from exc
            if record.get("rollout_key") != result.rollout_key:
                raise ValueError(f"动作记录实验键不匹配: {trace_path.name}第{line_number}行")
            if any(array.shape != (len(ACTION_DIMENSIONS),) for array in (mask, amount, lower, upper)):
                raise ValueError(f"{trace_path.name}第{line_number}行必须包含七维动作数据")
            ranges = upper - lower
            if not np.isfinite(amount).all() or not np.isfinite(ranges).all() or np.any(ranges <= 0):
                raise ValueError(f"{trace_path.name}第{line_number}行动作数据包含无效数值或范围")
            normalized_excess = amount / ranges
            trace_steps += 1
            rollout_trace_steps += 1
            clipped_trace_steps += int(mask.any())
            clipped_counts += mask.astype(np.int64)
            absolute_excess_sums += np.where(mask, amount, 0.0)
            absolute_excess_max = np.maximum(absolute_excess_max, np.where(mask, amount, 0.0))
            normalized_excess_sums += np.where(mask, normalized_excess, 0.0)
            normalized_excess_max = np.maximum(
                normalized_excess_max,
                np.where(mask, normalized_excess, 0.0),
            )
        if not result.error and rollout_trace_steps != result.steps:
            raise ValueError(
                f"动作记录步数不完整: {result.rollout_key}: "
                f"trace={rollout_trace_steps}, result={result.steps}"
            )

    per_dimension: list[dict[str, Any]] = []
    for index, name in enumerate(ACTION_DIMENSIONS):
        count = int(clipped_counts[index])
        per_dimension.append(
            {
                "index": index,
                "name": name,
                "clipped_steps": count,
                "clipped_step_rate": count / trace_steps if trace_steps else 0.0,
                "mean_abs_clip_amount_when_clipped": (
                    float(absolute_excess_sums[index] / count) if count else 0.0
                ),
                "max_abs_clip_amount": float(absolute_excess_max[index]),
                "mean_normalized_excess_when_clipped": (
                    float(normalized_excess_sums[index] / count) if count else 0.0
                ),
                "max_normalized_excess": float(normalized_excess_max[index]),
            }
        )
    maximum_count = int(clipped_counts.max()) if trace_steps else 0
    most_frequent = [
        item["name"] for item in per_dimension if maximum_count > 0 and item["clipped_steps"] == maximum_count
    ]
    clipped_elements = int(clipped_counts.sum())
    return {
        "trace_rollouts": trace_rollouts,
        "trace_steps": trace_steps,
        "clipped_trace_steps": clipped_trace_steps,
        "clipped_trace_step_rate": clipped_trace_steps / trace_steps if trace_steps else 0.0,
        "clipped_action_elements": clipped_elements,
        "action_elements": trace_steps * len(ACTION_DIMENSIONS),
        "clipped_action_element_rate": (
            clipped_elements / (trace_steps * len(ACTION_DIMENSIONS)) if trace_steps else 0.0
        ),
        "most_frequently_clipped_dimensions": most_frequent,
        "per_dimension": per_dimension,
    }


def write_action_clipping_outputs(output_dir: Path, results: list[RolloutResult]) -> dict[str, Any]:
    """写出动作裁剪JSON汇总和逐维CSV。"""
    clipping = summarize_action_clipping(results)
    write_json(output_dir / ACTION_CLIPPING_JSON, clipping)
    with (output_dir / ACTION_CLIPPING_CSV).open("w", encoding="utf-8-sig", newline="") as csv_file:
        fieldnames = (
            list(clipping["per_dimension"][0].keys())
            if clipping["per_dimension"]
            else ["index", "name", "clipped_steps", "clipped_step_rate"]
        )
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clipping["per_dimension"])
    return clipping


def _trace_motion_rows(result: RolloutResult, fps: int) -> dict[str, Any] | None:
    """从一条含执行后遥测的动作日志计算整轨迹平滑度指标。"""
    if not result.action_trace_path:
        return None
    path = Path(result.action_trace_path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records or "actual_qpos_after" not in records[0]:
        return None
    qpos = np.asarray([record["actual_qpos_after"] for record in records], dtype=np.float64)
    position = np.asarray([record["ee_position_after"] for record in records], dtype=np.float64)
    qvel = np.asarray([record["actual_qvel_after"] for record in records], dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != 6 or position.ndim != 2 or position.shape[1] != 3:
        raise ValueError(f"{path.name}的运动遥测维度错误")
    if qvel.shape != qpos.shape or not (np.isfinite(qpos).all() and np.isfinite(position).all() and np.isfinite(qvel).all()):
        raise ValueError(f"{path.name}的运动遥测包含无效数值")
    dt = 1.0 / fps
    delta_q = np.linalg.norm(np.diff(qpos, axis=0), axis=1)
    delta2_q = np.linalg.norm(qpos[2:] - 2.0 * qpos[1:-1] + qpos[:-2], axis=1)
    speed = np.linalg.norm(np.diff(position, axis=0), axis=1) / dt
    jerk = np.linalg.norm(
        position[3:] - 3.0 * position[2:-1] + 3.0 * position[1:-2] - position[:-3],
        axis=1,
    ) / (dt**3)
    boundary = np.asarray(
        [delta_q[index - 1] for index in range(1, len(records)) if bool(records[index].get("chunk_start", False))],
        dtype=np.float64,
    )
    interior = np.asarray(
        [delta_q[index - 1] for index in range(1, len(records)) if not bool(records[index].get("chunk_start", False))],
        dtype=np.float64,
    )
    gripper_closed = np.asarray([float(record["executed_action"][6]) >= 0.5 for record in records], dtype=np.bool_)
    toggles = int(np.count_nonzero(gripper_closed[1:] != gripper_closed[:-1]))
    return {
        "rollout_key": result.rollout_key,
        "scene_seed": result.scene_seed,
        "task_id": result.task_id,
        "success": result.success,
        "steps": result.steps,
        "mean_delta_q_rad": float(np.mean(delta_q)) if delta_q.size else 0.0,
        "p95_delta_q_rad": percentile(delta_q, 95),
        "mean_delta2_q_rad": float(np.mean(delta2_q)) if delta2_q.size else 0.0,
        "p95_delta2_q_rad": percentile(delta2_q, 95),
        "mean_ee_speed_m_s": float(np.mean(speed)) if speed.size else 0.0,
        "p95_ee_speed_m_s": percentile(speed, 95),
        "mean_ee_jerk_m_s3": float(np.mean(jerk)) if jerk.size else 0.0,
        "p95_ee_jerk_m_s3": percentile(jerk, 95),
        "chunk_boundary_count": int(boundary.size),
        "mean_chunk_boundary_jump_rad": float(np.mean(boundary)) if boundary.size else 0.0,
        "p95_chunk_boundary_jump_rad": percentile(boundary, 95),
        "mean_non_boundary_jump_rad": float(np.mean(interior)) if interior.size else 0.0,
        "chunk_boundary_jump_ratio": (
            float(np.mean(boundary) / np.mean(interior)) if boundary.size and interior.size and float(np.mean(interior)) > 0.0 else 0.0
        ),
        "gripper_toggle_count": toggles,
        "gripper_excess_toggle_count": max(0, toggles - 2),
    }


def _bootstrap_motion_ci(rows: list[dict[str, Any]], field: str, repeats: int = 10_000) -> list[float]:
    """按scene分层重采样计算一个逐轨迹指标中位数的置信区间。"""
    if not rows:
        return [0.0, 0.0]
    grouped = {
        scene: np.asarray([float(row[field]) for row in rows if int(row["scene_seed"]) == scene], dtype=np.float64)
        for scene in sorted({int(row["scene_seed"]) for row in rows})
    }
    scenes = np.asarray(list(grouped), dtype=np.int64)
    rng = np.random.default_rng(20260818)
    samples = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        chosen = rng.choice(scenes, size=len(scenes), replace=True)
        samples[index] = float(np.median(np.concatenate([grouped[int(scene)] for scene in chosen])))
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def write_motion_metrics_outputs(output_dir: Path, results: list[RolloutResult], fps: int) -> dict[str, Any]:
    """写出逐轨迹平滑度CSV、总体JSON和中文Markdown报告。"""
    rows = [row for result in results if (row := _trace_motion_rows(result, fps)) is not None]
    fields = (
        "mean_delta_q_rad", "p95_delta_q_rad", "mean_delta2_q_rad", "p95_delta2_q_rad",
        "mean_ee_speed_m_s", "p95_ee_speed_m_s", "mean_ee_jerk_m_s3", "p95_ee_jerk_m_s3",
        "mean_chunk_boundary_jump_rad", "p95_chunk_boundary_jump_rad", "chunk_boundary_jump_ratio",
        "gripper_toggle_count", "gripper_excess_toggle_count",
    )
    summary = {
        "schema_version": 1,
        "fps": fps,
        "telemetry_rollouts": len(rows),
        "result_rollouts": len(results),
        "metrics": {
            field: {
                "median": float(np.median([float(row[field]) for row in rows])) if rows else 0.0,
                "scene_bootstrap_ci95": _bootstrap_motion_ci(rows, field),
            }
            for field in fields
        },
    }
    with (output_dir / MOTION_METRICS_CSV).open("w", encoding="utf-8-sig", newline="") as csv_file:
        fieldnames = list(rows[0].keys()) if rows else ["rollout_key", "scene_seed", *fields]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_dir / MOTION_METRICS_JSON, summary)
    metrics = summary["metrics"]
    lines = [
        "# SmolVLA 闭环轨迹平滑度报告",
        "",
        f"- 含执行后遥测的轨迹：{len(rows)}/{len(results)}",
        f"- 控制频率：{fps} Hz",
        "",
        "| 指标 | 逐轨迹中位数 | Scene Bootstrap 95% CI |",
        "| --- | ---: | --- |",
    ]
    for field in fields:
        value = metrics[field]
        ci = value["scene_bootstrap_ci95"]
        lines.append(f"| {field} | {value['median']:.8g} | [{ci[0]:.8g}, {ci[1]:.8g}] |")
    (output_dir / MOTION_METRICS_REPORT).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def summarize_results(results: list[RolloutResult]) -> dict[str, Any]:
    """生成总体、分组、稳定性、动作和延迟统计。"""
    successful = [result for result in results if result.success]
    task_rates = grouped_rates(results, "task_id") if results else {}
    prompt_rates = grouped_rates(results, "prompt_type") if results else {}
    seen_values = [prompt_rates[key]["success_rate"] for key in ("canonical", "synonym") if key in prompt_rates]
    seen_rate = float(np.mean(seen_values)) if seen_values else 0.0
    unseen_rate = float(prompt_rates.get("unseen", {}).get("success_rate", 0.0))
    stability: dict[str, int] = {"stable_success_2_of_2": 0, "sampling_sensitive_1_of_2": 0, "stable_failure_0_of_2": 0}
    base_keys = sorted({(result.scene_seed, result.task_id, result.prompt_type) for result in results})
    for base_key in base_keys:
        group = [result for result in results if (result.scene_seed, result.task_id, result.prompt_type) == base_key]
        count = sum(result.success for result in group)
        if len(group) == 2:
            label = ("stable_failure_0_of_2", "sampling_sensitive_1_of_2", "stable_success_2_of_2")[count]
            stability[label] += 1
    total_steps = sum(result.steps for result in results)
    clipped_steps = sum(result.clipped_action_steps for result in results)
    macro = float(np.mean([value["success_rate"] for value in task_rates.values()])) if task_rates else 0.0
    return {
        "rollouts": len(results),
        "successes": len(successful),
        "success_rate": len(successful) / len(results) if results else 0.0,
        "success_rate_ci95_scene_bootstrap": bootstrap_success_ci(results),
        "task_macro_success_rate": macro,
        "by_task": task_rates,
        "by_prompt": prompt_rates,
        "by_scene_seed": grouped_rates(results, "scene_seed") if results else {},
        "by_policy_seed": grouped_rates(results, "policy_seed") if results else {},
        "seen_success_rate": seen_rate,
        "unseen_success_rate": unseen_rate,
        "language_generalization_gap": seen_rate - unseen_rate,
        "stability": stability,
        "failures": {
            mode: sum(result.failure_mode == mode for result in results)
            for mode in sorted({result.failure_mode for result in results})
        },
        "control_exceptions": sum(result.failure_mode == "control_exception" for result in results),
        "successful_steps_median": float(np.median([result.steps for result in successful])) if successful else 0.0,
        "successful_steps_p90": percentile((result.steps for result in successful), 90),
        "trajectories_with_clipping_rate": (
            sum(result.clipped_action_steps > 0 for result in results) / len(results) if results else 0.0
        ),
        "clipped_steps_rate": clipped_steps / total_steps if total_steps else 0.0,
        "latency_median_ms": float(np.median([result.latency_mean_ms for result in results])) if results else 0.0,
        "latency_p95_ms": percentile((result.latency_p95_ms for result in results), 95),
    }


def retain_videos(output_dir: Path, results: list[RolloutResult], keep_all: bool) -> dict[str, Any]:
    """保留全部失败视频及每个任务措辞组合的首条成功视频。"""
    retained_success_keys: set[str] = set()
    if not keep_all:
        for task_id in sorted({result.task_id for result in results}):
            for prompt in sorted({result.prompt_type for result in results}):
                candidates = sorted(
                    (result for result in results if result.success and result.task_id == task_id and result.prompt_type == prompt),
                    key=lambda result: (result.scene_seed, result.policy_seed),
                )
                if candidates:
                    retained_success_keys.add(candidates[0].rollout_key)
    removed: list[str] = []
    retained: list[str] = []
    for result in results:
        keep = keep_all or not result.success or result.rollout_key in retained_success_keys
        video = Path(result.video_path)
        if keep:
            result.video_retained = True
            retained.append(result.rollout_key)
        else:
            # video_path 为空时 Path("") 会指向当前目录，需跳过
            if result.video_path and video.is_file():
                video.unlink()
            result.video_retained = False
            result.video_path = ""
            removed.append(result.rollout_key)
    manifest = {"keep_all_videos": keep_all, "retained_rollout_keys": retained, "removed_rollout_keys": removed}
    write_json(output_dir / "video_retention.json", manifest)
    return manifest


def write_report(
    path: Path,
    summary: dict[str, Any],
    manifest: dict[str, Any],
    update_info: dict[str, Any] | None = None,
) -> None:
    """生成便于人工阅读和归档的Markdown评测报告。"""
    ci = summary["success_rate_ci95_scene_bootstrap"]
    lines = [
        "# SmolVLA 闭环评测报告",
        "",
        f"- Checkpoint：`{manifest['checkpoint_path']}`",
        f"- Rollout：{summary['rollouts']}",
        f"- 总体严格成功率：{summary['success_rate']:.2%}",
        f"- Scene分层Bootstrap 95% CI：[{ci[0]:.2%}, {ci[1]:.2%}]",
        f"- 四任务宏平均成功率：{summary['task_macro_success_rate']:.2%}",
        f"- Seen成功率：{summary['seen_success_rate']:.2%}",
        f"- Unseen成功率：{summary['unseen_success_rate']:.2%}",
        f"- 语言泛化差距：{summary['language_generalization_gap']:.2%}",
        f"- 控制异常：{summary['control_exceptions']}",
        f"- Execution horizon：{manifest.get('execution_horizon', 'unknown')}步",
        f"- 杯子外观：`{manifest.get('appearance_variant', 'original')}`",
        f"- 外观纹理SHA-256：`{manifest.get('appearance_texture_sha256', 'not_applicable')}`",
        "",
        "## 分任务成功率",
        "",
        "| 任务 | 成功数 | 总数 | 成功率 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for task_id, value in summary["by_task"].items():
        lines.append(f"| {task_id} | {value['successes']} | {value['rollouts']} | {value['success_rate']:.2%} |")
    clipping = summary.get("action_clipping", {})
    lines.extend(
        [
            "",
            "## 动作裁剪",
            "",
            f"- 至少一维被裁剪的控制步比例：{clipping.get('clipped_trace_step_rate', 0.0):.2%}",
            f"- 七维动作元素裁剪比例：{clipping.get('clipped_action_element_rate', 0.0):.2%}",
            "- 最常裁剪维度："
            + (", ".join(clipping.get("most_frequently_clipped_dimensions", [])) or "无"),
            "",
            "| 维度 | 裁剪步数 | 裁剪率 | 平均归一化越界量 | 最大归一化越界量 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in clipping.get("per_dimension", []):
        lines.append(
            f"| {item['name']} | {item['clipped_steps']} | {item['clipped_step_rate']:.2%} | "
            f"{item['mean_normalized_excess_when_clipped']:.6f} | {item['max_normalized_excess']:.6f} |"
        )
    if update_info is not None:
        lines.extend(
            [
                "",
                "## 数据更新说明",
                "",
                "- 本结果仅重跑并替换了历史失败rollout；原成功rollout未重新执行。",
                f"- 重跑失败项：{len(update_info['replayed_rollout_keys'])} 条。",
                f"- 原目录缺少manifest：{'是' if update_info['legacy_without_manifest'] else '否'}。",
                f"- 更新审计：`{UPDATE_JSON}`。",
            ]
        )
    lines.extend(["", "## 结果边界", "", "结果仅代表本机MuJoCo仿真闭环能力，不代表真实UR10e成功率。"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_results(
    output_dir: Path,
    results: list[RolloutResult],
    manifest: dict[str, Any] | None = None,
    update_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """写出CSV、汇总JSON和Markdown报告。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "rollouts.csv"
    fieldnames = list(asdict(results[0]).keys()) if results else list(RolloutResult.__dataclass_fields__)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    summary = summarize_results(results)
    summary["action_clipping"] = write_action_clipping_outputs(output_dir, results)
    summary["motion_metrics"] = write_motion_metrics_outputs(
        output_dir,
        results,
        int(manifest.get("fps", 20)) if manifest is not None else 20,
    )
    write_json(output_dir / "summary.json", summary)
    if manifest is not None:
        write_report(output_dir / "report.md", summary, manifest, update_info)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """加载配置并执行、恢复、汇总完整评测矩阵。"""
    args = build_parser().parse_args(argv)
    if args.rerun_failures and args.resume:
        raise ValueError("--rerun-failures不能与--resume同时使用")
    if args.rerun_failures and args.scene_seed is not None:
        raise ValueError("--rerun-failures不能与--scene-seed同时使用")
    if args.rerun_failures and args.max_rollouts is not None:
        raise ValueError("--rerun-failures不能与--max-rollouts同时使用")
    checkpoint = find_pretrained_model(args.checkpoint)
    config = load_yaml_config(args.config)
    evaluation = config.get("evaluation", {})
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation配置段必须是映射")
    output_dir = resolve_path(args.output_dir or evaluation.get("output_dir", "outputs/eval/smolvla"))
    fps = int(evaluation.get("fps", 20))
    max_steps = args.max_steps if args.max_steps is not None else int(evaluation.get("max_steps", 400))
    if fps <= 0 or max_steps <= 0:
        raise ValueError("fps和max_steps必须大于零")
    motion_limiter = resolve_motion_limiter(evaluation, fps)
    execution_horizon, chunk_size = resolve_execution_horizon(
        checkpoint,
        evaluation,
        args.execution_horizon,
    )
    chunk_blend = max(0, int(args.chunk_blend))
    if chunk_blend > chunk_size:
        raise ValueError(f"chunk-blend不能超过模型chunk_size: {chunk_blend} > {chunk_size}")
    full_specs = build_specs(evaluation)
    specs = select_scene_specs(full_specs, args.scene_seed)
    if args.max_rollouts is not None:
        if args.max_rollouts <= 0:
            raise ValueError("max-rollouts必须大于零")
        specs = specs[:args.max_rollouts]
    environment = str(evaluation.get("environment", "cube"))
    appearance_variant = str(evaluation.get("appearance_variant", "original"))
    if environment == "mug":
        texture_path = resolve_mug_texture_path(
            PROJECT_ROOT / "assets" / "mujoco",
            appearance_variant,
        )
        appearance_texture_sha256 = sha256_file(texture_path)
    else:
        if appearance_variant != "original":
            raise ValueError("积木环境不支持杯子appearance_variant")
        appearance_texture_sha256 = "not_applicable"
    checkpoint_hash = sha256_file(checkpoint / "model.safetensors")
    manifest = build_manifest(
        checkpoint,
        checkpoint_hash,
        config,
        specs,
        args.device,
        fps,
        max_steps,
        not args.prune_videos,
        execution_horizon,
        chunk_size,
        appearance_variant,
        appearance_texture_sha256,
        motion_limiter,
        chunk_blend,
    )
    update_info: dict[str, Any] | None = None
    if args.rerun_failures:
        results, previous_failures, legacy_without_manifest = prepare_failure_update(
            output_dir,
            manifest,
            full_specs,
            checkpoint_hash,
        )
        if not previous_failures:
            print("没有失败rollout，无需更新。", flush=True)
            return 0
        repair_legacy_artifact_paths(output_dir, results)
        pending = [
            RolloutSpec(result.scene_seed, result.task_id, result.prompt_type, result.policy_seed)
            for result in previous_failures
        ]
        old_results_by_key = {result.rollout_key: asdict(result) for result in previous_failures}
    else:
        results = prepare_run(output_dir, manifest, args.resume)
        completed = {result.rollout_key for result in results}
        pending = [spec for spec in specs if spec.key not in completed]
        previous_failures = []
        legacy_without_manifest = False
        old_results_by_key = {}
    if pending:
        policy, preprocessor, postprocessor = load_policy_bundle(
            checkpoint,
            args.device,
            execution_horizon,
        )
        if chunk_blend > 0:
            policy = ChunkBlendPolicy(policy, chunk_blend)
        if args.rerun_failures:
            remove_failure_artifacts(output_dir, previous_failures)
        rerun_results: list[RolloutResult] = []
        for spec in pending:
            result = run_single_rollout(
                policy,
                preprocessor,
                postprocessor,
                spec,
                output_dir,
                fps,
                max_steps,
                args.device,
                checkpoint_hash,
                execution_horizon,
                environment,
                appearance_variant,
                motion_limiter,
                chunk_blend,
            )
            if args.rerun_failures:
                rerun_results.append(result)
            else:
                append_jsonl(output_dir / RESULTS_JSONL, result)
                results.append(result)
            print(json.dumps(asdict(result), ensure_ascii=False), flush=True)
        if args.rerun_failures:
            replacements = {result.rollout_key: result for result in rerun_results}
            results = [replacements.get(result.rollout_key, result) for result in results]
            update_info = {
                "schema_version": 1,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "mode": "rerun_failures",
                "legacy_without_manifest": legacy_without_manifest,
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_hash,
                "source_sha256": manifest["source_sha256"],
                "config": config,
                "execution_horizon": execution_horizon,
                "replayed_rollout_keys": [spec.key for spec in pending],
                "old_results": [old_results_by_key[spec.key] for spec in pending],
                "new_results": [asdict(replacements[spec.key]) for spec in pending],
            }
    expected_specs = full_specs if args.rerun_failures else specs
    order = {spec.key: index for index, spec in enumerate(expected_specs)}
    results.sort(key=lambda result: order[result.rollout_key])
    if len(results) != len(expected_specs):
        raise RuntimeError(f"评测结果不完整: expected={len(expected_specs)}, actual={len(results)}")
    validate_completed_results(results)
    retain_videos(output_dir, results, not args.prune_videos)
    rewrite_jsonl(output_dir / RESULTS_JSONL, results)
    if update_info is not None:
        write_json(output_dir / UPDATE_JSON, update_info)
    summary = write_results(output_dir, results, manifest, update_info)
    return 1 if summary["control_exceptions"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
