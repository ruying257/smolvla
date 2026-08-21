"""SmolVLA 视觉鲁棒性评测专用 rollout 变体（``evaluate/rollout.py`` 的薄副本）。

本文件是 ``evaluate/rollout.py`` 的专用薄副本，服务于视觉鲁棒性扰动
评测工具（``evaluate.diagnose_mug_visual_robustness``），避免向核心评测
入口增加图像扰动等诊断参数。原 ``evaluate/rollout.py`` 保持零改动。

结构与同步约定：

- 通过 ``from evaluate.rollout import *`` re-export 原模块的全部公开符号；
  原模块未定义 ``__all__``，因此 ``np``、``time``、``datetime``、
  ``CleanTabletopEnv``、``MugTabletopEnv``、``MugStageTracker`` 等
  ``run_single_rollout`` 函数体引用到的名字均被带入本模块命名空间。
- 除 ``run_single_rollout`` 外，本文件不复制任何函数：其余符号与原模块
  自动同步，原模块的修复无需手动搬移。
- 被 re-export 的 ``main`` 的 ``__globals__`` 仍指向原模块，因此本文件
  的 CLI 入口（若被运行）执行的是原模块逻辑。本文件定位为被诊断脚本
  import 复用，诊断脚本直接调用本文件的 ``run_single_rollout``，不经
  ``main``。
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from evaluate.rollout import *  # noqa: F401,F403  (re-export 原模块全部公开符号)
from evaluate.rollout import (  # 显式声明 run_single_rollout 函数体依赖的公开符号
    ACTION_TRACE_DIR,
    GripperFilterResult,
    GripperHysteresisFilter,
    MugStageStepResult,
    MugStageTracker,
    MUG_BODY_NAME,
    MUG_TASK_IDS,
    RolloutResult,
    RolloutSpec,
    action_to_vector,
    build_prompt,
    convert_policy_action,
    make_policy_observation,
    percentile,
    resolve_gripper_filter,
    resolve_stage_detection,
    rollout_artifact_stem,
    set_policy_seed,
    write_action_trace_record,
)


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
    gripper_filter_settings: dict[str, Any] | None = None,
    stage_detection_settings: dict[str, Any] | None = None,
    image_transform: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]] | None = None,
    artifact_stem_override: str | None = None,
) -> RolloutResult:
    """执行一条固定场景和模型随机种子的闭环rollout。

    本函数是 ``evaluate.rollout.run_single_rollout`` 的副本，差异有二：
    新增 ``image_transform`` 参数（在 MuJoCo 相机图像捕获之后、构造策略
    观测之前应用图像变换，视频记录变换后的图像，即策略实际所见）与
    ``artifact_stem_override`` 参数（覆盖视频/动作日志文件名前缀，供
    同一 scene/task/policy 下多个扰动条件独立落盘，避免互相覆盖）。
    两个参数为``None``时行为与原模块完全一致。

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
        chunk_blend: 动作chunk边界插值帧数。
        gripper_filter_settings: 已解析的夹爪迟滞过滤配置。
        stage_detection_settings: 已解析的杯子阶段检测配置。
        image_transform: 可选图像变换；接收 ``{"agent": ..., "wrist": ...}``
            图像字典并返回同键、同shape、同dtype（``uint8``）的字典，
            在构造策略观测前应用。为``None``时不应用任何变换。
        artifact_stem_override: 可选视频/动作日志文件名前缀覆盖；为``None``
            时使用默认的 ``scene_<seed>_policy_<seed>_<task>_<prompt>``。

    Returns:
        单条闭环评测结果。
    """
    import torch

    task_text = build_prompt(spec.task_id, spec.prompt_type)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_stem = artifact_stem_override or rollout_artifact_stem(spec)
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
            stage_settings = (
                stage_detection_settings
                if stage_detection_settings is not None
                else resolve_stage_detection({})
            )
            stage_detection_applicable = spec.task_id in MUG_TASK_IDS
            stage_detection_enabled = bool(
                stage_detection_applicable and stage_settings.get("enabled", False)
            )
            stage_tracker: MugStageTracker | None = None
            if stage_detection_enabled:
                initial_layout = env.task_layout()
                initial_bottom_z = float(
                    np.asarray(
                        initial_layout[MUG_BODY_NAME]["bottom_site_position"],
                        dtype=np.float64,
                    )[2]
                )
                stage_tracker = MugStageTracker(
                    initial_bottom_z=initial_bottom_z,
                    gripper_closed_threshold=float(
                        stage_settings["gripper_closed_threshold"]
                    ),
                    lift_delta_m=float(stage_settings["lift_delta_m"]),
                )
            physics_steps = max(1, round((1.0 / fps) / float(env.model.opt.timestep)))
            action_lower = np.concatenate([env.model.actuator_ctrlrange[:6, 0], [0.0]])
            action_upper = np.concatenate([env.model.actuator_ctrlrange[:6, 1], [1.0]])
            filter_settings = (
                gripper_filter_settings
                if gripper_filter_settings is not None
                else resolve_gripper_filter({})
            )
            gripper_filter_enabled = bool(filter_settings.get("enabled", False))
            gripper_filter = (
                GripperHysteresisFilter(
                    close_threshold=float(filter_settings["close_threshold"]),
                    open_threshold=float(filter_settings["open_threshold"]),
                    close_confirmation_frames=int(filter_settings["close_confirmation_frames"]),
                    open_confirmation_frames=int(filter_settings["open_confirmation_frames"]),
                )
                if gripper_filter_enabled
                else None
            )
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
                if image_transform is not None:
                    images = image_transform(images)
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
                filtered_action = safe_action.command.copy()
                raw_gripper_command = float(filtered_action[6])
                if gripper_filter is None:
                    gripper_filter_result = GripperFilterResult(
                        command=raw_gripper_command,
                        state="passthrough",
                        confirmation_count=0,
                        transitioned=False,
                        transition="none",
                    )
                else:
                    gripper_filter_result = gripper_filter.apply(raw_gripper_command)
                    filtered_action[6] = gripper_filter_result.command
                if limiter is None:
                    executed_action = filtered_action
                    motion_limited_mask = np.zeros(7, dtype=np.bool_)
                    motion_limit_amount = np.zeros(7, dtype=np.float32)
                    reference_velocity = np.zeros(6, dtype=np.float64)
                else:
                    executed_action, motion_limited_mask, motion_limit_amount = limiter.limit(filtered_action)
                    reference_velocity = limiter.reference_velocity
                env.apply_joint_action(executed_action, physics_steps=physics_steps)
                actual_state_after = env.get_state()
                completed_steps = step_index + 1
                evaluation = env.evaluate_task(spec.task_id, completed_steps / fps, max_steps / fps)
                stage_result: MugStageStepResult | None = None
                if stage_tracker is not None:
                    stage_result = stage_tracker.update(
                        step=completed_steps,
                        elapsed_seconds=completed_steps / fps,
                        gripper_state=float(actual_state_after[6]),
                        metrics=evaluation.metrics,
                        success=evaluation.success,
                    )
                stage_trace: dict[str, Any] = {
                    "stage_detection_applicable": stage_detection_applicable,
                    "stage_detection_enabled": stage_detection_enabled,
                }
                if stage_tracker is not None and stage_result is not None:
                    stage_trace.update(
                        {
                            "initial_bottom_z": stage_tracker.initial_bottom_z,
                            "bottom_z": float(evaluation.metrics["bottom_z"]),
                            "lift_threshold_z": stage_tracker.lift_threshold_z,
                            "stage_current_conditions": stage_result.current_conditions,
                            "stage_direct_reached": stage_result.direct_reached,
                            "stage_first_reached_step": stage_result.first_reached_step,
                            "stage_first_reached_seconds": stage_result.first_reached_seconds,
                            "sequential_stage": stage_result.sequential_stage,
                            "stage_transition": stage_result.transition,
                            "highest_direct_stage": stage_result.highest_direct_stage,
                            "stage_order_anomaly": stage_result.order_anomaly,
                            "task_metrics": dict(evaluation.metrics),
                        }
                    )
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
                        "gripper_filter_enabled": gripper_filter_enabled,
                        "gripper_raw_command": raw_gripper_command,
                        "gripper_filtered_command": float(gripper_filter_result.command),
                        "gripper_filter_state": gripper_filter_result.state,
                        "gripper_confirmation_count": gripper_filter_result.confirmation_count,
                        "gripper_state_transitioned": gripper_filter_result.transitioned,
                        "gripper_transition": gripper_filter_result.transition,
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
                        **stage_trace,
                    },
                )
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
