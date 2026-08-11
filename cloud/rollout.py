"""在项目 MuJoCo 环境中执行 SmolVLA 的 EGL 闭环 rollout。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np

from cloud.common import (
    PROJECT_ROOT,
    convert_policy_action,
    find_pretrained_model,
    load_yaml_config,
    percentile,
    resolve_path,
    write_json,
)
from collector.task_spec import TASKS
from sim.environment import CleanTabletopEnv


UNSEEN_TEMPLATE = "Move the {cube_color} cube to the {pad_color} pad."


@dataclass(frozen=True)
class RolloutResult:
    """一条闭环评测记录。"""

    scene_seed: int
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
    video_path: str
    error: str


def build_parser() -> argparse.ArgumentParser:
    """创建 rollout 命令行解析器。"""
    parser = argparse.ArgumentParser(description="执行 SmolVLA UR10e headless rollout")
    parser.add_argument("--checkpoint", type=Path, required=True, help="模型、checkpoint 或训练输出目录")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "cloud_eval.yaml",
        help="评测 YAML 配置",
    )
    parser.add_argument("--output-dir", type=Path, help="覆盖配置中的评测输出目录")
    parser.add_argument("--device", default="cuda", help="策略推理设备")
    parser.add_argument("--max-rollouts", type=int, help="只执行前 N 条 rollout，用于 smoke test")
    parser.add_argument("--max-steps", type=int, help="覆盖每条 rollout 的最大控制步数")
    return parser


def build_prompt(task_id: str, prompt_type: str) -> str:
    """构造 canonical 或未见措辞任务文本。"""
    task = TASKS[task_id]
    if prompt_type == "canonical":
        return task.prompt("canonical")
    if prompt_type == "unseen":
        return UNSEEN_TEMPLATE.format(cube_color=task.cube_color, pad_color=task.pad_color)
    raise ValueError(f"未知 prompt_type={prompt_type!r}")


def make_policy_observation(images: dict[str, np.ndarray], state: np.ndarray, task: str) -> dict[str, Any]:
    """把 MuJoCo 观测转换为 LeRobot 策略的 batch 输入。

    Args:
        images: 包含 ``agent`` 和 ``wrist`` 的 HWC RGB uint8 图像。
        state: 七维当前状态。
        task: 英文任务文本。

    Returns:
        带 batch 维的 Tensor 字典和任务列表。
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


def load_policy_bundle(checkpoint: Path, device: str) -> tuple[Any, Callable[[Any], Any], Callable[[Any], Any]]:
    """加载 checkpoint 策略及其预处理和后处理流水线。"""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    config = PreTrainedConfig.from_pretrained(
        str(checkpoint),
        cli_overrides=[f"--device={device}", "--push_to_hub=false"],
    )
    policy_class = get_policy_class(config.type)
    policy = policy_class.from_pretrained(str(checkpoint), config=config)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor


def run_single_rollout(
    policy: Any,
    preprocessor: Callable[[Any], Any],
    postprocessor: Callable[[Any], Any],
    scene_seed: int,
    task_id: str,
    prompt_type: str,
    output_dir: Path,
    fps: int,
    max_steps: int,
    device: str,
) -> RolloutResult:
    """执行一条确定性场景的闭环 rollout 并保存视频。"""
    import torch

    task_text = build_prompt(task_id, prompt_type)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "videos" / f"seed_{scene_seed}_{task_id}_{prompt_type}.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    latencies: list[float] = []
    clipped_steps = 0
    failure_mode = "timeout"
    error = ""
    completed_steps = 0
    rollout_start = time.perf_counter()

    try:
        with CleanTabletopEnv() as env, imageio.get_writer(
            video_path,
            fps=fps,
            codec="libx264",
            quality=7,
            macro_block_size=None,
        ) as writer:
            env.reset(scene_seed)
            policy.reset()
            physics_steps = max(1, round((1.0 / fps) / float(env.model.opt.timestep)))
            for step_index in range(max_steps):
                images = env.capture_training_images()
                writer.append_data(np.concatenate([images["agent"], images["wrist"]], axis=1))
                observation = make_policy_observation(images, env.get_state(), task_text)
                started = time.perf_counter()
                processed = preprocessor(observation)
                autocast_context = (
                    torch.autocast(device_type="cuda")
                    if device.startswith("cuda") and bool(getattr(policy.config, "use_amp", False))
                    else nullcontext()
                )
                with torch.inference_mode(), autocast_context:
                    raw_action = policy.select_action(processed)
                raw_action = postprocessor(raw_action)
                latencies.append((time.perf_counter() - started) * 1000.0)
                safe_action = convert_policy_action(raw_action, env.model.actuator_ctrlrange[:6])
                clipped_steps += int(safe_action.clipped)
                env.apply_joint_action(safe_action.command, physics_steps=physics_steps)
                completed_steps = step_index + 1
                elapsed_sim = completed_steps / fps
                evaluation = env.evaluate_task(task_id, elapsed_sim, max_steps / fps)
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
        scene_seed=scene_seed,
        task_id=task_id,
        task=task_text,
        prompt_type=prompt_type,
        success=success,
        failure_mode=failure_mode,
        steps=completed_steps,
        elapsed_seconds=elapsed,
        latency_mean_ms=float(np.mean(latencies)) if latencies else 0.0,
        latency_p95_ms=percentile(latencies, 95),
        clipped_action_steps=clipped_steps,
        video_path=str(video_path.resolve()),
        error=error,
    )


def write_results(output_dir: Path, results: list[RolloutResult]) -> None:
    """写出逐条 CSV 和汇总 JSON。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "rollouts.csv"
    fieldnames = list(asdict(results[0]).keys()) if results else list(RolloutResult.__dataclass_fields__)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    successful = sum(result.success for result in results)
    summary = {
        "rollouts": len(results),
        "successes": successful,
        "success_rate": successful / len(results) if results else 0.0,
        "failures": {
            mode: sum(result.failure_mode == mode for result in results)
            for mode in sorted({result.failure_mode for result in results})
        },
        "latency_mean_ms": float(np.mean([result.latency_mean_ms for result in results])) if results else 0.0,
        "latency_p95_ms": percentile((result.latency_p95_ms for result in results), 95),
    }
    write_json(output_dir / "summary.json", summary)


def main(argv: Sequence[str] | None = None) -> int:
    """加载配置并执行所需 seed、任务与措辞组合。"""
    args = build_parser().parse_args(argv)
    checkpoint = find_pretrained_model(args.checkpoint)
    config = load_yaml_config(args.config)
    evaluation = config.get("evaluation", {})
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation 配置段必须是映射")
    output_dir = resolve_path(args.output_dir or evaluation.get("output_dir", "outputs/eval/smolvla"))
    fps = int(evaluation.get("fps", 20))
    max_steps = args.max_steps if args.max_steps is not None else int(evaluation.get("max_steps", 600))
    seeds = [int(seed) for seed in evaluation.get("scene_seeds", [10000])]
    task_ids = list(evaluation.get("task_ids", TASKS.keys()))
    prompt_types = list(evaluation.get("prompt_types", ["canonical"]))
    if fps <= 0 or max_steps <= 0:
        raise ValueError("fps 和 max_steps 必须大于零")
    if any(task_id not in TASKS for task_id in task_ids):
        raise ValueError(f"评测包含未知任务: {task_ids}")

    combinations = [(seed, task_id, prompt) for seed in seeds for task_id in task_ids for prompt in prompt_types]
    if args.max_rollouts is not None:
        if args.max_rollouts <= 0:
            raise ValueError("max-rollouts 必须大于零")
        combinations = combinations[: args.max_rollouts]
    policy, preprocessor, postprocessor = load_policy_bundle(checkpoint, args.device)
    results = [
        run_single_rollout(
            policy,
            preprocessor,
            postprocessor,
            seed,
            task_id,
            prompt,
            output_dir,
            fps,
            max_steps,
            args.device,
        )
        for seed, task_id, prompt in combinations
    ]
    write_results(output_dir, results)
    for result in results:
        print(json.dumps(asdict(result), ensure_ascii=False))
    return 1 if any(result.error for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
