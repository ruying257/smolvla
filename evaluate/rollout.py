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

from collector.task_spec import TASKS
from evaluate.common import (
    PROJECT_ROOT,
    convert_policy_action,
    find_pretrained_model,
    load_yaml_config,
    percentile,
    resolve_path,
    write_json,
)
from sim.environment import CleanTabletopEnv


UNSEEN_TEMPLATE = "Move the {cube_color} cube to the {pad_color} pad."
RESULTS_JSONL = "rollouts.jsonl"
MANIFEST_JSON = "run_manifest.json"


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
    parser.add_argument("--max-rollouts", type=int, help="只执行前 N 条 rollout，用于 smoke test")
    parser.add_argument("--max-steps", type=int, help="覆盖每条 rollout 的最大控制步数")
    parser.add_argument("--resume", action="store_true", help="校验manifest后续跑未完成或异常轨迹")
    parser.add_argument("--keep-all-videos", action="store_true", help="保留全部成功与失败视频")
    return parser


def build_prompt(task_id: str, prompt_type: str) -> str:
    """构造训练已见或训练未见的任务文本。

    Args:
        task_id: 四类搬运任务之一。
        prompt_type: ``canonical``、``synonym`` 或 ``unseen``。

    Returns:
        英文任务指令。
    """
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


def load_policy_bundle(checkpoint: Path, device: str) -> tuple[Any, Callable[[Any], Any], Callable[[Any], Any]]:
    """加载checkpoint策略及其预处理和后处理流水线。"""
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
    spec: RolloutSpec,
    output_dir: Path,
    fps: int,
    max_steps: int,
    device: str,
    checkpoint_sha256: str,
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

    Returns:
        单条闭环评测结果。
    """
    import torch

    task_text = build_prompt(spec.task_id, spec.prompt_type)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "videos" / (
        f"scene_{spec.scene_seed}_policy_{spec.policy_seed}_{spec.task_id}_{spec.prompt_type}.mp4"
    )
    video_path.parent.mkdir(parents=True, exist_ok=True)
    latencies: list[float] = []
    clipped_steps = 0
    failure_mode = "timeout"
    error = ""
    completed_steps = 0
    rollout_start = time.perf_counter()

    try:
        set_policy_seed(spec.policy_seed)
        with CleanTabletopEnv() as env, imageio.get_writer(
            video_path,
            fps=fps,
            codec="libx264",
            quality=7,
            macro_block_size=None,
        ) as writer:
            env.reset(spec.scene_seed)
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
    task_ids = [str(value) for value in evaluation.get("task_ids", TASKS)]
    prompt_types = [str(value) for value in evaluation.get("prompt_types", ["canonical"])]
    policy_seeds = [int(value) for value in evaluation.get("policy_seeds", [20260])]
    if any(task_id not in TASKS for task_id in task_ids):
        raise ValueError(f"评测包含未知任务: {task_ids}")
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
    paths = sorted((PROJECT_ROOT / "evaluate").glob("*.py")) + [
        PROJECT_ROOT / "sim" / "environment.py",
        PROJECT_ROOT / "collector" / "task_spec.py",
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
) -> dict[str, Any]:
    """构造用于恢复校验的完整运行清单。"""
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "source_sha256": source_sha256(),
        "git": git_identity(),
        "environment": package_versions(device),
        "amp_enabled": bool(json.loads((checkpoint / "config.json").read_text(encoding="utf-8")).get("use_amp")),
        "fps": fps,
        "max_steps": max_steps,
        "keep_all_videos": keep_all_videos,
        "config": config,
        "rollout_count": len(specs),
        "rollout_keys": [spec.key for spec in specs],
    }


def manifest_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    """提取恢复时必须完全一致的manifest字段。"""
    keys = (
        "schema_version",
        "checkpoint_path",
        "checkpoint_sha256",
        "source_sha256",
        "environment",
        "amp_enabled",
        "fps",
        "max_steps",
        "keep_all_videos",
        "config",
        "rollout_count",
        "rollout_keys",
    )
    return {key: manifest.get(key) for key in keys}


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
            if video.exists():
                video.unlink()
            result.video_retained = False
            result.video_path = ""
            removed.append(result.rollout_key)
    manifest = {"keep_all_videos": keep_all, "retained_rollout_keys": retained, "removed_rollout_keys": removed}
    write_json(output_dir / "video_retention.json", manifest)
    return manifest


def write_report(path: Path, summary: dict[str, Any], manifest: dict[str, Any]) -> None:
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
        "",
        "## 分任务成功率",
        "",
        "| 任务 | 成功数 | 总数 | 成功率 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for task_id, value in summary["by_task"].items():
        lines.append(f"| {task_id} | {value['successes']} | {value['rollouts']} | {value['success_rate']:.2%} |")
    lines.extend(["", "## 结果边界", "", "结果仅代表本机MuJoCo仿真闭环能力，不代表真实UR10e成功率。"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_results(output_dir: Path, results: list[RolloutResult], manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """写出CSV、汇总JSON和Markdown报告。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "rollouts.csv"
    fieldnames = list(asdict(results[0]).keys()) if results else list(RolloutResult.__dataclass_fields__)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    summary = summarize_results(results)
    write_json(output_dir / "summary.json", summary)
    if manifest is not None:
        write_report(output_dir / "report.md", summary, manifest)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """加载配置并执行、恢复、汇总完整评测矩阵。"""
    args = build_parser().parse_args(argv)
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
    specs = build_specs(evaluation, args.max_rollouts)
    checkpoint_hash = sha256_file(checkpoint / "model.safetensors")
    manifest = build_manifest(
        checkpoint,
        checkpoint_hash,
        config,
        specs,
        args.device,
        fps,
        max_steps,
        args.keep_all_videos,
    )
    results = prepare_run(output_dir, manifest, args.resume)
    completed = {result.rollout_key for result in results}
    pending = [spec for spec in specs if spec.key not in completed]
    if pending:
        policy, preprocessor, postprocessor = load_policy_bundle(checkpoint, args.device)
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
            )
            append_jsonl(output_dir / RESULTS_JSONL, result)
            results.append(result)
            print(json.dumps(asdict(result), ensure_ascii=False), flush=True)
    results.sort(key=lambda result: [spec.key for spec in specs].index(result.rollout_key))
    if len(results) != len(specs):
        raise RuntimeError(f"评测结果不完整: expected={len(specs)}, actual={len(results)}")
    validate_completed_results(results)
    retain_videos(output_dir, results, args.keep_all_videos)
    rewrite_jsonl(output_dir / RESULTS_JSONL, results)
    summary = write_results(output_dir, results, manifest)
    return 1 if summary["control_exceptions"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
