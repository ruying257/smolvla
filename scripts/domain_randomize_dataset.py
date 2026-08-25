"""重放重渲染：环境级域随机化（纹理 + 光照）数据生成工具。

原理：V3 数据集记录的动作是绝对关节目标，MuJoCo 物理确定且
``reset(scene_seed)`` 确定性复现杯子初始位姿，因此在随机化环境里按采集
语义重放（记录 state[t] 后施加 action[t] 并推进 25 个物理步）可**逐位复现**
原始轨迹（已实测对齐偏差 0.0），只有渲染随纹理/光照变化。由此用同一批
专家动作生成物理一致、外观多样的新 episode，无需重新遥操作。

输出数据集 = 40 条原版（默认光照 + original 纹理，作为干净锚点）+
``variants_per_episode × 40`` 条 DR 变体（纹理拉丁方 + 光照按
``domain_seed`` 确定性采样）。每个 episode 的 domain 元数据写入
``meta/domain_manifest.json`` 供训练与评测复现。

用法::

    python -m scripts.domain_randomize_dataset --config configs/domain_randomize.yaml
    python -m scripts.domain_randomize_dataset --config configs/domain_randomize.yaml --limit 1
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from collector.v3.dataset_io import (
    _accessible_mkdtemp,
    concatenate_video_files_utf8,
    configure_hf_datasets_cache,
    dataset_features,
    find_ffmpeg,
)
from sim.mug_environment import MUG_APPEARANCE_TEXTURES, MugTabletopEnv


EXPECTED_CONFIG_KEYS = {
    "source_root", "output_root", "repo_id", "dataset_version", "fps",
    "variants_per_episode", "lighting_presets", "variants", "include_originals",
    "state_tolerance",
}
MANIFEST_FILENAME = "domain_manifest.json"
LIGHTING_PRESET_KEYS = {"a_scale", "b_azimuth_deg", "c_scale"}


@dataclass(frozen=True)
class DomainRandomizeConfig:
    """已校验的域随机化数据生成配置。"""

    source_root: Path
    output_root: Path
    repo_id: str
    dataset_version: str
    fps: int
    variants_per_episode: int
    lighting_presets: dict[str, dict[str, float]]
    variants: tuple[dict[str, str], ...]
    include_originals: bool
    state_tolerance: float
    snapshot: dict[str, Any]

    @property
    def physics_steps_per_frame(self) -> int:
        """20 Hz 动作帧对应的确定性物理步数（timestep=2ms 时为 25）。"""
        return 25


def load_config(path: Path) -> DomainRandomizeConfig:
    """读取并严格校验域随机化配置。"""
    source_path = path.resolve()
    snapshot = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(snapshot, dict):
        raise ValueError("域随机化配置顶层必须是映射")
    if set(snapshot) != EXPECTED_CONFIG_KEYS:
        raise ValueError(
            "域随机化配置字段必须严格匹配，"
            f"missing={sorted(EXPECTED_CONFIG_KEYS - set(snapshot))}, "
            f"extra={sorted(set(snapshot) - EXPECTED_CONFIG_KEYS)}"
        )
    textures = tuple(snapshot["textures"]) if "textures" in snapshot else None
    lighting_presets = snapshot["lighting_presets"]
    if not isinstance(lighting_presets, dict) or not lighting_presets:
        raise ValueError("lighting_presets必须是非空映射")
    for name, params in lighting_presets.items():
        if not isinstance(params, dict) or set(params) != LIGHTING_PRESET_KEYS:
            raise ValueError(f"光照预设{name!r}必须含a_scale/b_azimuth_deg/c_scale")
        for value in params.values():
            if not np.isfinite(float(value)):
                raise ValueError(f"光照预设{name!r}参数必须为有限数")
    variants = tuple(snapshot["variants"])
    if not variants:
        raise ValueError("variants必须非空")
    for combo in variants:
        if not isinstance(combo, dict) or set(combo) != {"texture", "lighting"}:
            raise ValueError(f"组合变体必须含texture/lighting: {combo!r}")
        if combo["texture"] not in MUG_APPEARANCE_TEXTURES:
            raise ValueError(f"未知纹理变体: {combo['texture']}")
        if combo["lighting"] not in lighting_presets:
            raise ValueError(f"未知光照预设: {combo['lighting']}")
    variants_count = int(snapshot["variants_per_episode"])
    if not 1 <= variants_count <= len(variants):
        raise ValueError(f"variants_per_episode必须位于[1,{len(variants)}]")
    fps = int(snapshot["fps"])
    if fps != 20:
        raise ValueError("域随机化数据生成必须为20 Hz")
    tolerance = float(snapshot["state_tolerance"])
    if not 0.0 < tolerance <= 0.1:
        raise ValueError("state_tolerance必须位于(0,0.1]")
    project_root = source_path.parents[1]
    root = Path(snapshot["output_root"])
    if not root.is_absolute():
        root = project_root / root
    source = Path(snapshot["source_root"])
    if not source.is_absolute():
        source = project_root / source
    return DomainRandomizeConfig(
        source_root=source.resolve(),
        output_root=root.resolve(),
        repo_id=str(snapshot["repo_id"]),
        dataset_version=str(snapshot["dataset_version"]),
        fps=fps,
        variants_per_episode=variants_count,
        lighting_presets={name: {key: float(value) for key, value in params.items()} for name, params in lighting_presets.items()},
        variants=variants,
        include_originals=bool(snapshot["include_originals"]),
        state_tolerance=tolerance,
        snapshot=dict(snapshot),
    )


def build_parser() -> argparse.ArgumentParser:
    """创建重放重渲染命令行解析器。"""
    parser = argparse.ArgumentParser(description="生成环境级域随机化（纹理+光照）数据集")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/domain_randomize.yaml"),
        help="域随机化数据生成YAML配置",
    )
    parser.add_argument("--limit", type=int, help="仅处理前N条源episode，用于冒烟")
    parser.add_argument("--output-root", type=Path, help="覆盖配置中的输出数据集目录（冒烟用）")
    parser.add_argument("--skip-originals", action="store_true", help="不写入40条原版干净锚点")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划与配置")
    return parser


def domain_plan(
    config: DomainRandomizeConfig,
    source_episode_count: int,
    include_originals: bool,
) -> list[dict[str, Any]]:
    """为每条源episode生成原版/组合变体计划（确定性循环分配）。"""
    plan: list[dict[str, Any]] = []
    for episode_index in range(source_episode_count):
        if include_originals:
            plan.append(
                {
                    "source_episode": episode_index,
                    "variant": "original",
                    "texture": "original",
                    "lighting": "default",
                    "lighting_params": dict(config.lighting_presets["default"]),
                }
            )
        for variant_index in range(config.variants_per_episode):
            combo = config.variants[(episode_index + variant_index) % len(config.variants)]
            lighting = combo["lighting"]
            plan.append(
                {
                    "source_episode": episode_index,
                    "variant": f"dr{variant_index}",
                    "texture": combo["texture"],
                    "lighting": lighting,
                    "lighting_params": dict(config.lighting_presets[lighting]),
                }
            )
    return plan


def _state_match(actual: np.ndarray, expected: np.ndarray, tolerance: float) -> float:
    """返回两个七维状态的最大绝对偏差。"""
    return float(np.max(np.abs(np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64))))


def main(argv: list[str] | None = None) -> int:
    """执行重放重渲染并生成域随机化数据集。"""
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.output_root is not None:
        config = DomainRandomizeConfig(
            source_root=config.source_root,
            output_root=args.output_root.resolve(),
            repo_id=config.repo_id,
            dataset_version=config.dataset_version,
            fps=config.fps,
            variants_per_episode=config.variants_per_episode,
            lighting_presets=config.lighting_presets,
            variants=config.variants,
            include_originals=config.include_originals,
            state_tolerance=config.state_tolerance,
            snapshot=config.snapshot,
        )
    if config.output_root.exists():
        raise FileExistsError(f"输出数据集目录已存在，拒绝覆盖: {config.output_root}")
    if find_ffmpeg() is None:
        raise RuntimeError("找不到FFmpeg；请使用smolvla-collector-clean环境")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    configure_hf_datasets_cache(config.output_root.parent / ".hf-lerobot-cache")
    source = LeRobotDataset(repo_id="smolvla_ur10e_mug_v1", root=config.source_root)
    source_count = source.num_episodes
    plan = domain_plan(config, source_count, config.include_originals and not args.skip_originals)
    if args.limit is not None:
        allowed_sources = set(range(args.limit))
        plan = [item for item in plan if item["source_episode"] in allowed_sources]

    print(
        f"源episode={source_count}，计划条目={len(plan)}（原版 + {config.variants_per_episode} 变体/条）",
        flush=True,
    )
    if args.dry_run:
        for item in plan[:12]:
            print(item)
        print("dry-run 结束，未写入任何数据。")
        return 0

    # 预加载全部源 episode 的 state/action（内存 ~9843×7×2 float ≈ 1.1MB，可接受）。
    source_episodes: list[dict[str, Any]] = []
    for episode_index in range(source_count):
        row = source.meta.episodes[episode_index]
        from_index = int(row["dataset_from_index"])
        to_index = int(row["dataset_to_index"])
        scene_seed = int(source[from_index]["scene_seed"])
        task_text = str(source[from_index]["task"])
        states = np.stack(
            [np.asarray(source[i]["observation.state"], dtype=np.float32) for i in range(from_index, to_index)]
        )
        actions = np.stack(
            [np.asarray(source[i]["action"], dtype=np.float32) for i in range(from_index, to_index)]
        )
        source_episodes.append(
            {
                "scene_seed": scene_seed,
                "task_text": task_text,
                "states": states,
                "actions": actions,
            }
        )

    output = LeRobotDataset.create(
        repo_id=config.repo_id,
        root=config.output_root,
        robot_type="ur10e_mujoco_mug",
        fps=config.fps,
        features=dataset_features(),
        use_videos=True,
        image_writer_threads=4,
        image_writer_processes=0,
        video_backend="pyav",
        vcodec="h264",
    )

    physics_steps = config.physics_steps_per_frame
    manifest_episodes: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    episode_index = 0

    from lerobot.datasets import lerobot_dataset as lerobot_dataset_module

    try:
        for item in plan:
            source_ep = source_episodes[item["source_episode"]]
            states = source_ep["states"]
            actions = source_ep["actions"]
            task_text = source_ep["task_text"]
            scene_seed = source_ep["scene_seed"]
            task_id = "mug_on_blue" if "blue" in task_text else "mug_on_yellow"
            if task_id not in {"mug_on_blue", "mug_on_yellow"}:
                raise ValueError(f"未知任务文本: {task_text!r}")

            env = MugTabletopEnv(appearance_variant=item["texture"])
            env.set_lighting(**item["lighting_params"])
            lighting_params = dict(item["lighting_params"])
            snapshot = env.reset(scene_seed=scene_seed)

            # 校验初始状态与记录一致。
            initial_dev = _state_match(env.get_state(), states[0], config.state_tolerance)
            if initial_dev > config.state_tolerance:
                rejected.append({**item, "reason": f"initial_state_dev={initial_dev:.4f}"})
                env.close()
                continue

            frames_agent: list[np.ndarray] = []
            frames_wrist: list[np.ndarray] = []
            max_dev = initial_dev
            first_success_frame: int | None = None
            for t in range(len(states)):
                images = env.capture_training_images()
                frames_agent.append(np.asarray(images["agent"]))
                frames_wrist.append(np.asarray(images["wrist"]))
                env.apply_joint_action(actions[t], physics_steps=physics_steps)
                if t + 1 < len(states):
                    dev = _state_match(env.get_state(), states[t + 1], config.state_tolerance)
                    max_dev = max(max_dev, dev)
                    if dev > config.state_tolerance:
                        rejected.append({**item, "reason": f"state_dev@{t+1}={dev:.4f}"})
                        break
                evaluation = env.evaluate_task(
                    task_id,
                    elapsed_seconds=(t + 1) / config.fps,
                    timeout_seconds=None,
                )
                if evaluation.success and first_success_frame is None:
                    first_success_frame = t + 1
            else:
                final_eval = env.evaluate_task(
                    task_id,
                    elapsed_seconds=len(states) / config.fps,
                    timeout_seconds=None,
                )
                if final_eval.success:
                    for frame_index in range(len(states)):
                        output.add_frame(
                            {
                                "observation.images.agent": frames_agent[frame_index],
                                "observation.images.wrist": frames_wrist[frame_index],
                                "observation.state": states[frame_index],
                                "action": actions[frame_index],
                                "scene_seed": np.asarray([scene_seed], dtype=np.int64),
                                "mug_initial_pose": snapshot.mug_initial_pose.astype(np.float32),
                                "task": task_text,
                            }
                        )
                    original_mkdtemp = tempfile.mkdtemp
                    original_concatenate = lerobot_dataset_module.concatenate_video_files
                    tempfile.mkdtemp = _accessible_mkdtemp
                    lerobot_dataset_module.concatenate_video_files = concatenate_video_files_utf8
                    try:
                        output.save_episode(parallel_encoding=False)
                    finally:
                        tempfile.mkdtemp = original_mkdtemp
                        lerobot_dataset_module.concatenate_video_files = original_concatenate
                        for directory in config.output_root.glob(".lerobot-encode-*"):
                            shutil.rmtree(directory, ignore_errors=True)
                    manifest_episodes.append(
                        {
                            "episode_index": episode_index,
                            "source_episode": item["source_episode"],
                            "variant": item["variant"],
                            "texture": item["texture"],
                            "lighting": item["lighting"],
                            "lighting_params": lighting_params,
                            "scene_seed": scene_seed,
                            "task_id": task_id,
                            "task": task_text,
                            "frame_count": len(states),
                            "max_state_dev": round(max_dev, 6),
                            "first_success_frame": first_success_frame,
                        }
                    )
                    episode_index += 1
                    print(
                        f"episode {episode_index:03d}: src={item['source_episode']:02d} "
                        f"{item['variant']:<9} texture={item['texture']:<12} "
                        f"frames={len(states)} dev={max_dev:.5f}",
                        flush=True,
                    )
                else:
                    rejected.append({**item, "reason": f"final_success={final_eval.success}"})
            env.close()
    finally:
        output.finalize()

    manifest = {
        "dataset_version": config.dataset_version,
        "repo_id": config.repo_id,
        "config": config.snapshot,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "episode_count": episode_index,
        "rejected_count": len(rejected),
        "rejected": rejected,
        "episodes": manifest_episodes,
    }
    manifest_path = config.output_root / "meta" / MANIFEST_FILENAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完成：写入 {episode_index} 条，拒绝 {len(rejected)} 条。")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
