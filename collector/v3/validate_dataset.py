"""杯子V3分片、确定性动作回放、人工复核与最终数据集验收。"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import uuid
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import av
import numpy as np

from collector.v3.collection_plan import (
    PILOT_VALIDATION_FILENAME,
    REDO_KEYS_FILENAME,
    REVIEW_FILENAME,
    QueueItem,
    atomic_write_json,
    build_plan,
    load_config,
    load_progress,
    plan_for_mode,
    utc_now,
)
from collector.v3.dataset_io import (
    CAMERA_FEATURES,
    MugEpisodeWriter,
    configure_hf_datasets_cache,
    decode_video,
    read_shard_table,
    task_texts,
    validate_episode_shard,
    vector_column,
)
from sim.mug_environment import MugSceneSnapshot, MugTabletopEnv


VALIDATION_FILENAME = "dataset_validation.json"
EPISODE_CSV_FILENAME = "episode_manifest.csv"


def build_parser() -> argparse.ArgumentParser:
    """创建杯子V3验收命令行解析器。

    Returns:
        支持pilot验收和全量最终物化的解析器。
    """
    parser = argparse.ArgumentParser(description="验收杯子V3配对数据集")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true", help="只验收两个scene的4条pilot")
    parser.add_argument("--finalize", action="store_true", help="40条PASS后物化最终LeRobot数据集")
    return parser


def _read_review_status(path: Path) -> dict[int, str]:
    """读取用户完成的逐scene人工复核状态。

    Args:
        path: ``review_status.csv``路径。

    Returns:
        scene seed到状态文本的映射；文件缺失时返回空映射。
    """
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {int(row["scene_seed"]): row["status"] for row in csv.DictReader(handle)}


def _montage_is_valid(path: Path) -> bool:
    """检查蓝黄并排蒙太奇可解码且分辨率正确。

    Args:
        path: scene级复核MP4路径。

    Returns:
        至少包含一帧512×512视频时返回 ``True``。
    """
    try:
        count = 0
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                if frame.width != 512 or frame.height != 512:
                    return False
                count += 1
        return count > 0
    except Exception:
        return False


def _has_close_then_release(actions: np.ndarray) -> bool:
    """判断动作轨迹是否真实包含闭合后再次释放。

    Args:
        actions: ``(frames,7)``绝对动作数组。

    Returns:
        出现至少一帧闭合指令且其后出现释放指令时返回 ``True``。
    """
    closed = np.flatnonzero(actions[:, 6] >= 0.5)
    return bool(len(closed) and np.any(actions[closed[0] + 1 :, 6] < 0.5))


def replay_actions(
    env: MugTabletopEnv,
    item: QueueItem,
    actions: np.ndarray,
    fps: int,
) -> tuple[bool, str]:
    """从锁定seed按20 Hz确定性回放七维绝对动作。

    采集器会在每个20 Hz采样点先检查严格成功，再决定是否写入当前帧。
    因此，检测到成功的采样点本身不会进入episode。回放完已记录动作后，
    需要将末个绝对目标额外保持一个控制周期，才能复现采集器执行成功
    检查时的真实仿真时刻。

    Args:
        env: 可重复使用的真实杯子MuJoCo环境。
        item: 当前队列项，提供seed和任务标识。
        actions: ``(frames,7)``绝对关节目标动作。
        fps: 固定动作频率，V3必须为20。

    Returns:
        ``(是否严格成功, 最终失败分类)``。

    Raises:
        ValueError: 动作shape、频率或仿真步长无法表示20 Hz时抛出。
    """
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError(f"回放动作必须是有限(N,7)数组，实际为{actions.shape}")
    if fps != 20:
        raise ValueError(f"杯子V3动作回放必须为20 Hz，实际为{fps}")
    env.reset(item.scene_seed)
    step_ratio = (1.0 / fps) / float(env.model.opt.timestep)
    rounded_steps = int(round(step_ratio))
    if rounded_steps < 1 or not np.isclose(step_ratio, rounded_steps, atol=1e-9):
        raise ValueError(f"MuJoCo timestep无法精确表示20 Hz: ratio={step_ratio}")
    evaluation = env.evaluate_task(item.task_id)
    for frame_index, action in enumerate(actions):
        env.apply_joint_action(action, physics_steps=rounded_steps)
        evaluation = env.evaluate_task(
            item.task_id,
            elapsed_seconds=(frame_index + 1) / fps,
            timeout_seconds=len(actions) / fps,
        )
    if len(actions) and not evaluation.success:
        env.apply_joint_action(actions[-1], physics_steps=rounded_steps)
        terminal_elapsed = (len(actions) + 1) / fps
        evaluation = env.evaluate_task(
            item.task_id,
            elapsed_seconds=terminal_elapsed,
            timeout_seconds=terminal_elapsed,
        )
    return evaluation.success, evaluation.failure_mode


def _append_error(
    errors: list[str],
    redo_keys: set[str],
    message: str,
    keys: Sequence[str],
) -> None:
    """同时登记人类可读错误和可直接重采的唯一键。

    Args:
        errors: 验收错误列表。
        redo_keys: 去重后的redo key集合。
        message: 当前错误说明。
        keys: 受该错误影响的队列键。
    """
    errors.append(message)
    redo_keys.update(keys)


def _validate_records(config: Any, items: list[QueueItem]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """执行分片、数值、夹爪、配对、回放和人工复核验收。

    Args:
        config: 已严格校验的V3配置。
        items: pilot四项或正式四十项。

    Returns:
        ``(JSON报告, episode CSV行)``。
    """
    progress = load_progress(config)
    completed = progress["completed"]
    errors: list[str] = []
    redo_keys: set[str] = set()
    selected_keys = [item.queue_key for item in items]
    missing = [key for key in selected_keys if key not in completed]
    for key in missing:
        _append_error(errors, redo_keys, f"缺少计划键: {key}", [key])
    records = {key: completed[key] for key in selected_keys if key in completed}
    episode_rows: list[dict[str, Any]] = []
    task_counts: Counter[str] = Counter()
    frame_counts: list[int] = []
    actions_by_key: dict[str, np.ndarray] = {}
    item_by_key = {item.queue_key: item for item in items}
    with MugTabletopEnv() as env:
        arm_limits = np.asarray(env.model.actuator_ctrlrange[:6], dtype=np.float64)
        for key, record in records.items():
            item = item_by_key[key]
            shard = config.shard_root / record["shard_name"]
            item_errors: list[str] = []
            try:
                validate_episode_shard(shard, record)
                table = read_shard_table(shard)
                states = vector_column(table, "observation.state")
                actions = vector_column(table, "action")
                poses = vector_column(table, "mug_initial_pose")
                actions_by_key[key] = actions
                frame_count = int(record["frame_count"])
                frame_counts.append(frame_count)
                task_counts[item.task_id] += 1
                if not 1 <= frame_count <= config.max_frames:
                    item_errors.append("frame_count_out_of_range")
                if np.any(actions[:, :6] < arm_limits[:, 0]) or np.any(actions[:, :6] > arm_limits[:, 1]):
                    item_errors.append("joint_limit_violation")
                if np.any(actions[:, 6] < 0.0) or np.any(actions[:, 6] > 1.0):
                    item_errors.append("gripper_range_violation")
                if not _has_close_then_release(actions):
                    item_errors.append("missing_gripper_close_then_release")
                if set(task_texts(shard)) != {item.prompt}:
                    item_errors.append("noncanonical_task")
                seeds = {int(np.asarray(value).reshape(-1)[0]) for value in table["scene_seed"].to_pylist()}
                if seeds != {item.scene_seed}:
                    item_errors.append("scene_seed_column_mismatch")
                expected_pose = np.asarray(record["mug_initial_pose"], dtype=np.float32)
                if not np.array_equal(poses, np.repeat(expected_pose[None], frame_count, axis=0)):
                    item_errors.append("mug_pose_column_mismatch")
                reset_snapshot = env.reset(item.scene_seed)
                if not np.array_equal(
                    np.asarray(reset_snapshot.mug_initial_pose, dtype=np.float64),
                    np.asarray(record["mug_initial_pose"], dtype=np.float64),
                ):
                    item_errors.append("mug_pose_not_reproducible")
                replay_success, replay_failure = replay_actions(env, item, actions, config.fps)
                if not replay_success:
                    item_errors.append(f"deterministic_replay_failed:{replay_failure}")
                if not np.isfinite(states).all() or not np.isfinite(actions).all() or not np.isfinite(poses).all():
                    item_errors.append("non_finite_values")
            except Exception as exc:
                item_errors.append(f"validation_exception:{type(exc).__name__}:{exc}")
                frame_count = int(record.get("frame_count", 0))
            if item_errors:
                _append_error(errors, redo_keys, f"{key}: {item_errors}", [key])
            episode_rows.append({
                "queue_index": item.queue_index,
                "queue_key": key,
                "scene_seed": item.scene_seed,
                "task_id": item.task_id,
                "task": item.prompt,
                "frame_count": frame_count,
                "status": "fail" if item_errors else "pass",
                "errors": ";".join(item_errors),
            })

    for seed in dict.fromkeys(item.scene_seed for item in items):
        scene_items = [item for item in items if item.scene_seed == seed]
        scene_keys = [item.queue_key for item in scene_items]
        scene_records = [records.get(key) for key in scene_keys]
        if any(record is None for record in scene_records):
            continue
        if {item.task_id for item in scene_items} != set(config.tasks):
            _append_error(errors, redo_keys, f"scene={seed}未恰好覆盖蓝黄任务", scene_keys)
        state_hashes = {record["initial_robot_state_sha256"] for record in scene_records if record}
        poses = {
            json.dumps(record["mug_initial_pose"], separators=(",", ":"))
            for record in scene_records if record
        }
        if len(state_hashes) != 1 or len(poses) != 1:
            _append_error(errors, redo_keys, f"scene={seed}蓝黄初始条件不一致", scene_keys)

    scene_seeds = list(dict.fromkeys(item.scene_seed for item in items))
    expected_counts = Counter({task_id: len(scene_seeds) for task_id in config.tasks})
    if task_counts != expected_counts:
        errors.append(f"任务数量不正确: expected={dict(expected_counts)}, actual={dict(task_counts)}")
    review = _read_review_status(config.work_root / REVIEW_FILENAME)
    for seed in scene_seeds:
        scene_keys = [item.queue_key for item in items if item.scene_seed == seed]
        review_status = review.get(seed)
        if review_status != "pass":
            affected_keys = scene_keys
            if review_status in {"redo_mug_on_blue", "redo_mug_on_yellow"}:
                redo_task = review_status.removeprefix("redo_")
                affected_keys = [
                    item.queue_key
                    for item in items
                    if item.scene_seed == seed and item.task_id == redo_task
                ]
            _append_error(
                errors, redo_keys,
                f"scene={seed}人工复核未PASS: {review_status or 'missing'}",
                affected_keys,
            )
        montage = config.work_root / "review_montages" / f"scene_{seed}.mp4"
        if not _montage_is_valid(montage):
            _append_error(errors, redo_keys, f"scene={seed}蒙太奇缺失或损坏", scene_keys)

    report = {
        "status": "pass" if not errors else "fail",
        "mode": "pilot" if len(items) == 4 else "full",
        "validated_at": utc_now(),
        "validated_keys": selected_keys,
        "episode_count": len(records),
        "scene_count": len(scene_seeds),
        "task_counts": dict(task_counts),
        "canonical_count": len(records),
        "other_prompt_count": 0,
        "frame_count": {
            "minimum": min(frame_counts) if frame_counts else 0,
            "maximum": max(frame_counts) if frame_counts else 0,
            "total": sum(frame_counts),
        },
        "deterministic_replay_required": True,
        "replay_terminal_hold_frames": 1,
        "grid_coverage": {"columns": 4, "rows": 5, "covered_cells": 20 if len(items) == 40 else 2},
        "errors": errors,
        "redo_keys": [item.queue_key for item in items if item.queue_key in redo_keys],
    }
    return report, episode_rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """以Excel友好的UTF-8 BOM写出episode验收清单。

    Args:
        path: 目标CSV路径。
        rows: 结构一致的episode记录；允许为空。
    """
    fieldnames = [
        "queue_index", "queue_key", "scene_seed", "task_id", "task",
        "frame_count", "status", "errors",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_redo_keys(path: Path, keys: Sequence[str]) -> None:
    """写出每行一个、可直接复制到redo命令的队列键。

    Args:
        path: ``redo_keys.txt``路径。
        keys: 按计划顺序排列的失败键。
    """
    path.write_text("".join(f"{key}\n" for key in keys), encoding="utf-8")


def _materialize_final(config: Any) -> None:
    """按固定40项队列把验收通过的分片物化为最终数据集。

    Args:
        config: 当前V3配置。

    Raises:
        FileExistsError: 最终目录非空时抛出。
        RuntimeError: 视频帧数、episode顺序或最终重读不一致时抛出。
    """
    if config.root.exists() and any(config.root.iterdir()):
        raise FileExistsError(f"最终V3目录非空，拒绝覆盖: {config.root}")
    if config.root.exists():
        config.root.rmdir()
    temporary = config.root.parent / f".{config.root.name}_finalizing_{uuid.uuid4().hex}"
    progress = load_progress(config)
    writer: MugEpisodeWriter | None = MugEpisodeWriter(
        temporary,
        contract_extras={
            "config_sha256": config.sha256,
            "scene_seeds": list(config.scene_seeds),
            "matrix": "20_scenes_x_2_tasks",
        },
    )
    try:
        for item in build_plan(config):
            record = progress["completed"][item.queue_key]
            shard = config.shard_root / record["shard_name"]
            table = read_shard_table(shard)
            states = vector_column(table, "observation.state")
            actions = vector_column(table, "action")
            camera_frames: dict[str, list[np.ndarray]] = {}
            for feature, output_name in (
                ("observation.images.agent", "agent"),
                ("observation.images.wrist", "wrist"),
            ):
                paths = list((shard / "videos" / feature).glob("chunk-*/*.mp4"))
                if len(paths) != 1:
                    raise RuntimeError(f"物化时视频数量错误: {item.queue_key}, {feature}")
                camera_frames[output_name] = decode_video(paths[0])
            frame_count = int(record["frame_count"])
            if any(len(frames) != frame_count for frames in camera_frames.values()):
                raise RuntimeError(f"物化时视频帧数漂移: {item.queue_key}")
            snapshot = MugSceneSnapshot(
                scene_seed=item.scene_seed,
                mug_initial_pose=np.asarray(record["mug_initial_pose"], dtype=np.float64),
                pad_positions=np.asarray([
                    config.fixed_pad_positions["blue"], config.fixed_pad_positions["yellow"],
                ], dtype=np.float64),
            )
            for frame_index in range(frame_count):
                writer.add_frame(
                    {
                        "agent": camera_frames["agent"][frame_index],
                        "wrist": camera_frames["wrist"][frame_index],
                    },
                    states[frame_index], actions[frame_index], snapshot, item.prompt,
                )
            actual_index = writer.save_episode(item.task_id, item.prompt, snapshot, frame_count)
            if actual_index != item.queue_index:
                raise RuntimeError(
                    f"V3最终episode顺序漂移: expected={item.queue_index}, actual={actual_index}"
                )
        writer.close()
        writer = None
        temporary.replace(config.root)
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    configure_hf_datasets_cache(config.root.parent / ".hf-lerobot-cache")
    dataset = LeRobotDataset(config.repo_id, root=config.root, video_backend="pyav")
    if int(dataset.meta.total_episodes) != 40:
        raise RuntimeError(f"V3最终数据集episode数不是40: {dataset.meta.total_episodes}")
    for index in range(len(dataset)):
        sample = dataset[index]
        for feature in CAMERA_FEATURES:
            image = sample[feature]
            shape = tuple(image.shape)
            if shape not in {(3, 256, 256), (256, 256, 3)}:
                raise RuntimeError(f"V3最终视频重读shape错误: frame={index}, {feature}={shape}")


def run_validation(args: argparse.Namespace) -> int:
    """运行pilot或正式验收，并按需物化最终40条数据集。

    Args:
        args: ``build_parser``解析出的参数。

    Returns:
        PASS返回0，FAIL返回1。

    Raises:
        ValueError: pilot与finalize同时使用时抛出。
    """
    if args.pilot and args.finalize:
        raise ValueError("--pilot与--finalize不能同时使用")
    config = load_config(args.config)
    load_progress(config)
    items = plan_for_mode(config, pilot=args.pilot)
    report, rows = _validate_records(config, items)
    report_path = config.work_root / (
        PILOT_VALIDATION_FILENAME if args.pilot else VALIDATION_FILENAME
    )
    atomic_write_json(report_path, report)
    _write_csv(config.work_root / EPISODE_CSV_FILENAME, rows)
    _write_redo_keys(config.work_root / REDO_KEYS_FILENAME, report["redo_keys"])
    if report["status"] != "pass":
        print(
            f"杯子V3验收FAIL；剩余正式采集不开放。redo keys见: "
            f"{config.work_root / REDO_KEYS_FILENAME}"
        )
        return 1
    if args.pilot:
        print("杯子V3 pilot PASS：4/4严格成功且人工复核通过，可以开始剩余36条。")
        return 0
    if args.finalize:
        _materialize_final(config)
        shutil.copy2(report_path, config.root / VALIDATION_FILENAME)
        shutil.copy2(config.work_root / EPISODE_CSV_FILENAME, config.root / EPISODE_CSV_FILENAME)
        shutil.copy2(config.work_root / REDO_KEYS_FILENAME, config.root / REDO_KEYS_FILENAME)
        shutil.copytree(
            config.work_root / "review_montages",
            config.root / "review_montages",
            dirs_exist_ok=True,
        )
        print(f"杯子V3正式验收PASS并已物化40条数据集: {config.root}")
    else:
        print("杯子V3全量自动与人工验收PASS；添加--finalize后可物化最终数据集。")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """解析参数并执行杯子V3验收。

    Args:
        argv: 可选命令行参数；为空时读取当前进程参数。

    Returns:
        验收流程退出码。
    """
    return run_validation(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
