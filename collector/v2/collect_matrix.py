"""按scene优先的Grounding v2反事实矩阵人工采集入口。"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import glfw
import numpy as np

from collector.collection_plan import (
    PILOT_VALIDATION_FILENAME,
    PROGRESS_FILENAME,
    REVIEW_FILENAME,
    QueueItem,
    atomic_write_json,
    initial_reference_path,
    initialize_workspace,
    load_initial_reference,
    load_config,
    load_progress,
    plan_for_mode,
    record_completion,
    save_initial_reference,
    utc_now,
    validate_completed_shards,
    validate_frame_count,
)
from collector.control import DifferentialIKController, read_teleop_delta
from collector.dataset_io import LeRobotEpisodeWriter
from collector.state_machine import CollectionPhase, CollectionStateMachine
from sim.environment import CleanTabletopEnv, SceneSnapshot
from sim.mujoco_viewer import EmbeddedCameraViewer


DISPLAY_HZ = 60
NOTICE_SUCCESS = "Strict success: Enter saves; Backspace retries this key"
NOTICE_CANCELLED = "Cancelled: retrying the same queue key"
NOTICE_DISCARDED = "Discarded: retrying the same queue key"
NOTICE_FRAME_LIMIT = "400-frame limit: discarded and retrying this queue key"


def build_parser() -> argparse.ArgumentParser:
    """创建矩阵采集命令行解析器。"""
    parser = argparse.ArgumentParser(description="采集SmolVLA颜色Grounding v2配对矩阵")
    parser.add_argument("--config", type=Path, required=True, help="锁定的Grounding v2 YAML配置")
    parser.add_argument("--pilot", action="store_true", help="只开放前两个scene的8条pilot")
    parser.add_argument("--resume", action="store_true", help="严格校验后恢复已有工作区")
    parser.add_argument(
        "--redo-key",
        default=None,
        help="归档并重采一个已完成键，格式scene=...|task=...|prompt=canonical",
    )
    return parser


def _status_text(
    item: QueueItem,
    phase: CollectionPhase,
    frame_count: int,
    completed_count: int,
    target_count: int,
    error: str,
) -> str:
    """生成MuJoCo左下角只含ASCII的状态文本。"""
    keys = (
        "Enter: save | Backspace: retry"
        if phase == CollectionPhase.PENDING_CONFIRMATION
        else "WASD/RF + arrows/QE | Space: grip | Z: retry"
    )
    lines = [
        item.prompt,
        f"scene={item.scene_seed} task={item.task_id} position={item.collection_position}/4",
        f"phase={phase.value} frames={frame_count}/400 completed={completed_count}/{target_count}",
        keys,
    ]
    if error:
        lines.append(error)
    return "\n".join(lines)


def _initial_observation(
    viewer: EmbeddedCameraViewer,
    env: CleanTabletopEnv,
    config: Any,
    item: QueueItem,
    snapshot: SceneSnapshot,
    completed: dict[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, str]]:
    """创建或复用同scene四任务共享的无损初始观测基准。

    MuJoCo/OpenGL重复渲染同一静态场景时可能出现极少量像素差，因此不能把
    四次独立渲染直接用于逐字节哈希。首个任务无损保存基准，后续任务把该基准
    作为动作前首帧，既保持物理初态一致，也保证跨任务和跨中断哈希完全一致。
    """
    reference_path = initial_reference_path(config, item.scene_seed)
    if not reference_path.is_file():
        existing_scene_records = [
            record for record in completed.values()
            if int(record.get("scene_seed", -1)) == item.scene_seed
        ]
        if existing_scene_records:
            first_key = existing_scene_records[0]["queue_key"]
            raise RuntimeError(
                "该scene已有旧版分片但缺少无损初始观测基准，不能安全伪造哈希。"
                f"请先重采已有键: --resume --redo-key \"{first_key}\""
            )
        live_images = viewer.capture_training_images(256)
        save_initial_reference(
            config,
            item.scene_seed,
            env.get_state(),
            live_images["agent"],
            live_images["wrist"],
            snapshot.cube_initial_poses,
        )
    reference = load_initial_reference(config, item.scene_seed)
    current_state = env.get_state().astype(np.float32)
    if not np.array_equal(reference["state"], current_state):
        raise RuntimeError(f"scene={item.scene_seed}初始机器人状态与无损基准不一致")
    if not np.array_equal(reference["cube_initial_poses"], snapshot.cube_initial_poses):
        raise RuntimeError(f"scene={item.scene_seed}积木初始位姿与无损基准不一致")
    images = {"agent": reference["agent"], "wrist": reference["wrist"]}
    hashes = {
        "initial_robot_state_sha256": reference["initial_robot_state_sha256"],
        "initial_agent_raw_sha256": reference["initial_agent_raw_sha256"],
        "initial_wrist_raw_sha256": reference["initial_wrist_raw_sha256"],
    }
    return images, reference["state"], hashes


def _enforce_scene_pairing(
    item: QueueItem,
    snapshot: SceneSnapshot,
    hashes: dict[str, str],
    completed: dict[str, Any],
) -> None:
    """在写入前保证同scene四任务的初始状态逐字节一致。"""
    expected_pose = np.asarray(snapshot.cube_initial_poses, dtype=np.float64)
    for record in completed.values():
        if int(record["scene_seed"]) != item.scene_seed:
            continue
        mismatches = [name for name, value in hashes.items() if record.get(name) != value]
        recorded_pose = np.asarray(record.get("cube_initial_poses"), dtype=np.float64)
        if recorded_pose.shape != (2, 7) or not np.array_equal(recorded_pose, expected_pose):
            mismatches.append("cube_initial_poses")
        if mismatches:
            raise RuntimeError(
                f"同scene初始条件不一致，已在写入前停止: scene={item.scene_seed}, {mismatches}"
            )


def _archive_orphan_shards(config: Any, progress: dict[str, Any]) -> None:
    """把崩溃遗留且未登记的分片移出活动目录。"""
    active = {str(record.get("shard_name")) for record in progress["completed"].values()}
    orphans = [path for path in config.shard_root.iterdir() if path.is_dir() and path.name not in active]
    if not orphans:
        return
    archive = config.work_root / "abandoned_shards" / utc_now().replace(":", "-")
    archive.mkdir(parents=True, exist_ok=True)
    for path in orphans:
        shutil.move(str(path), str(archive / path.name))


def _prepare_redo(config: Any, queue_key: str) -> QueueItem:
    """归档旧分片、删除活动记录并把人工复核状态重置为pending。"""
    if not queue_key:
        raise ValueError("--redo-key不能为空")
    progress = load_progress(config)
    record = progress["completed"].get(queue_key)
    plan = {item.queue_key: item for item in plan_for_mode(config, pilot=False)}
    if queue_key not in plan:
        raise ValueError(f"未知队列键: {queue_key}")
    if record is None:
        raise ValueError(f"只能重采已完成队列键: {queue_key}")
    source = config.shard_root / record["shard_name"]
    archive = config.work_root / "archived_shards" / utc_now().replace(":", "-")
    archive.mkdir(parents=True, exist_ok=True)
    if not source.is_dir():
        raise FileNotFoundError(f"待重采旧分片缺失: {source}")
    shutil.move(str(source), str(archive / source.name))
    del progress["completed"][queue_key]
    progress["updated_at"] = utc_now()
    atomic_write_json(config.work_root / PROGRESS_FILENAME, progress)
    review_path = config.work_root / REVIEW_FILENAME
    if review_path.is_file():
        rows = review_path.read_text(encoding="utf-8-sig").splitlines()
        updated = []
        for line in rows:
            if line.startswith(f"{plan[queue_key].scene_seed},"):
                updated.append(f"{plan[queue_key].scene_seed},pending")
            else:
                updated.append(line)
        review_path.write_text("\n".join(updated) + "\n", encoding="utf-8-sig")
    return plan[queue_key]


def _require_pilot_gate(config: Any, progress: dict[str, Any]) -> None:
    """全量采集前要求前8条完整且pilot验收文件明确PASS。"""
    pilot_items = plan_for_mode(config, pilot=True)
    completed = progress["completed"]
    missing = [item.queue_key for item in pilot_items if item.queue_key not in completed]
    if missing:
        raise RuntimeError(f"全量采集前必须先完成8条pilot，缺少{len(missing)}条")
    path = config.work_root / PILOT_VALIDATION_FILENAME
    try:
        validation = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("请先生成pilot蒙太奇、完成人工复核并运行pilot验收") from exc
    if validation.get("status") != "pass" or validation.get("validated_keys") != [
        item.queue_key for item in pilot_items
    ]:
        raise RuntimeError("pilot验收未PASS或验收键与当前计划不一致")


def _create_writer(config: Any, item: QueueItem) -> LeRobotEpisodeWriter:
    """为一个唯一队列键创建单episode LeRobot分片。"""
    return LeRobotEpisodeWriter(
        config.shard_root / item.shard_name,
        dataset_version=config.dataset_version,
        repo_id=config.repo_id,
        contract_extras={
            "config_sha256": config.sha256,
            "queue_key": item.queue_key,
            "prompt_mode": item.prompt_mode,
        },
    )


def _collect_one(
    config: Any,
    item: QueueItem,
    env: CleanTabletopEnv,
    viewer: EmbeddedCameraViewer,
    completed_count: int,
    target_count: int,
) -> dict[str, Any] | None:
    """持续重试同一队列键，直到人工确认保存或关闭Viewer。"""
    writer = _create_writer(config, item)
    machine = CollectionStateMachine()
    recent_error = ""
    display_frame = 0
    saved = False
    writer_closed = False
    try:
        while viewer.is_running():
            snapshot = env.reset(item.scene_seed)
            controller = DifferentialIKController(env)
            completed = load_progress(config)["completed"]
            initial_images, initial_state, initial_hashes = _initial_observation(
                viewer, env, config, item, snapshot, completed,
            )
            _enforce_scene_pairing(item, snapshot, initial_hashes, completed)
            machine.reset()
            physics_accumulator = 0.0
            retry = False
            while viewer.is_running() and not retry:
                frame_start = time.perf_counter()
                if machine.phase == CollectionPhase.PENDING_CONFIRMATION:
                    if viewer.consume_key_press(glfw.KEY_ENTER):
                        frame_count = machine.frame_count
                        validate_frame_count(frame_count, config.max_frames)
                        writer.save_episode(
                            item.task_id,
                            "canonical",
                            item.prompt,
                            snapshot,
                            frame_count,
                        )
                        writer.close()
                        writer_closed = True
                        machine.confirm()
                        saved = True
                        return {
                            "queue_key": item.queue_key,
                            "episode_index": item.queue_index,
                            "shard_episode_index": 0,
                            "shard_name": item.shard_name,
                            "scene_seed": item.scene_seed,
                            "task_id": item.task_id,
                            "prompt_mode": item.prompt_mode,
                            "task": item.prompt,
                            "frame_count": frame_count,
                            **initial_hashes,
                            "cube_initial_poses": snapshot.cube_initial_poses.tolist(),
                            "completed_at": utc_now(),
                        }
                    backspace = viewer.consume_key_press(glfw.KEY_BACKSPACE)
                    cancel = viewer.consume_key_press(glfw.KEY_Z)
                    if backspace or cancel:
                        writer.discard_episode()
                        machine.discard()
                        recent_error = NOTICE_CANCELLED if cancel else NOTICE_DISCARDED
                        retry = True
                elif viewer.consume_key_press(glfw.KEY_Z):
                    writer.discard_episode()
                    machine.discard()
                    recent_error = NOTICE_CANCELLED
                    retry = True

                sample_due = display_frame % (DISPLAY_HZ // config.fps) == 0
                if sample_due and machine.phase != CollectionPhase.PENDING_CONFIRMATION and not retry:
                    evaluation = env.evaluate_task(
                        item.task_id,
                        elapsed_seconds=machine.frame_count / config.fps,
                        timeout_seconds=config.max_frames / config.fps,
                    )
                    if evaluation.success and machine.phase == CollectionPhase.RECORDING:
                        if machine.frame_count < config.max_frames and machine.observe_action(False):
                            writer.add_frame(
                                viewer.capture_training_images(256),
                                env.get_state(),
                                controller.last_action,
                                snapshot,
                                item.prompt,
                            )
                        machine.observe_success(True)
                        recent_error = NOTICE_SUCCESS
                    elif machine.phase == CollectionPhase.RECORDING and machine.frame_count >= config.max_frames:
                        writer.discard_episode()
                        machine.discard()
                        recent_error = NOTICE_FRAME_LIMIT
                        retry = True
                    else:
                        delta = read_teleop_delta(viewer, controller.gripper)
                        previous_action = controller.last_action.copy()
                        previous_gripper = controller.gripper
                        action, ik_error = controller.command(delta)
                        effective = bool(delta.meaningful and not ik_error)
                        if ik_error:
                            controller.last_action = previous_action
                            controller.gripper = previous_gripper
                            action = previous_action
                            recent_error = ik_error
                        elif effective:
                            recent_error = ""
                        if machine.observe_action(effective):
                            first_frame = machine.frame_count == 1
                            writer.add_frame(
                                initial_images if first_frame else viewer.capture_training_images(256),
                                initial_state if first_frame else env.get_state(),
                                action,
                                snapshot,
                                item.prompt,
                            )
                        env.apply_joint_action(action)

                if machine.phase == CollectionPhase.RECORDING and not retry:
                    physics_accumulator += 1.0 / DISPLAY_HZ
                    physics_steps = int(physics_accumulator / env.model.opt.timestep)
                    if physics_steps:
                        env.step(physics_steps)
                        physics_accumulator -= physics_steps * env.model.opt.timestep

                viewer.set_status(
                    "Grounding v2 Matrix Collector",
                    _status_text(
                        item, machine.phase, machine.frame_count,
                        completed_count, target_count, recent_error,
                    ),
                )
                viewer.render()
                display_frame += 1
                remaining = 1.0 / DISPLAY_HZ - (time.perf_counter() - frame_start)
                if remaining > 0:
                    time.sleep(remaining)
        return None
    finally:
        if machine.phase != CollectionPhase.IDLE:
            writer.discard_episode()
        if not writer_closed:
            try:
                writer.close()
            except Exception:
                pass
        shard = config.shard_root / item.shard_name
        if shard.is_dir() and not saved:
            shutil.rmtree(shard, ignore_errors=True)


def run_collection(args: argparse.Namespace) -> int:
    """执行pilot、全量续采或指定键重采。"""
    config = load_config(args.config)
    if args.redo_key and not args.resume:
        raise ValueError("--redo-key必须与--resume同时使用")
    progress = initialize_workspace(config, resume=args.resume)
    _archive_orphan_shards(config, progress)
    progress = load_progress(config)
    validate_completed_shards(config, progress)
    if args.redo_key:
        target_plan = [_prepare_redo(config, args.redo_key)]
    else:
        if not args.pilot:
            _require_pilot_gate(config, progress)
        target_plan = plan_for_mode(config, pilot=args.pilot)
    completed = load_progress(config)["completed"]
    pending = [item for item in target_plan if item.queue_key not in completed]
    if not pending:
        print(f"目标队列已全部完成: {len(target_plan)}/{len(target_plan)}")
        return 0

    with CleanTabletopEnv() as env:
        with EmbeddedCameraViewer(
            env.model,
            env.data,
            title="SmolVLA Grounding v2 Matrix Collector",
            show_fixed_cameras=True,
        ) as viewer:
            for item in pending:
                current_completed = load_progress(config)["completed"]
                record = _collect_one(
                    config, item, env, viewer,
                    completed_count=sum(key in current_completed for key in [entry.queue_key for entry in target_plan]),
                    target_count=len(target_plan),
                )
                if record is None:
                    print(f"Viewer已关闭，当前未确认键未保存: {item.queue_key}")
                    return 1
                record_completion(config, item, record)
                print(
                    f"已确认分片 {record['episode_index']:03d}: {item.queue_key}, "
                    f"frames={record['frame_count']}"
                )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """解析参数并启动Grounding v2矩阵采集。"""
    return run_collection(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
