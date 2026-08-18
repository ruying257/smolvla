"""杯子V3按scene配对的人工键盘矩阵采集入口。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import glfw
import numpy as np

from collector.common.control import DifferentialIKController, read_teleop_delta
from collector.common.state_machine import CollectionPhase, CollectionStateMachine
from collector.v3.collection_plan import (
    PILOT_VALIDATION_FILENAME,
    QueueItem,
    build_plan,
    initialize_workspace,
    load_config,
    load_progress,
    plan_for_mode,
    prepare_redo,
    record_completion,
    utc_now,
    validate_completed_shards,
)
from collector.v3.dataset_io import MugEpisodeWriter
from sim.mug_environment import MugSceneSnapshot, MugTabletopEnv
from sim.mujoco_viewer import EmbeddedCameraViewer


NOTICE_SUCCESS = "Strict success: Enter saves; Backspace retries this key"
NOTICE_CANCELLED = "Cancelled: retrying the same queue key"
NOTICE_DISCARDED = "Discarded: retrying the same queue key"
NOTICE_FRAME_LIMIT = "400-frame limit: discarded and retrying this key"


def build_parser() -> argparse.ArgumentParser:
    """创建杯子V3矩阵采集参数解析器。

    Returns:
        支持pilot、严格resume和单键redo的解析器。
    """
    parser = argparse.ArgumentParser(description="采集SmolVLA杯子V3双任务配对矩阵")
    parser.add_argument("--config", type=Path, required=True, help="锁定的杯子V3 YAML配置")
    parser.add_argument("--pilot", action="store_true", help="只开放两个scene的4条pilot")
    parser.add_argument("--resume", action="store_true", help="严格校验后恢复已有工作区")
    parser.add_argument(
        "--redo-key",
        default=None,
        help="归档并重采一个已完成键，格式scene=...|task=...|prompt=canonical",
    )
    return parser


def _hash_array(value: Any) -> str:
    """计算包含dtype、shape和字节内容的数组SHA-256。

    Args:
        value: NumPy兼容数组。

    Returns:
        可用于配对检查的十六进制摘要。
    """
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _status_text(
    item: QueueItem,
    phase: CollectionPhase,
    frame_count: int,
    completed_count: int,
    target_count: int,
    max_frames: int,
    error: str,
) -> str:
    """生成MuJoCo overlay可安全显示的ASCII采集状态。

    Args:
        item: 当前队列项。
        phase: 当前采集状态机阶段。
        frame_count: 当前episode已录帧数。
        completed_count: 当前模式已完成项数。
        target_count: 当前模式总项数。
        max_frames: 单条帧数上限。
        error: 最近一次IK或重试提示。

    Returns:
        不含中文字符的多行状态文本，避免Viewer乱码。
    """
    keys = (
        "Enter: save | Backspace: retry"
        if phase == CollectionPhase.PENDING_CONFIRMATION
        else "WASD/RF + arrows/QE | Space: grip | Z: retry"
    )
    lines = [
        item.prompt,
        f"scene={item.scene_seed} task={item.task_id} position={item.collection_position}/2",
        f"phase={phase.value} frames={frame_count}/{max_frames} "
        f"completed={completed_count}/{target_count}",
        keys,
    ]
    if error:
        lines.append(error)
    return "\n".join(lines)


def _enforce_scene_pairing(
    item: QueueItem,
    snapshot: MugSceneSnapshot,
    initial_state: np.ndarray,
    completed: dict[str, Any],
) -> None:
    """保证同seed蓝黄任务的杯子位姿和机器人初态完全一致。

    Args:
        item: 当前待采队列项。
        snapshot: 本次真实reset快照。
        initial_state: 首次动作前七维机器人状态。
        completed: 当前所有已完成记录。

    Raises:
        RuntimeError: 同seed已有记录与当前初始状态或杯子位姿不一致时抛出。
    """
    state_hash = _hash_array(np.asarray(initial_state, dtype=np.float32))
    pose = np.asarray(snapshot.mug_initial_pose, dtype=np.float64)
    for record in completed.values():
        if int(record.get("scene_seed", -1)) != item.scene_seed:
            continue
        mismatches: list[str] = []
        if record.get("initial_robot_state_sha256") != state_hash:
            mismatches.append("initial_robot_state")
        recorded_pose = np.asarray(record.get("mug_initial_pose"), dtype=np.float64)
        if recorded_pose.shape != (7,) or not np.array_equal(recorded_pose, pose):
            mismatches.append("mug_initial_pose")
        if mismatches:
            raise RuntimeError(
                f"同seed蓝黄初始条件不一致: scene={item.scene_seed}, fields={mismatches}"
            )


def _archive_orphan_shards(config: Any, progress: dict[str, Any]) -> None:
    """归档崩溃遗留且尚未登记完成的分片目录。

    Args:
        config: 当前V3配置。
        progress: 当前已验证进度。
    """
    if not config.shard_root.is_dir():
        return
    active = {str(record["shard_name"]) for record in progress["completed"].values()}
    orphans = [path for path in config.shard_root.iterdir() if path.is_dir() and path.name not in active]
    if not orphans:
        return
    archive = config.work_root / "abandoned_shards" / utc_now().replace(":", "-")
    archive.mkdir(parents=True, exist_ok=True)
    for orphan in orphans:
        shutil.move(str(orphan), str(archive / orphan.name))


def _require_pilot_gate(config: Any, progress: dict[str, Any]) -> None:
    """在开放剩余36条前验证四条pilot已经明确PASS。

    Args:
        config: 当前V3配置。
        progress: 当前恢复进度。

    Raises:
        RuntimeError: pilot键不完整、报告缺失或validated_keys漂移时抛出。
    """
    pilot_items = plan_for_mode(config, pilot=True)
    missing = [item.queue_key for item in pilot_items if item.queue_key not in progress["completed"]]
    if missing:
        raise RuntimeError(f"正式采集前必须完成4条pilot，缺少{len(missing)}条")
    path = config.work_root / PILOT_VALIDATION_FILENAME
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("pilot验收报告缺失；不得开始剩余36条") from exc
    expected_keys = [item.queue_key for item in pilot_items]
    if report.get("status") != "pass" or report.get("validated_keys") != expected_keys:
        raise RuntimeError("pilot尚未PASS或报告与当前四个pilot键不一致")


def _create_writer(config: Any, item: QueueItem) -> MugEpisodeWriter:
    """为唯一队列键创建全新单episode V3分片写入器。

    Args:
        config: 当前V3配置。
        item: 当前队列项。

    Returns:
        契约包含配置哈希和队列键的独立写入器。
    """
    return MugEpisodeWriter(
        config.shard_root / item.shard_name,
        contract_extras={
            "config_sha256": config.sha256,
            "queue_key": item.queue_key,
            "scene_seeds": list(config.scene_seeds),
        },
    )


def _collect_one(
    config: Any,
    item: QueueItem,
    env: MugTabletopEnv,
    viewer: EmbeddedCameraViewer,
    completed_count: int,
    target_count: int,
) -> dict[str, Any] | None:
    """人工持续重试同一键，直到Enter确认或Viewer关闭。

    Args:
        config: 当前V3配置。
        item: 当前唯一队列项。
        env: 真实杯子MuJoCo环境。
        viewer: 提供键盘和三路辅助显示的Viewer。
        completed_count: 当前模式已完成数量。
        target_count: 当前模式总数量。

    Returns:
        Enter确认后返回完成记录；关闭Viewer或Ctrl+C前无确认则返回 ``None``。
    """
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
            initial_state = env.get_state().astype(np.float32)
            completed = load_progress(config)["completed"]
            _enforce_scene_pairing(item, snapshot, initial_state, completed)
            machine.reset()
            physics_accumulator = 0.0
            retry = False
            while viewer.is_running() and not retry:
                frame_start = time.perf_counter()
                if machine.phase == CollectionPhase.PENDING_CONFIRMATION:
                    if viewer.consume_key_press(glfw.KEY_ENTER):
                        frame_count = machine.frame_count
                        writer.save_episode(item.task_id, item.prompt, snapshot, frame_count)
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
                            "config_sha256": config.sha256,
                            "initial_robot_state_sha256": _hash_array(initial_state),
                            "mug_initial_pose": snapshot.mug_initial_pose.tolist(),
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

                sample_due = display_frame % (config.viewer_fps // config.fps) == 0
                if sample_due and machine.phase != CollectionPhase.PENDING_CONFIRMATION and not retry:
                    evaluation = env.evaluate_task(
                        item.task_id,
                        elapsed_seconds=machine.frame_count / config.fps,
                        timeout_seconds=config.max_frames / config.fps,
                    )
                    if evaluation.success and machine.phase == CollectionPhase.RECORDING:
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
                            writer.add_frame(
                                viewer.capture_training_images(256),
                                initial_state if machine.frame_count == 1 else env.get_state(),
                                action,
                                snapshot,
                                item.prompt,
                            )
                        env.apply_joint_action(action)

                if machine.phase == CollectionPhase.RECORDING and not retry:
                    physics_accumulator += 1.0 / config.viewer_fps
                    physics_steps = int(physics_accumulator / env.model.opt.timestep)
                    if physics_steps:
                        env.step(physics_steps)
                        physics_accumulator -= physics_steps * env.model.opt.timestep

                viewer.set_status(
                    "SmolVLA Mug V3 Collector",
                    _status_text(
                        item, machine.phase, machine.frame_count, completed_count,
                        target_count, config.max_frames, recent_error,
                    ),
                )
                viewer.render()
                display_frame += 1
                remaining = 1.0 / config.viewer_fps - (time.perf_counter() - frame_start)
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
    """执行pilot、全量续采或指定键局部重采。

    Args:
        args: ``build_parser``解析出的参数。

    Returns:
        全部目标键完成时返回0；Viewer提前关闭时返回1。

    Raises:
        ValueError: redo没有配合resume或配置不合法时抛出。
        RuntimeError: 分片损坏或pilot门禁未通过时抛出。
    """
    config = load_config(args.config)
    if args.redo_key and not args.resume:
        raise ValueError("--redo-key必须与--resume同时使用")
    progress = initialize_workspace(config, resume=args.resume)
    _archive_orphan_shards(config, progress)
    if args.redo_key:
        target_plan = [prepare_redo(config, args.redo_key)]
        validate_completed_shards(config, load_progress(config))
    else:
        progress = load_progress(config)
        validate_completed_shards(config, progress)
        if not args.pilot:
            _require_pilot_gate(config, progress)
        target_plan = plan_for_mode(config, pilot=args.pilot)
    completed = load_progress(config)["completed"]
    pending = [item for item in target_plan if item.queue_key not in completed]
    if not pending:
        print(f"V3目标队列已全部完成: {len(target_plan)}/{len(target_plan)}")
        return 0
    with MugTabletopEnv() as env:
        with EmbeddedCameraViewer(
            env.model,
            env.data,
            title="SmolVLA Mug V3 Matrix Collector",
            show_fixed_cameras=True,
        ) as viewer:
            for item in pending:
                current = load_progress(config)["completed"]
                target_keys = [entry.queue_key for entry in target_plan]
                record = _collect_one(
                    config,
                    item,
                    env,
                    viewer,
                    completed_count=sum(key in current for key in target_keys),
                    target_count=len(target_plan),
                )
                if record is None:
                    print(f"Viewer已关闭，未确认键未保存: {item.queue_key}")
                    return 1
                record_completion(config, item, record)
                print(f"已确认V3分片: {item.queue_key}, frames={record['frame_count']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """解析命令行并启动杯子V3人工采集。

    Args:
        argv: 可选参数列表；为空时读取当前进程参数。

    Returns:
        采集流程退出码。
    """
    return run_collection(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
