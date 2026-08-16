"""Windows与Ubuntu共用的MuJoCo键盘专家数据采集入口。"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

import glfw

from collector.control import DifferentialIKController, read_teleop_delta
from collector.dataset_io import DATASET_FPS, LeRobotEpisodeWriter
from collector.state_machine import CollectionPhase, CollectionStateMachine
from collector.task_spec import TASKS
from sim.environment import CleanTabletopEnv
from sim.mujoco_viewer import EmbeddedCameraViewer


DISPLAY_HZ = 60
NOTICE_DISCARDED = "Discarded successful episode; retrying the same seed"
NOTICE_CANCELLED = "Cancelled current attempt; retrying the same seed"
NOTICE_SUCCESS = "Strict success: press Enter to save or Backspace to discard"
NOTICE_TIMEOUT = "Episode timed out; retrying the same seed"


def parse_seed_list(value: str) -> list[int]:
    """解析逗号分隔的scene seed列表。

    Args:
        value: 例如 ``3,7,11`` 的文本。

    Returns:
        至少包含一个整数的列表。

    Raises:
        argparse.ArgumentTypeError: 内容为空或包含非整数时抛出。
    """
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--seeds 只接受逗号分隔的整数") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds 至少需要一个整数")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    """创建采集命令行参数解析器。

    Returns:
        配置完成的参数解析器。
    """
    parser = argparse.ArgumentParser(description="采集SmolVLA UR10e MuJoCo专家episode")
    parser.add_argument("--root", type=Path, required=True, help="LeRobot数据集输出目录")
    parser.add_argument("--task", choices=tuple(TASKS), default=None, help="本次采集的任务组合")
    parser.add_argument("--seed", type=int, default=0, help="未提供seed列表时的起始seed")
    parser.add_argument("--seeds", type=parse_seed_list, default=None, help="循环使用的显式seed列表")
    parser.add_argument("--episodes", type=int, default=1, help="本次需要确认保存的episode数量")
    parser.add_argument("--resume", action="store_true", help="严格校验后续采已有数据集")
    parser.add_argument("--timeout-seconds", type=float, default=40.0, help="单次尝试超时秒数")
    return parser


def select_task(task_id: str | None) -> str:
    """返回命令行任务或在启动Viewer前提示用户选择。

    Args:
        task_id: 命令行显式任务；为空时交互选择。

    Returns:
        四类任务之一的稳定标识。
    """
    if task_id is not None:
        return task_id
    task_ids = list(TASKS)
    print("请选择本次采集任务：")
    for index, candidate in enumerate(task_ids, start=1):
        print(f"  {index}. {TASKS[candidate].prompt('canonical')}")
    while True:
        answer = input("输入1-4: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(task_ids):
            return task_ids[int(answer) - 1]
        print("输入无效，请输入1、2、3或4。")


def choose_scene_seed(
    explicit_seeds: list[int] | None,
    base_seed: int,
    confirmed_count: int,
) -> int:
    """根据已确认数量选择当前scene seed。

    Args:
        explicit_seeds: 可选的循环seed列表。
        base_seed: 自动递增模式的起始seed。
        confirmed_count: 本次运行已经确认保存的数量。

    Returns:
        当前episode应使用的整数seed。
    """
    if explicit_seeds:
        return explicit_seeds[confirmed_count % len(explicit_seeds)]
    return base_seed + confirmed_count


def _status_text(
    phase: CollectionPhase,
    task_text: str,
    scene_seed: int,
    frames: int,
    saved: int,
    target: int,
    error: str,
) -> str:
    """构造Viewer左下角采集状态文本。

    Args:
        phase: 当前状态机阶段。
        task_text: 当前英文指令。
        scene_seed: 当前场景seed。
        frames: 当前缓冲区帧数。
        saved: 本次已确认保存数量。
        target: 本次目标数量。
        error: 最近一次控制错误。

    Returns:
        可直接交给MuJoCo overlay的多行文本。
    """
    instructions = (
        "Enter: save | Backspace: discard"
        if phase == CollectionPhase.PENDING_CONFIRMATION
        else "WASD/RF: move | arrows/QE: rotate | Space: gripper | Z: retry"
    )
    lines = [
        task_text,
        f"seed={scene_seed} phase={phase.value} frames={frames}",
        f"saved={saved}/{target}",
        instructions,
    ]
    if error:
        lines.append(error)
    return "\n".join(lines)


def run_collection(args: argparse.Namespace) -> int:
    """运行真实MuJoCo GUI采集循环。

    Args:
        args: 已解析的命令行参数。

    Returns:
        正常完成目标数量时返回0。
    """
    if args.episodes <= 0:
        raise ValueError("--episodes 必须大于0")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds 必须大于0")
    task_id = select_task(args.task)
    writer = LeRobotEpisodeWriter(args.root, resume=args.resume)
    machine = CollectionStateMachine()
    confirmed_count = 0
    recent_error = ""
    scene_seed = choose_scene_seed(args.seeds, args.seed, confirmed_count)

    try:
        with CleanTabletopEnv() as env:
            snapshot = env.reset(scene_seed)
            controller = DifferentialIKController(env)
            template_id = writer.next_template(task_id)
            task_text = TASKS[task_id].prompt(template_id)
            with EmbeddedCameraViewer(
                env.model,
                env.data,
                title="SmolVLA UR10e Expert Collector",
                show_fixed_cameras=True,
            ) as viewer:
                display_frame = 0
                physics_accumulator = 0.0
                while viewer.is_running() and confirmed_count < args.episodes:
                    frame_start = time.perf_counter()

                    if machine.phase == CollectionPhase.PENDING_CONFIRMATION:
                        if viewer.consume_key_press(glfw.KEY_ENTER):
                            frame_count = machine.frame_count
                            episode_index = writer.save_episode(
                                task_id,
                                template_id,
                                task_text,
                                snapshot,
                                frame_count,
                            )
                            if machine.confirm():
                                confirmed_count += 1
                                print(
                                    f"已保存 episode={episode_index}, seed={scene_seed}, "
                                    f"task={task_text}, frames={frame_count}"
                                )
                                if confirmed_count >= args.episodes:
                                    break
                                scene_seed = choose_scene_seed(args.seeds, args.seed, confirmed_count)
                                snapshot = env.reset(scene_seed)
                                controller.reset()
                                template_id = writer.next_template(task_id)
                                task_text = TASKS[task_id].prompt(template_id)
                                physics_accumulator = 0.0
                        elif viewer.consume_key_press(glfw.KEY_BACKSPACE):
                            writer.discard_episode()
                            machine.discard()
                            snapshot = env.reset(scene_seed)
                            controller.reset()
                            physics_accumulator = 0.0
                            recent_error = NOTICE_DISCARDED
                    elif viewer.consume_key_press(glfw.KEY_Z):
                        writer.discard_episode()
                        machine.discard()
                        snapshot = env.reset(scene_seed)
                        controller.reset()
                        physics_accumulator = 0.0
                        recent_error = NOTICE_CANCELLED

                    sample_due = display_frame % (DISPLAY_HZ // DATASET_FPS) == 0
                    if sample_due and machine.phase != CollectionPhase.PENDING_CONFIRMATION:
                        elapsed_seconds = machine.frame_count / DATASET_FPS
                        evaluation = env.evaluate_task(
                            task_id,
                            elapsed_seconds=elapsed_seconds,
                            timeout_seconds=args.timeout_seconds,
                        )
                        if evaluation.success and machine.phase == CollectionPhase.RECORDING:
                            images = viewer.capture_training_images(256)
                            if machine.observe_action(False):
                                writer.add_frame(
                                    images,
                                    env.get_state(),
                                    controller.last_action,
                                    snapshot,
                                    task_text,
                                )
                            machine.observe_success(True)
                            recent_error = NOTICE_SUCCESS
                        elif evaluation.failure_mode == "timeout":
                            writer.discard_episode()
                            machine.discard()
                            snapshot = env.reset(scene_seed)
                            controller.reset()
                            physics_accumulator = 0.0
                            recent_error = NOTICE_TIMEOUT
                        else:
                            delta = read_teleop_delta(viewer, controller.gripper)
                            action, ik_error = controller.command(delta)
                            if ik_error:
                                recent_error = ik_error
                            elif delta.meaningful:
                                recent_error = ""
                            should_record = machine.observe_action(delta.meaningful)
                            if should_record:
                                writer.add_frame(
                                    viewer.capture_training_images(256),
                                    env.get_state(),
                                    action,
                                    snapshot,
                                    task_text,
                                )
                            env.apply_joint_action(action)

                    if machine.phase != CollectionPhase.PENDING_CONFIRMATION:
                        physics_accumulator += 1.0 / DISPLAY_HZ
                        physics_steps = int(physics_accumulator / env.model.opt.timestep)
                        if physics_steps:
                            env.step(physics_steps)
                            physics_accumulator -= physics_steps * env.model.opt.timestep

                    viewer.set_status(
                        "SmolVLA Collector",
                        _status_text(
                            machine.phase,
                            task_text,
                            scene_seed,
                            machine.frame_count,
                            confirmed_count,
                            args.episodes,
                            recent_error,
                        ),
                    )
                    viewer.render()
                    display_frame += 1
                    remaining = 1.0 / DISPLAY_HZ - (time.perf_counter() - frame_start)
                    if remaining > 0:
                        time.sleep(remaining)
    finally:
        if machine.phase != CollectionPhase.IDLE:
            writer.discard_episode()
        writer.close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """解析参数并启动采集。

    Args:
        argv: 可选命令行参数；为空时读取进程参数。

    Returns:
        采集入口退出码。
    """
    return run_collection(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
