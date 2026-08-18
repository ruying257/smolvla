"""独立杯子双放置区MuJoCo场景的启动入口。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

import numpy as np

from sim.mug_environment import MugTabletopEnv


def build_parser() -> argparse.ArgumentParser:
    """创建杯子场景命令行参数解析器。

    Returns:
        已配置随机种子、GUI和Headless参数的解析器。
    """
    parser = argparse.ArgumentParser(description="展示独立的杯子双放置区MuJoCo场景")
    parser.add_argument("--scene-seed", type=int, default=0, help="杯子随机布局种子，默认0")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="不打开窗口，执行有限仿真步和相机检查后退出",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Headless模式在稳定reset后执行的物理步数，默认10",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="GUI最长展示秒数；默认持续到用户关闭窗口",
    )
    parser.add_argument(
        "--display-hz",
        type=float,
        default=60.0,
        help="MuJoCo主Viewer目标刷新频率，默认60 Hz",
    )
    parser.add_argument(
        "--no-camera-panel",
        action="store_true",
        help="只显示主视角，不显示MuJoCo窗口内三路固定相机",
    )
    return parser


def _json_value(value: object) -> object:
    """把NumPy数组转换成可序列化值。

    Args:
        value: 环境布局中的属性值。

    Returns:
        数组对应的列表，或未经修改的原值。
    """
    return value.tolist() if isinstance(value, np.ndarray) else value


def run_headless(steps: int, scene_seed: int = 0) -> dict[str, object]:
    """执行杯子场景无窗口仿真与相机检查。

    Args:
        steps: 稳定reset后需要推进的物理步数。
        scene_seed: 杯子随机布局种子。

    Returns:
        可序列化的模型、空间布局、杯子状态和相机摘要。
    """
    with MugTabletopEnv() as env:
        snapshot = env.reset(scene_seed)
        env.step(steps)
        images = env.capture_cameras()
        return {
            "steps": steps,
            "scene_seed": snapshot.scene_seed,
            "model": {
                "nq": env.model.nq,
                "nv": env.model.nv,
                "nu": env.model.nu,
                "ncam": env.model.ncam,
            },
            "layout": {
                key: _json_value(value)
                for key, value in env.spatial_layout().items()
            },
            "task_layout": {
                body_name: {
                    key: _json_value(value)
                    for key, value in properties.items()
                }
                for body_name, properties in env.task_layout().items()
            },
            "cameras": {
                name: {
                    "shape": list(image.shape),
                    "dtype": str(image.dtype),
                    "mean": float(image.mean()),
                }
                for name, image in images.items()
            },
        }


def main(argv: Sequence[str] | None = None) -> int:
    """解析参数并启动杯子GUI或Headless流程。

    Args:
        argv: 可选命令行参数；为空时读取当前进程参数。

    Returns:
        成功完成时返回0。
    """
    args = build_parser().parse_args(argv)
    if args.headless:
        print(json.dumps(run_headless(args.steps, args.scene_seed), ensure_ascii=False, indent=2))
        return 0

    with MugTabletopEnv() as env:
        env.reset(args.scene_seed)
        gui_stats = env.run(
            max_seconds=args.max_seconds,
            display_hz=args.display_hz,
            show_camera_panel=not args.no_camera_panel,
        )
    print(json.dumps({"gui": gui_stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
