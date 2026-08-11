"""ACT 双积木放置场景的标准启动入口。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

import numpy as np

from sim.environment import CleanTabletopEnv


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。

    Returns:
        已配置 GUI 和 Headless 参数的解析器。
    """
    parser = argparse.ArgumentParser(description="展示参考 ACT demo_scene 构建的双积木桌面场景")
    parser.add_argument(
        "--scene-seed",
        type=int,
        default=0,
        help="积木随机布局种子，默认0",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="不打开窗口，执行有限仿真步和相机渲染检查后退出",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Headless 模式执行的物理步数，默认10",
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
        help="MuJoCo主Viewer刷新频率，默认60 Hz",
    )
    parser.add_argument(
        "--no-camera-panel",
        action="store_true",
        help="只显示主视角，不显示MuJoCo窗口内的三路固定相机",
    )
    return parser


def run_headless(steps: int, scene_seed: int = 0) -> dict[str, object]:
    """执行无窗口仿真与相机检查。

    Args:
        steps: 需要推进的 MuJoCo 物理步数。
        scene_seed: 积木随机布局种子。

    Returns:
        可序列化的模型规模、空间布局和相机图像摘要。
    """
    with CleanTabletopEnv() as env:
        snapshot = env.reset(scene_seed)
        env.step(steps)
        images = env.capture_cameras()
        layout = env.spatial_layout()
        task_layout = env.task_layout()
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
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in layout.items()
            },
            "task_layout": {
                body_name: {
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in properties.items()
                }
                for body_name, properties in task_layout.items()
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
    """解析参数并启动 GUI 或 Headless 流程。

    Args:
        argv: 可选命令行参数；为空时读取当前进程参数。

    Returns:
        成功时返回0。
    """
    args = build_parser().parse_args(argv)
    if args.headless:
        print(json.dumps(run_headless(args.steps, args.scene_seed), ensure_ascii=False, indent=2))
        return 0

    with CleanTabletopEnv() as env:
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
