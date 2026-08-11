"""LeRobot v3数据写入、契约校验和显式续采封装。"""

from __future__ import annotations

import importlib.metadata
import json
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from collector.task_spec import choose_balanced_template
from sim.environment import (
    CUBE_MIN_CENTER_DISTANCE,
    CUBE_SAMPLE_X_RANGE,
    CUBE_SAMPLE_Y_RANGE,
    TASK_INITIAL_BODY_POSITIONS,
    SceneSnapshot,
)


DATASET_VERSION = "smolvla_ur10e_v1"
DATASET_REPO_ID = "smolvla_ur10e"
DATASET_FPS = 20
CONTRACT_FILENAME = "collector_contract.json"


def find_ffmpeg() -> Path | None:
    """查找系统或imageio-ffmpeg提供的可执行文件。

    Returns:
        可用FFmpeg绝对路径；完全不可用时返回 ``None``。
    """
    system_executable = shutil.which("ffmpeg")
    if system_executable:
        return Path(system_executable).resolve()
    try:
        import imageio_ffmpeg

        bundled_executable = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return None
    return bundled_executable.resolve() if bundled_executable.is_file() else None


def _accessible_mkdtemp(
    suffix: str | None = None,
    prefix: str | None = None,
    dir: str | Path | None = None,
) -> str:
    """创建继承父目录权限的视频编码临时目录。

    Windows上的部分受管Python环境会让标准 ``tempfile.mkdtemp`` 创建出
    当前进程无法再次访问的目录。本函数使用普通目录创建语义规避该问题。

    Args:
        suffix: 可选目录名后缀。
        prefix: 可选目录名前缀。
        dir: 父目录；为空时使用系统临时目录。

    Returns:
        新建临时目录的字符串路径。
    """
    parent = Path(dir) if dir is not None else Path(tempfile.gettempdir())
    directory_name = f"{prefix or '.lerobot-encode-'}{uuid.uuid4().hex}{suffix or ''}"
    temporary_directory = parent / directory_name
    temporary_directory.mkdir(parents=False, exist_ok=False)
    return str(temporary_directory)


def dataset_features() -> dict[str, dict[str, Any]]:
    """返回训练数据集的唯一feature schema。

    Returns:
        可直接传给 ``LeRobotDataset.create`` 的feature定义。
    """
    return {
        "observation.images.agent": {
            "dtype": "video",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": [
                "shoulder_pan",
                "shoulder_lift",
                "elbow",
                "wrist_1",
                "wrist_2",
                "wrist_3",
                "gripper",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": [
                "shoulder_pan_target",
                "shoulder_lift_target",
                "elbow_target",
                "wrist_1_target",
                "wrist_2_target",
                "wrist_3_target",
                "gripper_command",
            ],
        },
        "scene_seed": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["scene_seed"],
        },
        "cube_initial_poses": {
            "dtype": "float32",
            "shape": (14,),
            "names": [
                "red_x", "red_y", "red_z", "red_qw", "red_qx", "red_qy", "red_qz",
                "green_x", "green_y", "green_z", "green_qw", "green_qx", "green_qy", "green_qz",
            ],
        },
    }


def _json_schema() -> dict[str, dict[str, Any]]:
    """把tuple shape转换为稳定的JSON schema。

    Returns:
        适合序列化和严格比较的feature定义。
    """
    return {
        key: {**value, "shape": list(value["shape"])}
        for key, value in dataset_features().items()
    }


def expected_contract() -> dict[str, Any]:
    """构造本版本数据集不可漂移的契约字段。

    Returns:
        不包含动态episode记录的契约字典。
    """
    pad_positions = dict(TASK_INITIAL_BODY_POSITIONS)
    return {
        "dataset_version": DATASET_VERSION,
        "repo_id": DATASET_REPO_ID,
        "fps": DATASET_FPS,
        "lerobot_version": importlib.metadata.version("lerobot"),
        "features": _json_schema(),
        "cube_order": ["red", "green"],
        "cube_pose_order": ["x", "y", "z", "qw", "qx", "qy", "qz"],
        "cube_sample_x_range": list(CUBE_SAMPLE_X_RANGE),
        "cube_sample_y_range": list(CUBE_SAMPLE_Y_RANGE),
        "cube_min_center_distance": CUBE_MIN_CENTER_DISTANCE,
        "fixed_pad_positions": {
            "blue": list(pad_positions["task_blue_pad"]),
            "yellow": list(pad_positions["task_yellow_pad"]),
        },
    }


def contract_mismatches(actual: dict[str, Any]) -> list[str]:
    """比较已有契约与当前版本的不可漂移字段。

    Args:
        actual: 从已有数据集读取的完整契约。

    Returns:
        所有不一致或缺失的字段名。
    """
    expected = expected_contract()
    return [key for key, value in expected.items() if actual.get(key) != value]


class LeRobotEpisodeWriter:
    """安全创建、续采和保存SmolVLA专家episode。"""

    def __init__(self, root: Path, resume: bool = False) -> None:
        """创建新数据集或严格校验后续采。

        Args:
            root: LeRobot数据集根目录。
            resume: 是否允许在已有目录上追加。

        Raises:
            FileExistsError: 目录已存在但未显式续采时抛出。
            RuntimeError: FFmpeg或LeRobot依赖不可用时抛出。
            ValueError: 已有数据契约与当前版本不一致时抛出。
        """
        if find_ffmpeg() is None:
            raise RuntimeError("找不到FFmpeg；请安装conda-forge ffmpeg或imageio-ffmpeg")
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except Exception as exc:
            raise RuntimeError(f"LeRobot导入失败，请检查采集环境: {exc}") from exc

        self.root = root.resolve()
        self.contract_path = self.root / "meta" / CONTRACT_FILENAME
        self.contract = expected_contract()
        if self.root.exists():
            if not resume:
                raise FileExistsError(f"数据目录已存在；如需续采请显式添加 --resume: {self.root}")
            self._load_and_validate_contract()
            if self._is_recoverable_empty_dataset():
                shutil.rmtree(self.root)
                self.dataset = self._create_dataset(LeRobotDataset)
                self.contract = {**expected_contract(), "episodes": []}
                self._write_contract()
            else:
                self.dataset = LeRobotDataset(
                    DATASET_REPO_ID,
                    root=self.root,
                    video_backend="pyav",
                    vcodec="h264",
                )
                if int(self.dataset.meta.total_episodes) != self.total_episodes:
                    raise ValueError(
                        "续采数据不完整: "
                        f"LeRobot episodes={self.dataset.meta.total_episodes}, "
                        f"collector contract episodes={self.total_episodes}"
                    )
        else:
            self.dataset = self._create_dataset(LeRobotDataset)
            self.contract["episodes"] = []
            self._write_contract()

    def _create_dataset(self, dataset_class: Any) -> Any:
        """使用唯一参数集创建新的LeRobot数据集。

        Args:
            dataset_class: 已成功导入的 ``LeRobotDataset`` 类。

        Returns:
            可写入episode的新数据集实例。
        """
        return dataset_class.create(
            repo_id=DATASET_REPO_ID,
            root=self.root,
            robot_type="ur10e_mujoco",
            fps=DATASET_FPS,
            features=dataset_features(),
            use_videos=True,
            image_writer_threads=4,
            image_writer_processes=0,
            video_backend="pyav",
            vcodec="h264",
        )

    def _is_recoverable_empty_dataset(self) -> bool:
        """判断已有目录是否只是可安全重建的空初始化结果。

        Returns:
            契约为0条episode且不存在帧、视频或未知文件时返回 ``True``。
        """
        if self.total_episodes != 0:
            return False
        allowed_files = {
            (Path("meta") / CONTRACT_FILENAME).as_posix(),
            (Path("meta") / "info.json").as_posix(),
        }
        actual_files = {
            path.relative_to(self.root).as_posix()
            for path in self.root.rglob("*")
            if path.is_file()
        }
        return actual_files.issubset(allowed_files)

    @property
    def total_episodes(self) -> int:
        """返回已经确认保存的episode数量。"""
        return len(self.contract.get("episodes", []))

    def template_counts(self) -> dict[str, int]:
        """统计每类任务两种训练措辞的已保存数量。

        Returns:
            键格式为 ``task_id/template_id`` 的计数字典。
        """
        counts: dict[str, int] = {}
        for episode in self.contract.get("episodes", []):
            key = f"{episode['task_id']}/{episode['template_id']}"
            counts[key] = counts.get(key, 0) + 1
        return counts

    def next_template(self, task_id: str) -> str:
        """返回指定任务下一条episode应使用的均衡模板。

        Args:
            task_id: 四类固定任务之一。

        Returns:
            ``canonical`` 或 ``synonym``。
        """
        return choose_balanced_template(task_id, self.template_counts())

    def add_frame(
        self,
        images: dict[str, NDArray[np.uint8]],
        state: NDArray[np.float32],
        action: NDArray[np.float32],
        snapshot: SceneSnapshot,
        task_text: str,
    ) -> None:
        """向当前episode缓冲区添加一个20 Hz帧。

        Args:
            images: ``agent``和``wrist``两路RGB图像。
            state: 七维当前状态。
            action: 七维绝对目标动作。
            snapshot: 本episode的初始场景快照。
            task_text: 当前英文训练指令。
        """
        if set(images) != {"agent", "wrist"}:
            raise ValueError(f"相机键必须为agent和wrist，实际为 {set(images)}")
        for name, image in images.items():
            if image.shape != (256, 256, 3) or image.dtype != np.uint8:
                raise ValueError(f"{name}图像必须是256x256x3 uint8，实际为 {image.shape}/{image.dtype}")
        self.dataset.add_frame(
            {
                "observation.images.agent": images["agent"],
                "observation.images.wrist": images["wrist"],
                "observation.state": np.asarray(state, dtype=np.float32),
                "action": np.asarray(action, dtype=np.float32),
                "scene_seed": np.asarray([snapshot.scene_seed], dtype=np.int64),
                "cube_initial_poses": snapshot.cube_initial_poses.astype(np.float32).reshape(14),
                "task": task_text,
            }
        )

    def save_episode(
        self,
        task_id: str,
        template_id: str,
        task_text: str,
        snapshot: SceneSnapshot,
        frame_count: int,
    ) -> int:
        """编码视频、提交episode并更新采集契约。

        Args:
            task_id: 内部任务标识。
            template_id: 本episode训练措辞标识。
            task_text: 实际写入的英文指令。
            snapshot: 初始场景快照。
            frame_count: 本episode包含的帧数。

        Returns:
            新保存的episode编号。
        """
        episode_index = self.total_episodes
        # LeRobot 0.4.4内部直接调用tempfile.mkdtemp；短时替换为权限稳定的
        # 创建函数，避免Windows受管环境生成当前进程不可访问的临时目录。
        original_mkdtemp = tempfile.mkdtemp
        tempfile.mkdtemp = _accessible_mkdtemp
        try:
            self.dataset.save_episode(parallel_encoding=False)
        finally:
            tempfile.mkdtemp = original_mkdtemp
            for temporary_directory in self.root.glob(".lerobot-encode-*"):
                if temporary_directory.is_dir():
                    try:
                        temporary_directory.rmdir()
                    except OSError:
                        pass
        self.contract.setdefault("episodes", []).append(
            {
                "episode_index": episode_index,
                "scene_seed": snapshot.scene_seed,
                "task_id": task_id,
                "template_id": template_id,
                "task": task_text,
                "frame_count": frame_count,
                "cube_initial_poses": snapshot.cube_initial_poses.tolist(),
            }
        )
        self._write_contract()
        return episode_index

    def discard_episode(self) -> None:
        """删除当前未保存episode的图像和内存缓冲。"""
        self.dataset.clear_episode_buffer(delete_images=True)

    def close(self) -> None:
        """关闭Parquet写入器和视频编码资源。"""
        self.dataset.finalize()

    def _load_and_validate_contract(self) -> None:
        """读取已有契约并逐项校验所有不可漂移字段。"""
        if not self.contract_path.is_file():
            raise ValueError(f"续采目录缺少契约文件: {self.contract_path}")
        actual = json.loads(self.contract_path.read_text(encoding="utf-8"))
        mismatches = contract_mismatches(actual)
        if mismatches:
            raise ValueError(f"续采数据契约不匹配: {mismatches}")
        self.contract = actual

    def _write_contract(self) -> None:
        """使用临时文件原子更新采集契约。"""
        self.contract_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.contract_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(self.contract, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(self.contract_path)
