"""LeRobot v3数据写入、契约校验和显式续采封装。"""

from __future__ import annotations

import importlib.metadata
import json
import os
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


def _quote_ffconcat_path(path: Path) -> str:
    """转义FFmpeg concat清单中的单引号。

    Args:
        path: 需要写入concat清单的绝对路径。

    Returns:
        可安全放入单引号字段的路径文本。
    """
    return str(path.resolve()).replace("'", "'\\''")


def concatenate_video_files_utf8(
    input_video_paths: list[Path | str],
    output_video_path: Path,
    overwrite: bool = True,
) -> None:
    """使用UTF-8 concat清单合并视频，兼容Windows中文路径。

    LeRobot 0.4.4使用系统默认编码写入临时 ``.ffconcat`` 文件，而
    FFmpeg按UTF-8读取该清单。在中文路径下，从第二个episode开始合并视频
    时会把路径解码成乱码。本函数保持原有remux语义，只显式锁定UTF-8。

    Args:
        input_video_paths: 按时间顺序排列的输入视频。
        output_video_path: 合并后的目标视频。
        overwrite: 目标存在时是否覆盖。

    Raises:
        FileExistsError: 目标存在且禁止覆盖时抛出。
        FileNotFoundError: 输入列表为空或输入文件不存在时抛出。
    """
    import av

    output_path = Path(output_video_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"视频文件已存在: {output_path}")
    if not input_video_paths:
        raise FileNotFoundError("没有可合并的输入视频")

    resolved_inputs = [Path(path).resolve() for path in input_video_paths]
    missing_inputs = [path for path in resolved_inputs if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"缺少待合并视频: {missing_inputs}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    concat_path: Path | None = None
    temporary_output_path: Path | None = None
    input_container = None
    output_container = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            suffix=".ffconcat",
            delete=False,
        ) as concat_file:
            concat_file.write("ffconcat version 1.0\n")
            for input_path in resolved_inputs:
                concat_file.write(f"file '{_quote_ffconcat_path(input_path)}'\n")
            concat_path = Path(concat_file.name)

        input_container = av.open(
            str(concat_path),
            mode="r",
            format="concat",
            options={"safe": "0"},
        )
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as output_file:
            temporary_output_path = Path(output_file.name)
        output_container = av.open(
            str(temporary_output_path),
            mode="w",
            options={"movflags": "faststart"},
        )

        stream_map = {}
        for input_stream in input_container.streams:
            if input_stream.type in ("video", "audio", "subtitle"):
                output_stream = output_container.add_stream_from_template(
                    template=input_stream,
                    opaque=True,
                )
                output_stream.time_base = input_stream.time_base
                stream_map[input_stream.index] = output_stream

        for packet in input_container.demux():
            if packet.stream.index not in stream_map or packet.dts is None:
                continue
            packet.stream = stream_map[packet.stream.index]
            output_container.mux(packet)

        input_container.close()
        input_container = None
        output_container.close()
        output_container = None
        shutil.move(str(temporary_output_path), str(output_path))
        temporary_output_path = None
    finally:
        if input_container is not None:
            input_container.close()
        if output_container is not None:
            output_container.close()
        if concat_path is not None:
            concat_path.unlink(missing_ok=True)
        if temporary_output_path is not None:
            temporary_output_path.unlink(missing_ok=True)


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


def configure_hf_datasets_cache(cache_root: Path) -> Path:
    """把Hugging Face Datasets缓存固定到项目可写目录。

    Windows受管环境可能拒绝在用户级 ``.cache`` 下创建Parquet锁文件，导致
    已经完整落盘的LeRobot数据无法重新读取。该函数同时设置环境变量和已经
    导入的 ``datasets.config``，保证当前进程后续读取使用项目缓存。

    Args:
        cache_root: 本任务专用缓存目录。

    Returns:
        已创建的缓存目录绝对路径。
    """
    resolved = cache_root.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = str(resolved)
    try:
        from datasets import config as datasets_config

        datasets_config.HF_DATASETS_CACHE = str(resolved)
    except ImportError:
        pass
    return resolved


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


def expected_contract(
    dataset_version: str = DATASET_VERSION,
    repo_id: str = DATASET_REPO_ID,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造本版本数据集不可漂移的契约字段。

    Args:
        dataset_version: 数据集版本标识，默认保持旧v1值。
        repo_id: LeRobot仓库身份，默认保持旧v1值。
        extras: 不得覆盖基础字段的独立扩展契约。

    Returns:
        不包含动态episode记录的契约字典。
    """
    pad_positions = dict(TASK_INITIAL_BODY_POSITIONS)
    contract = {
        "dataset_version": dataset_version,
        "repo_id": repo_id,
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
    if extras:
        overlap = set(contract).intersection(extras)
        if overlap:
            raise ValueError(f"扩展契约不得覆盖基础字段: {sorted(overlap)}")
        contract.update(extras)
    return contract


def contract_mismatches(
    actual: dict[str, Any],
    expected: dict[str, Any] | None = None,
) -> list[str]:
    """比较已有契约与当前版本的不可漂移字段。

    Args:
        actual: 从已有数据集读取的完整契约。
        expected: 可选的目标契约；为空时使用旧v1默认契约。

    Returns:
        所有不一致或缺失的字段名。
    """
    target = expected or expected_contract()
    return [key for key, value in target.items() if actual.get(key) != value]


class LeRobotEpisodeWriter:
    """安全创建、续采和保存SmolVLA专家episode。"""

    def __init__(
        self,
        root: Path,
        resume: bool = False,
        *,
        dataset_version: str = DATASET_VERSION,
        repo_id: str = DATASET_REPO_ID,
        contract_extras: dict[str, Any] | None = None,
    ) -> None:
        """创建新数据集或严格校验后续采。

        Args:
            root: LeRobot数据集根目录。
            resume: 是否允许在已有目录上追加。
            dataset_version: 当前写入器的数据集版本，默认保持旧v1值。
            repo_id: 当前写入器的LeRobot仓库身份，默认保持旧v1值。
            contract_extras: 队列键等不可覆盖基础字段的扩展契约。

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
        self.dataset_version = dataset_version
        self.repo_id = repo_id
        self.contract_extras = dict(contract_extras or {})
        self.contract_path = self.root / "meta" / CONTRACT_FILENAME
        self.expected_contract = expected_contract(dataset_version, repo_id, self.contract_extras)
        self.contract = dict(self.expected_contract)
        if self.root.exists():
            if not resume:
                raise FileExistsError(f"数据目录已存在；如需续采请显式添加 --resume: {self.root}")
            self._load_and_validate_contract()
            if self._is_recoverable_empty_dataset():
                shutil.rmtree(self.root)
                self.dataset = self._create_dataset(LeRobotDataset)
                self.contract = {**self.expected_contract, "episodes": []}
                self._write_contract()
            else:
                self.dataset = LeRobotDataset(
                    self.repo_id,
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
            repo_id=self.repo_id,
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
        self._validate_pending_episode(frame_count)
        # LeRobot 0.4.4内部直接调用tempfile.mkdtemp；短时替换为权限稳定的
        # 创建函数，避免Windows受管环境生成当前进程不可访问的临时目录。
        from lerobot.datasets import lerobot_dataset as lerobot_dataset_module

        original_mkdtemp = tempfile.mkdtemp
        original_concatenate = lerobot_dataset_module.concatenate_video_files
        tempfile.mkdtemp = _accessible_mkdtemp
        lerobot_dataset_module.concatenate_video_files = concatenate_video_files_utf8
        try:
            self.dataset.save_episode(parallel_encoding=False)
        finally:
            tempfile.mkdtemp = original_mkdtemp
            lerobot_dataset_module.concatenate_video_files = original_concatenate
            for temporary_directory in self.root.glob(".lerobot-encode-*"):
                if temporary_directory.is_dir():
                    shutil.rmtree(temporary_directory, ignore_errors=True)
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

    def _validate_pending_episode(self, expected_frame_count: int) -> None:
        """在落盘前验证状态缓冲和两路相机帧数一致。

        Args:
            expected_frame_count: 采集状态机记录的当前episode帧数。

        Raises:
            RuntimeError: 状态缓冲或任一路相机帧数不一致时抛出。
        """
        self.dataset._wait_image_writer()
        buffered_count = int(self.dataset.episode_buffer["size"])
        if buffered_count != expected_frame_count:
            raise RuntimeError(
                "保存前帧数不一致: "
                f"state_machine={expected_frame_count}, buffer={buffered_count}"
            )

        episode_index = self.dataset.episode_buffer["episode_index"]
        for video_key in self.dataset.meta.video_keys:
            image_directory = self.dataset._get_image_file_dir(episode_index, video_key)
            image_count = len(list(image_directory.glob("frame-*.png")))
            if image_count != expected_frame_count:
                raise RuntimeError(
                    "保存前相机帧数不一致: "
                    f"camera={video_key}, expected={expected_frame_count}, actual={image_count}"
                )

    def discard_episode(self) -> None:
        """删除当前未保存episode的两路视频帧和内存缓冲。

        LeRobot 0.4.4的 ``clear_episode_buffer(delete_images=True)`` 只遍历
        ``image_keys``，不会清理以 ``video`` feature暂存的PNG。本项目两路
        相机均为video feature，因此需要在重置缓冲前显式删除对应目录。
        """
        self.dataset._wait_image_writer()
        episode_index = self.dataset.episode_buffer["episode_index"]
        for video_key in self.dataset.meta.video_keys:
            image_directory = self.dataset._get_image_file_dir(episode_index, video_key)
            if image_directory.is_dir():
                shutil.rmtree(image_directory)
        self.dataset.clear_episode_buffer(delete_images=True)

    def close(self) -> None:
        """关闭Parquet写入器和视频编码资源。"""
        self.dataset.finalize()

    def _load_and_validate_contract(self) -> None:
        """读取已有契约并逐项校验所有不可漂移字段。"""
        if not self.contract_path.is_file():
            raise ValueError(f"续采目录缺少契约文件: {self.contract_path}")
        actual = json.loads(self.contract_path.read_text(encoding="utf-8"))
        mismatches = contract_mismatches(actual, self.expected_contract)
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
