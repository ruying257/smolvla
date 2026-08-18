"""杯子V3专用LeRobot schema、原子episode写入和分片验签。"""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow.parquet as pq
from numpy.typing import NDArray

from collector.v3.collection_plan import MUG_DATASET_VERSION, MUG_REPO_ID
from collector.v3.task_spec import TASK_IDS, TASK_PROMPTS
from sim.mug_environment import MUG_SAMPLE_X_RANGE, MUG_SAMPLE_Y_RANGE, MugSceneSnapshot


DATASET_FPS = 20
CONTRACT_FILENAME = "collector_contract.json"
CAMERA_FEATURES = {
    "observation.images.agent": "agentview",
    "observation.images.wrist": "d435i_rgb",
}


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
    """使用UTF-8 concat清单合并中文路径下的视频。

    Args:
        input_video_paths: 按时间顺序排列的输入MP4。
        output_video_path: 合并后的目标MP4。
        overwrite: 目标存在时是否允许覆盖。

    Raises:
        FileExistsError: 目标存在且禁止覆盖时抛出。
        FileNotFoundError: 输入为空或任一输入文件不存在时抛出。
    """
    output_path = Path(output_video_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"视频文件已存在: {output_path}")
    resolved_inputs = [Path(path).resolve() for path in input_video_paths]
    if not resolved_inputs or any(not path.is_file() for path in resolved_inputs):
        raise FileNotFoundError(f"V3待合并视频缺失: {resolved_inputs}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    concat_path: Path | None = None
    temporary_output: Path | None = None
    input_container = None
    output_container = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", suffix=".ffconcat", delete=False,
        ) as handle:
            handle.write("ffconcat version 1.0\n")
            for input_path in resolved_inputs:
                handle.write(f"file '{_quote_ffconcat_path(input_path)}'\n")
            concat_path = Path(handle.name)
        input_container = av.open(str(concat_path), mode="r", format="concat", options={"safe": "0"})
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
            temporary_output = Path(handle.name)
        output_container = av.open(str(temporary_output), mode="w", options={"movflags": "faststart"})
        stream_map = {}
        for input_stream in input_container.streams:
            if input_stream.type in ("video", "audio", "subtitle"):
                output_stream = output_container.add_stream_from_template(input_stream, opaque=True)
                output_stream.time_base = input_stream.time_base
                stream_map[input_stream.index] = output_stream
        for packet in input_container.demux():
            if packet.stream.index in stream_map and packet.dts is not None:
                packet.stream = stream_map[packet.stream.index]
                output_container.mux(packet)
        input_container.close()
        input_container = None
        output_container.close()
        output_container = None
        shutil.move(str(temporary_output), str(output_path))
        temporary_output = None
    finally:
        if input_container is not None:
            input_container.close()
        if output_container is not None:
            output_container.close()
        if concat_path is not None:
            concat_path.unlink(missing_ok=True)
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)


def find_ffmpeg() -> Path | None:
    """查找环境已有的系统或imageio-ffmpeg可执行文件。

    Returns:
        可用FFmpeg绝对路径；当前固定环境完全不可用时返回 ``None``。
    """
    system_executable = shutil.which("ffmpeg")
    if system_executable:
        return Path(system_executable).resolve()
    try:
        import imageio_ffmpeg

        bundled = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return None
    return bundled.resolve() if bundled.is_file() else None


def _accessible_mkdtemp(
    suffix: str | None = None,
    prefix: str | None = None,
    dir: str | Path | None = None,
) -> str:
    """创建继承父目录权限的LeRobot视频临时目录。

    Args:
        suffix: 可选目录名后缀。
        prefix: 可选目录名前缀。
        dir: 可选父目录；为空时使用系统临时目录。

    Returns:
        新建临时目录的字符串路径。
    """
    parent = Path(dir) if dir is not None else Path(tempfile.gettempdir())
    name = f"{prefix or '.lerobot-encode-'}{uuid.uuid4().hex}{suffix or ''}"
    directory = parent / name
    directory.mkdir(parents=False, exist_ok=False)
    return str(directory)


def configure_hf_datasets_cache(cache_root: Path) -> Path:
    """把Hugging Face Datasets缓存固定到项目可写目录。

    Args:
        cache_root: V3回放专用缓存目录。

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
    """返回杯子V3唯一允许的LeRobot feature schema。

    Returns:
        可直接传给 ``LeRobotDataset.create`` 的feature定义。视频保持
        ``agent``和``wrist``训练键，同时在契约中记录其真实MuJoCo来源。
    """
    state_names = [
        "shoulder_pan", "shoulder_lift", "elbow", "wrist_1",
        "wrist_2", "wrist_3", "gripper",
    ]
    action_names = [
        "shoulder_pan_target", "shoulder_lift_target", "elbow_target",
        "wrist_1_target", "wrist_2_target", "wrist_3_target", "gripper_command",
    ]
    return {
        "observation.images.agent": {
            "dtype": "video", "shape": (256, 256, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "video", "shape": (256, 256, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {"dtype": "float32", "shape": (7,), "names": state_names},
        "action": {"dtype": "float32", "shape": (7,), "names": action_names},
        "scene_seed": {"dtype": "int64", "shape": (1,), "names": ["scene_seed"]},
        "mug_initial_pose": {
            "dtype": "float32", "shape": (7,),
            "names": ["x", "y", "z", "qw", "qx", "qy", "qz"],
        },
    }


def json_schema() -> dict[str, dict[str, Any]]:
    """生成适合稳定JSON比较的V3 schema。

    Returns:
        所有tuple shape已转换为列表的feature映射。
    """
    return {
        key: {**value, "shape": list(value["shape"])}
        for key, value in dataset_features().items()
    }


def expected_contract(extras: dict[str, Any] | None = None) -> dict[str, Any]:
    """构造杯子V3不可漂移的数据集契约。

    Args:
        extras: 单分片队列键、配置哈希等附加不可变字段。

    Returns:
        不含动态episode记录的完整契约。

    Raises:
        ValueError: 附加字段试图覆盖基础V3契约时抛出。
    """
    contract: dict[str, Any] = {
        "dataset_version": MUG_DATASET_VERSION,
        "repo_id": MUG_REPO_ID,
        "fps": DATASET_FPS,
        "lerobot_version": importlib.metadata.version("lerobot"),
        "features": json_schema(),
        "camera_features": CAMERA_FEATURES,
        "tasks": list(TASK_IDS),
        "canonical_prompts": TASK_PROMPTS,
        "prompt_mode": "canonical_only",
        "mug_pose_order": ["x", "y", "z", "qw", "qx", "qy", "qz"],
        "mug_sample_x_range": list(MUG_SAMPLE_X_RANGE),
        "mug_sample_y_range": list(MUG_SAMPLE_Y_RANGE),
        "fixed_pad_positions": {
            "blue": [0.55, -0.22, 0.8005],
            "yellow": [0.55, 0.22, 0.8005],
        },
    }
    if extras:
        overlap = set(contract).intersection(extras)
        if overlap:
            raise ValueError(f"V3附加契约不得覆盖基础字段: {sorted(overlap)}")
        contract.update(extras)
    return contract


def _validate_vector(name: str, value: Any, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
    """把输入转换为指定dtype并验证shape和有限性。

    Args:
        name: 错误消息中的字段名称。
        value: NumPy兼容输入。
        shape: 期望shape。
        dtype: 目标NumPy dtype。

    Returns:
        已转换且验证通过的数组。

    Raises:
        ValueError: shape不符或包含NaN、Inf时抛出。
    """
    array = np.asarray(value, dtype=dtype)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name}必须是有限{shape}向量，实际为{array.shape}")
    return array


class MugEpisodeWriter:
    """安全写入一个或多个杯子V3 LeRobot episode。

    采集矩阵为每个队列键创建独立单episode分片；只有视频编码、Parquet和
    契约全部完成后，上层才会把该键原子登记到progress。构造函数允许注入
    in-memory dataset工厂，供不依赖真实录制的恢复与原子性测试使用。
    """

    def __init__(
        self,
        root: Path,
        *,
        contract_extras: dict[str, Any] | None = None,
        dataset_factory: Any | None = None,
    ) -> None:
        """创建一个全新的杯子V3数据分片。

        Args:
            root: 分片或最终数据集根目录。
            contract_extras: 配置哈希、队列键等附加契约。
            dataset_factory: 测试专用工厂；为空时创建真实LeRobotDataset。

        Raises:
            FileExistsError: 目标路径已经存在时抛出，防止覆盖。
            RuntimeError: 真实模式缺少FFmpeg或LeRobot不可导入时抛出。
        """
        self.root = root.resolve()
        if self.root.exists():
            raise FileExistsError(f"V3分片目录已存在，拒绝覆盖: {self.root}")
        self.contract_path = self.root / "meta" / CONTRACT_FILENAME
        self.contract = {**expected_contract(contract_extras), "episodes": []}
        if dataset_factory is None:
            if find_ffmpeg() is None:
                raise RuntimeError("找不到FFmpeg；请使用smolvla-collector-clean现有依赖")
            try:
                from lerobot.datasets.lerobot_dataset import LeRobotDataset
            except Exception as exc:
                raise RuntimeError(f"LeRobot导入失败: {exc}") from exc
            self.dataset = LeRobotDataset.create(
                repo_id=MUG_REPO_ID,
                root=self.root,
                robot_type="ur10e_mujoco_mug",
                fps=DATASET_FPS,
                features=dataset_features(),
                use_videos=True,
                image_writer_threads=4,
                image_writer_processes=0,
                video_backend="pyav",
                vcodec="h264",
            )
        else:
            self.dataset = dataset_factory(self.root, dataset_features())
        self._write_contract()

    @property
    def total_episodes(self) -> int:
        """返回该写入器已确认保存的episode数。

        Returns:
            契约动态记录长度。
        """
        return len(self.contract["episodes"])

    def add_frame(
        self,
        images: dict[str, NDArray[np.uint8]],
        state: NDArray[np.float32],
        action: NDArray[np.float32],
        snapshot: MugSceneSnapshot,
        task_text: str,
    ) -> None:
        """向当前episode添加一帧双相机、状态和绝对动作。

        Args:
            images: 键严格为 ``agent``与``wrist``的RGB图像。
            state: 七维绝对关节/夹爪状态。
            action: 七维绝对关节目标/夹爪指令。
            snapshot: 当前episode的稳定杯子初始快照。
            task_text: 唯一canonical英文指令。

        Raises:
            ValueError: 相机、图像、向量、seed或任务文本违反V3 schema时抛出。
        """
        if set(images) != {"agent", "wrist"}:
            raise ValueError(f"V3相机键必须是agent和wrist，实际为{set(images)}")
        checked_images: dict[str, np.ndarray] = {}
        for name, image in images.items():
            array = np.asarray(image)
            if array.shape != (256, 256, 3) or array.dtype != np.uint8:
                raise ValueError(f"{name}必须为256x256x3 uint8，实际为{array.shape}/{array.dtype}")
            checked_images[name] = array
        state_array = _validate_vector("observation.state", state, (7,), np.float32)
        action_array = _validate_vector("action", action, (7,), np.float32)
        if not 0.0 <= float(action_array[6]) <= 1.0:
            raise ValueError("夹爪动作必须位于[0,1]")
        pose = _validate_vector("mug_initial_pose", snapshot.mug_initial_pose, (7,), np.float32)
        if task_text not in TASK_PROMPTS.values():
            raise ValueError(f"V3不允许写入非canonical文本: {task_text!r}")
        self.dataset.add_frame({
            "observation.images.agent": checked_images["agent"],
            "observation.images.wrist": checked_images["wrist"],
            "observation.state": state_array,
            "action": action_array,
            "scene_seed": np.asarray([snapshot.scene_seed], dtype=np.int64),
            "mug_initial_pose": pose,
            "task": task_text,
        })

    def save_episode(
        self,
        task_id: str,
        task_text: str,
        snapshot: MugSceneSnapshot,
        frame_count: int,
    ) -> int:
        """校验缓冲、编码视频并提交一个确认episode。

        Args:
            task_id: ``mug_on_blue``或``mug_on_yellow``。
            task_text: 与任务严格对应的canonical文本。
            snapshot: 当前episode初始杯子快照。
            frame_count: 状态机统计的帧数，必须为1至400。

        Returns:
            本分片内从0开始的episode索引。

        Raises:
            ValueError: 任务、文本或帧数不符合契约时抛出。
            RuntimeError: 内存帧数与相机PNG帧数不一致时抛出。
        """
        if task_id not in TASK_PROMPTS or TASK_PROMPTS[task_id] != task_text:
            raise ValueError("V3任务标识与canonical文本不匹配")
        if not 1 <= frame_count <= 400:
            raise ValueError(f"V3 episode帧数必须位于[1,400]，实际为{frame_count}")
        self._validate_pending_episode(frame_count)
        episode_index = self.total_episodes
        if self.dataset.__class__.__module__.startswith("lerobot"):
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
                for directory in self.root.glob(".lerobot-encode-*"):
                    shutil.rmtree(directory, ignore_errors=True)
        else:
            self.dataset.save_episode(parallel_encoding=False)
        self.contract["episodes"].append({
            "episode_index": episode_index,
            "scene_seed": int(snapshot.scene_seed),
            "task_id": task_id,
            "prompt_mode": "canonical",
            "task": task_text,
            "frame_count": int(frame_count),
            "mug_initial_pose": np.asarray(snapshot.mug_initial_pose).tolist(),
        })
        self._write_contract()
        return episode_index

    def _validate_pending_episode(self, expected_frame_count: int) -> None:
        """在编码前验证状态缓冲和双相机临时帧完全对齐。

        Args:
            expected_frame_count: 状态机记录的当前episode帧数。

        Raises:
            RuntimeError: 缓冲大小或任一路相机帧数不一致时抛出。
        """
        if not hasattr(self.dataset, "episode_buffer"):
            return
        if hasattr(self.dataset, "_wait_image_writer"):
            self.dataset._wait_image_writer()
        buffered = int(self.dataset.episode_buffer["size"])
        if buffered != expected_frame_count:
            raise RuntimeError(
                f"V3保存前缓冲帧数不一致: expected={expected_frame_count}, actual={buffered}"
            )
        if not hasattr(self.dataset, "meta") or not hasattr(self.dataset, "_get_image_file_dir"):
            return
        episode_index = self.dataset.episode_buffer["episode_index"]
        for video_key in self.dataset.meta.video_keys:
            directory = self.dataset._get_image_file_dir(episode_index, video_key)
            actual = len(list(directory.glob("frame-*.png")))
            if actual != expected_frame_count:
                raise RuntimeError(
                    f"V3保存前相机帧数不一致: camera={video_key}, "
                    f"expected={expected_frame_count}, actual={actual}"
                )

    def discard_episode(self) -> None:
        """彻底清除未确认episode的内存与双路临时PNG。

        关闭Viewer、Ctrl+C、Z、Backspace和超时都会调用本方法，保证不会
        遗留可被误认为完成分片的半条episode。
        """
        if hasattr(self.dataset, "_wait_image_writer"):
            self.dataset._wait_image_writer()
        if hasattr(self.dataset, "episode_buffer") and hasattr(self.dataset, "meta"):
            episode_index = self.dataset.episode_buffer["episode_index"]
            for video_key in self.dataset.meta.video_keys:
                directory = self.dataset._get_image_file_dir(episode_index, video_key)
                if directory.is_dir():
                    shutil.rmtree(directory)
        if hasattr(self.dataset, "clear_episode_buffer"):
            self.dataset.clear_episode_buffer(delete_images=True)

    def close(self) -> None:
        """关闭LeRobot或测试替身的写入资源。"""
        if hasattr(self.dataset, "finalize"):
            self.dataset.finalize()

    def _write_contract(self) -> None:
        """使用临时文件原子更新V3 collector契约。"""
        self.contract_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.contract_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.contract, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.contract_path)


def read_shard_table(shard: Path) -> Any:
    """读取单episode分片唯一Parquet表。

    Args:
        shard: 分片根目录。

    Returns:
        PyArrow Table对象。

    Raises:
        ValueError: 数据Parquet数量不是1时抛出。
    """
    paths = list((shard / "data").glob("chunk-*/*.parquet"))
    if len(paths) != 1:
        raise ValueError(f"V3分片Parquet数量必须为1: {shard}")
    return pq.read_table(paths[0])


def vector_column(table: Any, name: str, dtype: Any = np.float32) -> np.ndarray:
    """把Arrow定长列表列转换成二维NumPy数组。

    Args:
        table: 包含目标列的PyArrow表。
        name: feature列名称。
        dtype: 输出NumPy dtype。

    Returns:
        帧为第一维的NumPy数组。
    """
    return np.asarray(table[name].to_pylist(), dtype=dtype)


def task_texts(shard: Path) -> list[str]:
    """读取LeRobot原生tasks元数据中的任务文本。

    Args:
        shard: 单episode分片根目录。

    Returns:
        原生任务索引中保存的所有文本。

    Raises:
        ValueError: tasks.parquet缺少文本索引列时抛出。
    """
    table = pq.read_table(shard / "meta" / "tasks.parquet")
    column = "__index_level_0__"
    if column not in table.column_names:
        raise ValueError(f"V3 tasks.parquet缺少文本列: {shard}")
    return [str(value) for value in table[column].to_pylist()]


def decode_video(path: Path) -> list[np.ndarray]:
    """完整解码一个H.264视频并返回RGB帧。

    Args:
        path: MP4视频路径。

    Returns:
        按时间顺序排列的RGB uint8帧。

    Raises:
        ValueError: 视频无法解码、为空或分辨率错误时抛出。
    """
    try:
        with av.open(str(path)) as container:
            frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    except Exception as exc:
        raise ValueError(f"V3视频无法解码: {path}: {exc}") from exc
    if not frames or any(frame.shape != (256, 256, 3) for frame in frames):
        raise ValueError(f"V3视频为空或尺寸错误: {path}")
    return frames


def validate_episode_shard(shard: Path, record: dict[str, Any]) -> None:
    """验证单episode分片的契约、Parquet、任务和双路视频。

    Args:
        shard: 活动分片目录。
        record: 即将登记或恢复时读取的完成记录。

    Raises:
        ValueError: 任一文件、身份、字段或帧数不完整时抛出。
    """
    contract_path = shard / "meta" / CONTRACT_FILENAME
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"V3分片契约缺失或损坏: {shard}") from exc
    if contract.get("repo_id") != MUG_REPO_ID or contract.get("dataset_version") != MUG_DATASET_VERSION:
        raise ValueError("V3分片数据身份错误")
    if contract.get("features") != json_schema() or contract.get("camera_features") != CAMERA_FEATURES:
        raise ValueError("V3分片schema或相机契约漂移")
    episodes = contract.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 1:
        raise ValueError("V3活动分片必须恰好包含1条episode")
    episode = episodes[0]
    expected_count = int(record["frame_count"])
    if contract.get("queue_key") != record.get("queue_key"):
        raise ValueError("V3分片queue_key与完成记录不一致")
    if contract.get("config_sha256") != record.get("config_sha256"):
        raise ValueError("V3分片config_sha256与完成记录不一致")
    if (
        episode.get("frame_count") != expected_count
        or episode.get("task") != record.get("task")
        or episode.get("task_id") != record.get("task_id")
        or episode.get("scene_seed") != record.get("scene_seed")
        or episode.get("mug_initial_pose") != record.get("mug_initial_pose")
    ):
        raise ValueError("V3分片episode契约与完成记录不一致")
    table = read_shard_table(shard)
    if len(table) != expected_count:
        raise ValueError(f"V3 Parquet帧数不一致: expected={expected_count}, actual={len(table)}")
    required = {"observation.state", "action", "scene_seed", "mug_initial_pose"}
    if not required.issubset(table.column_names):
        raise ValueError(f"V3 Parquet缺少字段: {sorted(required-set(table.column_names))}")
    states = vector_column(table, "observation.state")
    actions = vector_column(table, "action")
    poses = vector_column(table, "mug_initial_pose")
    if states.shape != (expected_count, 7) or actions.shape != (expected_count, 7):
        raise ValueError("V3 state/action shape错误")
    if poses.shape != (expected_count, 7) or not np.isfinite(states).all() or not np.isfinite(actions).all() or not np.isfinite(poses).all():
        raise ValueError("V3 state/action/mug_initial_pose含非有限值或shape错误")
    closed = np.flatnonzero(actions[:, 6] >= 0.5)
    if not len(closed) or not np.any(actions[closed[0] + 1 :, 6] < 0.5):
        raise ValueError("V3分片缺少夹爪闭合后释放动作")
    seeds = {
        int(np.asarray(value).reshape(-1)[0])
        for value in table["scene_seed"].to_pylist()
    }
    if seeds != {int(record["scene_seed"])}:
        raise ValueError("V3分片scene_seed列与完成记录不一致")
    expected_pose = np.asarray(record["mug_initial_pose"], dtype=np.float32)
    if not np.array_equal(poses, np.repeat(expected_pose[None], expected_count, axis=0)):
        raise ValueError("V3分片mug_initial_pose列与完成记录不一致")
    if set(task_texts(shard)) != {record["task"]}:
        raise ValueError("V3分片任务文本不是唯一canonical")
    for feature in CAMERA_FEATURES:
        paths = list((shard / "videos" / feature).glob("chunk-*/*.mp4"))
        if len(paths) != 1 or len(decode_video(paths[0])) != expected_count:
            raise ValueError(f"V3视频缺失或帧数不一致: {feature}")
