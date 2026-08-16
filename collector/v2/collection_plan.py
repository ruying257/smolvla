"""Grounding v2配置、Latin square队列与严格恢复状态。"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GROUNDING_REPO_ID = "smolvla_ur10e_grounding_v2"
GROUNDING_DATASET_VERSION = "smolvla_ur10e_grounding_v2"
PROMPT_MODE = "canonical"
MANIFEST_FILENAME = "collection_manifest.json"
QUEUE_FILENAME = "collection_queue.csv"
PROGRESS_FILENAME = "collection_progress.json"
REVIEW_FILENAME = "review_status.csv"
PILOT_VALIDATION_FILENAME = "pilot_validation.json"
INITIAL_REFERENCE_DIRNAME = "initial_references"


@dataclass(frozen=True)
class GroundingConfig:
    """保存已经校验的Grounding v2锁定配置。"""

    source_path: Path
    root: Path
    repo_id: str
    dataset_version: str
    scene_seeds: tuple[int, ...]
    tasks: tuple[str, ...]
    canonical_prompts: dict[str, str]
    latin_square: tuple[tuple[str, ...], ...]
    fps: int
    max_frames: int
    pilot_scene_count: int
    expected_total: int
    strict_success_required: bool
    montage_speed: float
    snapshot: dict[str, Any]
    sha256: str

    @property
    def work_root(self) -> Path:
        """返回与最终数据集隔离的可恢复采集工作目录。"""
        return self.root.parent / f".{self.root.name}_collection"

    @property
    def shard_root(self) -> Path:
        """返回按队列键存储单episode分片的目录。"""
        return self.work_root / "episode_shards"


@dataclass(frozen=True)
class QueueItem:
    """描述一个不可重复的scene-task-canonical采集组合。"""

    queue_index: int
    scene_index: int
    collection_position: int
    scene_seed: int
    task_id: str
    prompt_mode: str
    prompt: str
    queue_key: str

    @property
    def shard_name(self) -> str:
        """返回与Windows路径兼容的稳定分片目录名。"""
        digest = hashlib.sha256(self.queue_key.encode("utf-8")).hexdigest()[:12]
        return f"{self.queue_index:03d}_{self.scene_seed}_{self.task_id}_{digest}"


def utc_now() -> str:
    """返回ISO 8601格式的UTC时间。"""
    return datetime.now(timezone.utc).isoformat()


def stable_json_sha256(value: Any) -> str:
    """计算稳定JSON序列化结果的SHA-256。"""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_array(value: Any) -> str:
    """按dtype、shape和原始字节计算NumPy兼容数组哈希。"""
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def initial_reference_path(config: GroundingConfig, scene_seed: int) -> Path:
    """返回指定scene的无损初始观测基准文件路径。

    Args:
        config: 当前锁定配置。
        scene_seed: 场景随机种子。

    Returns:
        工作目录下的 ``.npz`` 文件路径。
    """
    return config.work_root / INITIAL_REFERENCE_DIRNAME / f"scene_{scene_seed}.npz"


def save_initial_reference(
    config: GroundingConfig,
    scene_seed: int,
    state: Any,
    agent_image: Any,
    wrist_image: Any,
    cube_initial_poses: Any,
) -> Path:
    """原子保存一个scene跨任务复用的无损初始观测。

    Args:
        config: 当前锁定配置。
        scene_seed: 场景随机种子。
        state: 七维初始机器人状态。
        agent_image: 第三方相机原始RGB图像。
        wrist_image: 腕部相机原始RGB图像。
        cube_initial_poses: 红绿积木的两组七维初始位姿。

    Returns:
        已写入的基准文件路径。

    Raises:
        FileExistsError: 同一scene的基准已经存在时抛出。
    """
    import numpy as np

    path = initial_reference_path(config, scene_seed)
    if path.exists():
        raise FileExistsError(f"初始观测基准已存在: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            scene_seed=np.asarray([scene_seed], dtype=np.int64),
            state=np.asarray(state, dtype=np.float32),
            agent=np.asarray(agent_image, dtype=np.uint8),
            wrist=np.asarray(wrist_image, dtype=np.uint8),
            cube_initial_poses=np.asarray(cube_initial_poses, dtype=np.float64),
        )
    temporary.replace(path)
    return path


def load_initial_reference(config: GroundingConfig, scene_seed: int) -> dict[str, Any]:
    """读取并验证一个scene的无损初始观测基准。

    Args:
        config: 当前锁定配置。
        scene_seed: 场景随机种子。

    Returns:
        包含状态、双相机图像、积木位姿和三项哈希的字典。

    Raises:
        ValueError: 文件缺失、字段错误、shape错误或内容损坏时抛出。
    """
    import numpy as np

    path = initial_reference_path(config, scene_seed)
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {"scene_seed", "state", "agent", "wrist", "cube_initial_poses"}
            if set(archive.files) != required:
                raise ValueError(f"字段错误: {archive.files}")
            stored_seed = int(np.asarray(archive["scene_seed"]).reshape(-1)[0])
            state = np.asarray(archive["state"], dtype=np.float32).copy()
            agent = np.asarray(archive["agent"], dtype=np.uint8).copy()
            wrist = np.asarray(archive["wrist"], dtype=np.uint8).copy()
            poses = np.asarray(archive["cube_initial_poses"], dtype=np.float64).copy()
    except (OSError, ValueError, KeyError, IndexError) as exc:
        raise ValueError(f"初始观测基准缺失或损坏: scene={scene_seed}, path={path}") from exc
    if stored_seed != scene_seed:
        raise ValueError(f"初始观测基准seed错误: expected={scene_seed}, actual={stored_seed}")
    shapes = {
        "state": state.shape == (7,),
        "agent": agent.shape == (256, 256, 3),
        "wrist": wrist.shape == (256, 256, 3),
        "cube_initial_poses": poses.shape == (2, 7),
    }
    if not all(shapes.values()):
        raise ValueError(f"初始观测基准shape错误: {shapes}")
    return {
        "state": state,
        "agent": agent,
        "wrist": wrist,
        "cube_initial_poses": poses,
        "initial_robot_state_sha256": hash_array(state),
        "initial_agent_raw_sha256": hash_array(agent),
        "initial_wrist_raw_sha256": hash_array(wrist),
    }


def validate_frame_count(frame_count: int, max_frames: int) -> None:
    """在任何视频编码前拒绝空episode或超过上限的episode。"""
    if not 1 <= frame_count <= max_frames:
        raise ValueError(f"episode帧数必须位于[1,{max_frames}]，实际为{frame_count}")


def _read_yaml(path: Path) -> dict[str, Any]:
    """读取YAML配置并确保顶层为映射。"""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("缺少PyYAML，请在采集环境安装requirements-collector.txt") from exc
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Grounding配置顶层必须是映射")
    return value


def load_config(path: Path) -> GroundingConfig:
    """加载并严格校验Grounding v2锁定配置。

    Args:
        path: YAML配置路径。

    Returns:
        规范化后的只读配置。

    Raises:
        ValueError: 数据身份、矩阵或采集约束发生漂移时抛出。
    """
    source_path = path.resolve()
    raw = _read_yaml(source_path)
    dataset = raw.get("dataset", {})
    collection = raw.get("collection", {})
    root_value = Path(str(dataset.get("root", "")))
    root = root_value if root_value.is_absolute() else source_path.parents[1] / root_value
    snapshot = json.loads(json.dumps(raw, ensure_ascii=False))
    config = GroundingConfig(
        source_path=source_path,
        root=root.resolve(),
        repo_id=str(dataset.get("repo_id", "")),
        dataset_version=str(dataset.get("version", "")),
        scene_seeds=tuple(int(value) for value in raw.get("scene_seeds", [])),
        tasks=tuple(str(value) for value in raw.get("tasks", [])),
        canonical_prompts={str(key): str(value) for key, value in raw.get("canonical_prompts", {}).items()},
        latin_square=tuple(tuple(str(task) for task in row) for row in raw.get("latin_square", [])),
        fps=int(collection.get("fps", 0)),
        max_frames=int(collection.get("max_frames", 0)),
        pilot_scene_count=int(collection.get("pilot_scene_count", 0)),
        expected_total=int(collection.get("expected_total", 0)),
        strict_success_required=bool(collection.get("strict_success_required", False)),
        montage_speed=float(collection.get("montage_speed", 0.0)),
        snapshot=snapshot,
        sha256=stable_json_sha256(snapshot),
    )
    _validate_locked_config(config)
    return config


def _validate_locked_config(config: GroundingConfig) -> None:
    """拒绝任何偏离已锁定Grounding v2矩阵的配置。"""
    expected_seeds = (210, 212, 248, 253, 260, 265, 286, 292, 296, 304,
                      306, 319, 326, 336, 345, 357, 358, 381, 385, 396)
    expected_tasks = ("red_on_blue", "green_on_blue", "red_on_yellow", "green_on_yellow")
    expected_prompts = {
        "red_on_blue": "Put the red cube on the blue pad.",
        "green_on_blue": "Put the green cube on the blue pad.",
        "red_on_yellow": "Put the red cube on the yellow pad.",
        "green_on_yellow": "Put the green cube on the yellow pad.",
    }
    expected_square = tuple(
        tuple(expected_tasks[(row + column) % 4] for column in range(4))
        for row in range(4)
    )
    checks = {
        "repo_id": config.repo_id == GROUNDING_REPO_ID,
        "dataset_version": config.dataset_version == GROUNDING_DATASET_VERSION,
        "scene_seeds": config.scene_seeds == expected_seeds,
        "tasks": config.tasks == expected_tasks,
        "canonical_prompts": config.canonical_prompts == expected_prompts,
        "latin_square": config.latin_square == expected_square,
        "fps": config.fps == 20,
        "max_frames": config.max_frames == 400,
        "pilot_scene_count": config.pilot_scene_count == 2,
        "expected_total": config.expected_total == 80,
        "strict_success_required": config.strict_success_required,
        "montage_speed": 2.0 <= config.montage_speed <= 4.0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Grounding v2锁定配置发生漂移: {failed}")


def build_plan(config: GroundingConfig) -> list[QueueItem]:
    """按scene优先和Latin square顺序生成完整80条队列。"""
    plan: list[QueueItem] = []
    for scene_index, scene_seed in enumerate(config.scene_seeds):
        for position, task_id in enumerate(config.latin_square[scene_index % 4], start=1):
            queue_key = f"scene={scene_seed}|task={task_id}|prompt={PROMPT_MODE}"
            plan.append(
                QueueItem(
                    queue_index=len(plan),
                    scene_index=scene_index,
                    collection_position=position,
                    scene_seed=scene_seed,
                    task_id=task_id,
                    prompt_mode=PROMPT_MODE,
                    prompt=config.canonical_prompts[task_id],
                    queue_key=queue_key,
                )
            )
    keys = [item.queue_key for item in plan]
    combinations = {(item.scene_seed, item.task_id) for item in plan}
    expected = {(seed, task) for seed in config.scene_seeds for task in config.tasks}
    if len(plan) != config.expected_total or len(set(keys)) != len(keys) or combinations != expected:
        raise ValueError("矩阵计划存在重复键或缺失scene-task组合")
    return plan


def plan_for_mode(config: GroundingConfig, pilot: bool) -> list[QueueItem]:
    """返回pilot的8条或全量80条计划。"""
    plan = build_plan(config)
    if pilot:
        return [item for item in plan if item.scene_index < config.pilot_scene_count]
    return plan


def code_identity(project_root: Path) -> dict[str, Any]:
    """记录Git提交、工作区状态和采集关键代码哈希。"""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root, check=True,
            capture_output=True, text=True, encoding="utf-8",
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=project_root, check=True,
            capture_output=True, text=True, encoding="utf-8",
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unavailable", None
    files = [
        "collector/collect_matrix.py", "collector/collection_plan.py",
        "collector/dataset_io.py", "collector/control.py", "sim/environment.py",
    ]
    hashes = {}
    for relative in files:
        target = project_root / relative
        if target.is_file():
            hashes[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
    return {"git_commit": commit, "git_dirty": dirty, "file_sha256": hashes}


def atomic_write_json(path: Path, value: Any) -> None:
    """使用同目录临时文件原子写入JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_json_unique(path: Path) -> Any:
    """读取JSON并拒绝同一对象内的重复键。"""
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        """把键值对转为字典，并在发现重复键时立即失败。"""
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"JSON包含重复键: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)


def _queue_csv_matches(path: Path, plan: list[QueueItem]) -> bool:
    """确认恢复时使用的队列CSV仍与锁定计划逐行一致。"""
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(plan):
        return False
    for row, item in zip(rows, plan, strict=True):
        if (
            row.get("queue_key") != item.queue_key
            or row.get("prompt") != item.prompt
            or row.get("task_id") != item.task_id
            or row.get("scene_seed") != str(item.scene_seed)
            or row.get("collection_position") != str(item.collection_position)
        ):
            return False
    return True


def initialize_workspace(config: GroundingConfig, resume: bool) -> dict[str, Any]:
    """创建采集工作区，或严格核对后恢复已有工作区。"""
    work_root = config.work_root
    manifest_path = work_root / MANIFEST_FILENAME
    progress_path = work_root / PROGRESS_FILENAME
    plan = build_plan(config)
    if work_root.exists() and any(work_root.iterdir()):
        if not resume:
            raise FileExistsError(f"采集工作目录非空；续采必须显式添加 --resume: {work_root}")
        if not manifest_path.is_file() or not progress_path.is_file():
            raise ValueError("采集工作目录缺少manifest或progress，禁止恢复")
        manifest = load_json_unique(manifest_path)
        progress = load_json_unique(progress_path)
        expected_keys = [item.queue_key for item in plan]
        mismatches = []
        if manifest.get("config_sha256") != config.sha256:
            mismatches.append("config_sha256")
        if manifest.get("repo_id") != config.repo_id:
            mismatches.append("repo_id")
        if manifest.get("dataset_version") != config.dataset_version:
            mismatches.append("dataset_version")
        if manifest.get("plan_keys") != expected_keys:
            mismatches.append("plan_keys")
        if not _queue_csv_matches(work_root / QUEUE_FILENAME, plan):
            mismatches.append("collection_queue.csv")
        completed = progress.get("completed")
        if not isinstance(completed, dict) or not set(completed).issubset(expected_keys):
            mismatches.append("completed")
        if mismatches:
            raise ValueError(f"严格恢复检查失败: {mismatches}")
        current_identity = code_identity(config.source_path.parents[1])
        history = manifest.setdefault(
            "code_identity_history",
            [{"observed_at": manifest.get("created_at"), "identity": manifest.get("code_identity", {})}],
        )
        if not history or history[-1].get("identity") != current_identity:
            history.append({"observed_at": utc_now(), "identity": current_identity})
            atomic_write_json(manifest_path, manifest)
        return progress
    if resume:
        raise FileNotFoundError(f"--resume指定的采集工作目录不存在: {work_root}")
    if config.root.exists() and any(config.root.iterdir()):
        raise FileExistsError(f"最终数据目录已非空，拒绝覆盖: {config.root}")
    work_root.mkdir(parents=True, exist_ok=True)
    config.shard_root.mkdir(parents=True, exist_ok=True)
    identity = code_identity(config.source_path.parents[1])
    manifest = {
        "schema_version": 1,
        "repo_id": config.repo_id,
        "dataset_version": config.dataset_version,
        "config_snapshot": config.snapshot,
        "config_sha256": config.sha256,
        "code_identity": identity,
        "code_identity_history": [{"observed_at": utc_now(), "identity": identity}],
        "plan_keys": [item.queue_key for item in plan],
        "created_at": utc_now(),
    }
    progress = {"schema_version": 1, "completed": {}, "updated_at": utc_now()}
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(progress_path, progress)
    _write_queue_csv(work_root / QUEUE_FILENAME, plan)
    return progress


def _write_queue_csv(path: Path, plan: list[QueueItem]) -> None:
    """写出便于人工核对的固定80条队列。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(plan[0])))
        writer.writeheader()
        writer.writerows(asdict(item) for item in plan)


def load_progress(config: GroundingConfig) -> dict[str, Any]:
    """读取采集进度，缺失或损坏时明确失败。"""
    path = config.work_root / PROGRESS_FILENAME
    try:
        progress = load_json_unique(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"采集进度损坏: {path}") from exc
    if not isinstance(progress.get("completed"), dict):
        raise ValueError("采集进度缺少completed映射")
    return progress


def validate_shard_record(
    config: GroundingConfig,
    item: QueueItem,
    record: dict[str, Any],
) -> list[str]:
    """校验一个已完成分片的契约、Parquet和两路视频。

    Args:
        config: 当前锁定配置。
        item: 分片对应队列项。
        record: ``collection_progress.json`` 中的完成记录。

    Returns:
        空列表表示完整，否则返回全部错误描述。
    """
    import av
    import pyarrow.parquet as pq

    errors: list[str] = []
    shard_name = str(record.get("shard_name", ""))
    shard = config.shard_root / shard_name
    frame_count = int(record.get("frame_count", -1))
    if shard_name != item.shard_name:
        errors.append("shard_name")
    contract_path = shard / "meta" / "collector_contract.json"
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        contract = {}
        errors.append("contract")
    if contract.get("repo_id") != config.repo_id:
        errors.append("contract.repo_id")
    if contract.get("dataset_version") != config.dataset_version:
        errors.append("contract.dataset_version")
    if contract.get("queue_key") != item.queue_key:
        errors.append("contract.queue_key")
    episodes = contract.get("episodes", [])
    if len(episodes) != 1 or (episodes and int(episodes[0].get("frame_count", -1)) != frame_count):
        errors.append("contract.episodes")
    parquet_paths = list((shard / "data").glob("chunk-*/*.parquet"))
    try:
        parquet_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in parquet_paths)
    except Exception:
        parquet_rows = -1
    if len(parquet_paths) != 1 or parquet_rows != frame_count:
        errors.append("parquet")
    for camera in ("observation.images.agent", "observation.images.wrist"):
        video_paths = list((shard / "videos" / camera).glob("chunk-*/*.mp4"))
        try:
            decoded = 0
            for path in video_paths:
                with av.open(str(path)) as container:
                    decoded += sum(1 for _ in container.decode(video=0))
        except Exception:
            decoded = -1
        if len(video_paths) != 1 or decoded != frame_count:
            errors.append(f"video.{camera}")
    return errors


def validate_completed_shards(config: GroundingConfig, progress: dict[str, Any]) -> None:
    """在resume跳过组合前逐条验证所有已完成分片。"""
    plan_by_key = {item.queue_key: item for item in build_plan(config)}
    failures = {}
    for key, record in progress["completed"].items():
        errors = validate_shard_record(config, plan_by_key[key], record)
        if errors:
            failures[key] = errors
    if failures:
        raise ValueError(f"已完成分片不完整，禁止resume跳过: {failures}")


def record_completion(config: GroundingConfig, item: QueueItem, record: dict[str, Any]) -> None:
    """原子登记一个经人工Enter确认的唯一队列键。"""
    progress = load_progress(config)
    completed = progress["completed"]
    if item.queue_key in completed:
        raise ValueError(f"拒绝重复保存队列键: {item.queue_key}")
    completed[item.queue_key] = record
    progress["updated_at"] = utc_now()
    atomic_write_json(config.work_root / PROGRESS_FILENAME, progress)


def copy_sidecars_to_final(config: GroundingConfig) -> None:
    """把采集契约、队列、进度和报告复制到最终数据集根目录。"""
    import shutil

    names = (
        MANIFEST_FILENAME, QUEUE_FILENAME, PROGRESS_FILENAME, REVIEW_FILENAME,
        "dataset_validation.json", "paired_initial_state_check.csv",
        "paired_action_difference.csv", "grounding_v2_eda_report.md",
    )
    for name in names:
        source = config.work_root / name
        if source.is_file():
            shutil.copy2(source, config.root / name)
    for directory_name in ("review_montages", INITIAL_REFERENCE_DIRNAME):
        source_directory = config.work_root / directory_name
        target_directory = config.root / directory_name
        if source_directory.is_dir():
            if target_directory.exists():
                shutil.rmtree(target_directory)
            shutil.copytree(source_directory, target_directory)
