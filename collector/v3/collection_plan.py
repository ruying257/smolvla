"""杯子V3锁定配置、交替任务队列和严格恢复状态。"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from collector.v3.task_spec import PROMPT_MODE, TASK_IDS, TASK_PROMPTS
from sim.mug_environment import MUG_SAMPLE_X_RANGE, MUG_SAMPLE_Y_RANGE


MUG_REPO_ID = "smolvla_ur10e_mug_v3"
MUG_DATASET_VERSION = "smolvla_ur10e_mug_v3"
MANIFEST_FILENAME = "collection_manifest.json"
QUEUE_FILENAME = "collection_queue.csv"
PROGRESS_FILENAME = "collection_progress.json"
REVIEW_FILENAME = "review_status.csv"
PILOT_VALIDATION_FILENAME = "pilot_validation.json"
REDO_KEYS_FILENAME = "redo_keys.txt"
EXPECTED_CAMERA_FEATURES = {
    "observation.images.agent": "agentview",
    "observation.images.wrist": "d435i_rgb",
}


@dataclass(frozen=True)
class MugCollectionConfig:
    """保存已经完整校验的杯子V3采集配置。

    Attributes:
        source_path: 配置文件绝对路径。
        root: 最终40条LeRobot数据集目录。
        repo_id: 锁定的数据集仓库身份。
        dataset_version: 锁定的数据集版本。
        scene_seeds: 按4×5网格顺序排列的20个seed。
        pilot_scene_seeds: 两个中等难度pilot seed。
        tasks: 两个杯子放置任务。
        canonical_prompts: 任务到唯一训练指令的映射。
        fps: 采集与动作回放频率。
        viewer_fps: GUI刷新频率。
        max_frames: 每条episode最大帧数。
        expected_total: 正式矩阵总episode数。
        strict_success_required: 是否强制环境严格成功。
        montage_speed: 默认人工复核视频倍速。
        camera_features: LeRobot视频键到MuJoCo相机名称的映射。
        seed_selection: 固定筛选规格和报告身份。
        fixed_pad_positions: 蓝黄区域固定坐标。
        snapshot: 原始YAML的规范化副本。
        sha256: 配置快照稳定SHA-256。
    """

    source_path: Path
    root: Path
    repo_id: str
    dataset_version: str
    scene_seeds: tuple[int, ...]
    pilot_scene_seeds: tuple[int, ...]
    tasks: tuple[str, ...]
    canonical_prompts: dict[str, str]
    fps: int
    viewer_fps: int
    max_frames: int
    expected_total: int
    strict_success_required: bool
    montage_speed: float
    camera_features: dict[str, str]
    seed_selection: dict[str, Any]
    fixed_pad_positions: dict[str, list[float]]
    snapshot: dict[str, Any]
    sha256: str

    @property
    def work_root(self) -> Path:
        """返回与最终数据集隔离的可恢复工作目录。

        Returns:
            最终目录同级、名称带前导点和 ``_collection`` 的路径。
        """
        return self.root.parent / f".{self.root.name}_collection"

    @property
    def shard_root(self) -> Path:
        """返回单episode原子分片目录。

        Returns:
            工作目录下的 ``episode_shards`` 路径。
        """
        return self.work_root / "episode_shards"


@dataclass(frozen=True)
class QueueItem:
    """描述一个唯一且可恢复的seed-task采集项。

    Attributes:
        queue_index: 最终数据集中的稳定episode顺序。
        scene_index: seed在20场景列表中的索引。
        collection_position: 同一seed内任务采集位次，取1或2。
        scene_seed: 杯子布局seed。
        task_id: 内部任务标识。
        prompt_mode: 固定为canonical。
        prompt: 唯一英文训练指令。
        queue_key: 用于恢复和局部重采的稳定唯一键。
    """

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
        """生成Windows安全且稳定的单episode分片名。

        Returns:
            包含队列索引、seed、任务和键摘要的目录名。
        """
        digest = hashlib.sha256(self.queue_key.encode("utf-8")).hexdigest()[:12]
        return f"{self.queue_index:03d}_{self.scene_seed}_{self.task_id}_{digest}"


def utc_now() -> str:
    """返回带时区的UTC ISO 8601时间。

    Returns:
        可直接写入JSON记录的时间字符串。
    """
    return datetime.now(timezone.utc).isoformat()


def stable_json_sha256(value: Any) -> str:
    """计算与字典键顺序无关的稳定JSON SHA-256。

    Args:
        value: 可被JSON序列化的值。

    Returns:
        小写十六进制SHA-256。
    """
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """流式计算单个文件SHA-256。

    Args:
        path: 需要验签的文件路径。

    Returns:
        小写十六进制SHA-256。
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    """通过同目录临时文件原子写入UTF-8 JSON。

    Args:
        path: 目标JSON路径。
        value: 需要序列化的值。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _require_exact_keys(snapshot: dict[str, Any]) -> None:
    """拒绝配置缺字段或包含未识别字段。

    Args:
        snapshot: YAML顶层映射。

    Raises:
        ValueError: 字段集合与V3锁定契约不完全一致时抛出。
    """
    expected = {
        "root", "repo_id", "dataset_version", "scene_seeds", "pilot_scene_seeds",
        "tasks", "canonical_prompts", "fps", "viewer_fps", "max_frames",
        "expected_total", "strict_success_required", "montage_speed",
        "camera_features", "seed_selection", "fixed_pad_positions",
    }
    if set(snapshot) != expected:
        raise ValueError(
            f"V3配置字段必须严格匹配，missing={sorted(expected-set(snapshot))}, "
            f"extra={sorted(set(snapshot)-expected)}"
        )


def load_config(path: Path) -> MugCollectionConfig:
    """读取并严格验证杯子V3锁定配置。

    Args:
        path: ``configs/collect_mug_v3.yaml``或等价配置路径。

    Returns:
        路径已解析、字段已规范化的不可变配置。

    Raises:
        FileNotFoundError: 配置或seed筛选报告不存在时抛出。
        ValueError: 身份、seed、任务、频率、相机或报告发生漂移时抛出。
    """
    source_path = path.resolve()
    snapshot = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(snapshot, dict):
        raise ValueError("V3配置顶层必须是映射")
    _require_exact_keys(snapshot)
    seeds = tuple(snapshot["scene_seeds"])
    pilots = tuple(snapshot["pilot_scene_seeds"])
    tasks = tuple(snapshot["tasks"])
    if snapshot["repo_id"] != MUG_REPO_ID or snapshot["dataset_version"] != MUG_DATASET_VERSION:
        raise ValueError("V3 repo_id或dataset_version发生漂移")
    if len(seeds) != 20 or len(set(seeds)) != 20 or not all(isinstance(seed, int) for seed in seeds):
        raise ValueError("scene_seeds必须是20个唯一整数")
    if len(pilots) != 2 or len(set(pilots)) != 2 or not set(pilots).issubset(seeds):
        raise ValueError("pilot_scene_seeds必须是scene_seeds中的两个唯一seed")
    if tasks != TASK_IDS or snapshot["canonical_prompts"] != TASK_PROMPTS:
        raise ValueError("任务或canonical文本发生漂移")
    if snapshot["fps"] != 20 or snapshot["viewer_fps"] != 60:
        raise ValueError("V3采集必须为20 Hz且Viewer必须为60 Hz")
    if snapshot["max_frames"] != 400 or snapshot["expected_total"] != 40:
        raise ValueError("V3必须限制400帧且正式矩阵必须为40条")
    if snapshot["strict_success_required"] is not True:
        raise ValueError("V3必须启用严格成功门禁")
    if snapshot["camera_features"] != EXPECTED_CAMERA_FEATURES:
        raise ValueError("V3相机feature或MuJoCo相机映射发生漂移")
    selection = snapshot["seed_selection"]
    expected_selection = {
        "candidate_start": 0,
        "candidate_stop": 10000,
        "grid_columns": 4,
        "grid_rows": 5,
        "minimum_pairwise_distance": 0.04,
    }
    if any(selection.get(key) != value for key, value in expected_selection.items()):
        raise ValueError("seed筛选规格发生漂移")
    project_root = source_path.parents[1]
    report_path = project_root / selection["report_json"]
    if not report_path.is_file() or file_sha256(report_path) != selection["report_sha256"]:
        raise ValueError("seed筛选报告缺失或SHA-256不匹配")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("scene_seeds") != list(seeds) or report.get("pilot_scene_seeds") != list(pilots):
        raise ValueError("配置seed与筛选报告不一致")
    positions = np.asarray([(item["x"], item["y"]) for item in report["selected"]])
    minimum = min(
        np.linalg.norm(positions[left] - positions[right])
        for left in range(len(positions))
        for right in range(left + 1, len(positions))
    )
    if minimum < 0.04 or len({(item["column"], item["row"]) for item in report["selected"]}) != 20:
        raise ValueError("seed报告不满足4厘米间距或4×5完整覆盖")
    pads = snapshot["fixed_pad_positions"]
    if pads != {"blue": [0.55, -0.22, 0.8005], "yellow": [0.55, 0.22, 0.8005]}:
        raise ValueError("固定放置区坐标发生漂移")
    root = Path(snapshot["root"])
    if not root.is_absolute():
        root = project_root / root
    return MugCollectionConfig(
        source_path=source_path,
        root=root.resolve(),
        repo_id=snapshot["repo_id"],
        dataset_version=snapshot["dataset_version"],
        scene_seeds=seeds,
        pilot_scene_seeds=pilots,
        tasks=tasks,
        canonical_prompts=dict(snapshot["canonical_prompts"]),
        fps=20,
        viewer_fps=60,
        max_frames=400,
        expected_total=40,
        strict_success_required=True,
        montage_speed=float(snapshot["montage_speed"]),
        camera_features=dict(snapshot["camera_features"]),
        seed_selection=dict(selection),
        fixed_pad_positions={key: list(value) for key, value in pads.items()},
        snapshot=snapshot,
        sha256=stable_json_sha256(snapshot),
    )


def build_plan(config: MugCollectionConfig) -> list[QueueItem]:
    """生成20个共享scene、蓝黄顺序交替的40项队列。

    Args:
        config: 已严格校验的V3配置。

    Returns:
        按scene优先排列的40个唯一队列项。

    Raises:
        ValueError: 队列数量或唯一键不满足锁定矩阵时抛出。
    """
    items: list[QueueItem] = []
    for scene_index, seed in enumerate(config.scene_seeds):
        task_order = config.tasks if scene_index % 2 == 0 else tuple(reversed(config.tasks))
        for position, task_id in enumerate(task_order, start=1):
            key = f"scene={seed}|task={task_id}|prompt={PROMPT_MODE}"
            items.append(QueueItem(
                queue_index=len(items),
                scene_index=scene_index,
                collection_position=position,
                scene_seed=seed,
                task_id=task_id,
                prompt_mode=PROMPT_MODE,
                prompt=config.canonical_prompts[task_id],
                queue_key=key,
            ))
    if len(items) != config.expected_total or len({item.queue_key for item in items}) != len(items):
        raise ValueError("V3队列必须包含40个唯一键")
    return items


def plan_for_mode(config: MugCollectionConfig, pilot: bool) -> list[QueueItem]:
    """返回pilot四条或正式全部四十条计划。

    Args:
        config: 已验证配置。
        pilot: 是否仅选择两个pilot scene。

    Returns:
        保持完整队列相对顺序的计划子集。
    """
    plan = build_plan(config)
    if not pilot:
        return plan
    return [item for item in plan if item.scene_seed in config.pilot_scene_seeds]


def _code_identity(project_root: Path) -> dict[str, Any]:
    """读取当前Git提交与工作树状态作为采集代码身份。

    Args:
        project_root: Git项目根目录。

    Returns:
        包含commit和dirty布尔值的字典；Git不可用时使用unknown。
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=project_root,
            check=True, capture_output=True, text=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": True}
    return {"commit": commit, "dirty": dirty}


def initialize_workspace(config: MugCollectionConfig, resume: bool) -> dict[str, Any]:
    """创建新工作区或严格验证后恢复既有工作区。

    Args:
        config: 当前锁定配置。
        resume: 是否显式允许使用非空工作区。

    Returns:
        当前进度JSON内容。

    Raises:
        FileExistsError: 输出非空且没有 ``--resume``，或最终目录已存在时抛出。
        ValueError: 恢复时manifest、计划或进度损坏时抛出。
    """
    if config.root.exists() and any(config.root.iterdir()):
        raise FileExistsError(f"最终V3数据目录已存在，拒绝采集覆盖: {config.root}")
    work_nonempty = config.work_root.exists() and any(config.work_root.iterdir())
    if work_nonempty and not resume:
        raise FileExistsError(f"V3采集工作区非空；续采必须显式添加--resume: {config.work_root}")
    manifest_path = config.work_root / MANIFEST_FILENAME
    progress_path = config.work_root / PROGRESS_FILENAME
    plan = build_plan(config)
    plan_sha = stable_json_sha256([item.queue_key for item in plan])
    if work_nonempty:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("V3 manifest或progress缺失/损坏") from exc
        mismatches = [
            name for name, expected in {
                "config_sha256": config.sha256,
                "plan_sha256": plan_sha,
                "repo_id": config.repo_id,
                "dataset_version": config.dataset_version,
                "fps": config.fps,
                "camera_features": config.camera_features,
            }.items() if manifest.get(name) != expected
        ]
        if mismatches:
            raise ValueError(f"V3恢复契约不匹配: {mismatches}")
        _validate_progress_shape(progress, plan)
        return progress

    config.shard_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "repo_id": config.repo_id,
        "dataset_version": config.dataset_version,
        "config_snapshot": config.snapshot,
        "config_sha256": config.sha256,
        "plan_sha256": plan_sha,
        "planned_keys": [item.queue_key for item in plan],
        "fps": config.fps,
        "camera_features": config.camera_features,
        "code_identity": _code_identity(config.source_path.parents[1]),
        "created_at": utc_now(),
    }
    progress = {"config_sha256": config.sha256, "completed": {}, "updated_at": utc_now()}
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(progress_path, progress)
    with (config.work_root / QUEUE_FILENAME).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(plan[0])))
        writer.writeheader()
        writer.writerows(asdict(item) for item in plan)
    with (config.work_root / REVIEW_FILENAME).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scene_seed", "status"])
        writer.writerows((seed, "pending") for seed in config.scene_seeds)
    return progress


def _validate_progress_shape(progress: dict[str, Any], plan: Sequence[QueueItem]) -> None:
    """验证恢复进度只包含当前计划的唯一完成键。

    Args:
        progress: 从 ``collection_progress.json`` 读取的对象。
        plan: 当前锁定完整队列。

    Raises:
        ValueError: 进度结构错误、键越界或记录键不自洽时抛出。
    """
    completed = progress.get("completed")
    if not isinstance(completed, dict):
        raise ValueError("V3 progress.completed必须是映射")
    allowed = {item.queue_key for item in plan}
    if not set(completed).issubset(allowed):
        raise ValueError("V3 progress包含计划外队列键")
    for key, record in completed.items():
        if not isinstance(record, dict) or record.get("queue_key") != key:
            raise ValueError(f"V3 progress完成记录损坏: {key}")


def load_progress(config: MugCollectionConfig) -> dict[str, Any]:
    """读取并验证当前V3进度。

    Args:
        config: 当前锁定配置。

    Returns:
        完成项以队列键索引的进度对象。

    Raises:
        ValueError: 文件缺失、JSON损坏或结构不合法时抛出。
    """
    try:
        progress = json.loads((config.work_root / PROGRESS_FILENAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("V3 collection_progress.json缺失或损坏") from exc
    if progress.get("config_sha256") != config.sha256:
        raise ValueError("V3 progress配置身份不匹配")
    _validate_progress_shape(progress, build_plan(config))
    return progress


def record_completion(
    config: MugCollectionConfig,
    item: QueueItem,
    record: dict[str, Any],
) -> None:
    """在分片完整验签后原子登记一个完成键。

    Args:
        config: 当前锁定配置。
        item: 即将登记的队列项。
        record: 已通过视频、Parquet和契约校验的完成记录。

    Raises:
        ValueError: 队列键已经完成或记录身份不一致时抛出。
    """
    from collector.v3.dataset_io import validate_episode_shard

    progress = load_progress(config)
    if item.queue_key in progress["completed"]:
        raise ValueError(f"拒绝重复完成队列键: {item.queue_key}")
    if record.get("queue_key") != item.queue_key:
        raise ValueError("完成记录queue_key与计划不一致")
    if record.get("config_sha256") != config.sha256:
        raise ValueError("完成记录config_sha256与当前配置不一致")
    validate_episode_shard(config.shard_root / record["shard_name"], record)
    progress["completed"][item.queue_key] = record
    progress["updated_at"] = utc_now()
    atomic_write_json(config.work_root / PROGRESS_FILENAME, progress)


def validate_completed_shards(config: MugCollectionConfig, progress: dict[str, Any]) -> None:
    """逐条验证恢复进度引用的所有活动分片。

    Args:
        config: 当前锁定配置。
        progress: 已读取的恢复进度。

    Raises:
        RuntimeError: 任一契约、Parquet、视频或帧数不完整时抛出。
    """
    from collector.v3.dataset_io import validate_episode_shard

    for key, record in progress["completed"].items():
        if record.get("config_sha256") != config.sha256:
            raise RuntimeError(f"已完成V3分片配置哈希不匹配: {key}")
        try:
            validate_episode_shard(config.shard_root / record["shard_name"], record)
        except Exception as exc:
            raise RuntimeError(f"已完成V3分片损坏，不能跳过: {key}: {exc}") from exc


def prepare_redo(config: MugCollectionConfig, queue_key: str) -> QueueItem:
    """归档指定已完成分片并将其恢复为待采状态。

    Args:
        config: 当前锁定配置。
        queue_key: 完整的 ``scene=...|task=...|prompt=canonical`` 键。

    Returns:
        需要立即重采的队列项。

    Raises:
        ValueError: 键不属于计划或尚未完成时抛出。
        FileNotFoundError: 已登记活动分片缺失时抛出。
    """
    plan = {item.queue_key: item for item in build_plan(config)}
    if queue_key not in plan:
        raise ValueError(f"未知V3 redo key: {queue_key}")
    progress = load_progress(config)
    record = progress["completed"].get(queue_key)
    if record is None:
        raise ValueError(f"只能redo已完成键: {queue_key}")
    source = config.shard_root / record["shard_name"]
    if not source.is_dir():
        raise FileNotFoundError(f"redo分片不存在: {source}")
    archive = config.work_root / "archived_shards" / utc_now().replace(":", "-")
    archive.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(archive / source.name))
    del progress["completed"][queue_key]
    progress["updated_at"] = utc_now()
    atomic_write_json(config.work_root / PROGRESS_FILENAME, progress)
    (config.work_root / "dataset_validation.json").unlink(missing_ok=True)
    if plan[queue_key].scene_seed in config.pilot_scene_seeds:
        (config.work_root / PILOT_VALIDATION_FILENAME).unlink(missing_ok=True)
    (config.work_root / "review_montages" / f"scene_{plan[queue_key].scene_seed}.mp4").unlink(
        missing_ok=True
    )
    review_path = config.work_root / REVIEW_FILENAME
    if review_path.is_file():
        with review_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        with review_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["scene_seed", "status"])
            writer.writeheader()
            for row in rows:
                if int(row["scene_seed"]) == plan[queue_key].scene_seed:
                    row["status"] = "pending"
                writer.writerow(row)
    return plan[queue_key]
