"""Grounding v2配对分片、最终LeRobot数据集与人工复核验收。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import statistics
import uuid
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from collector.collection_plan import (
    MANIFEST_FILENAME,
    PILOT_VALIDATION_FILENAME,
    REVIEW_FILENAME,
    QueueItem,
    atomic_write_json,
    build_plan,
    copy_sidecars_to_final,
    hash_array,
    load_config,
    load_initial_reference,
    load_progress,
    plan_for_mode,
    stable_json_sha256,
    utc_now,
    validate_completed_shards,
)
from collector.dataset_io import LeRobotEpisodeWriter
from sim.environment import CleanTabletopEnv, SceneSnapshot, TASK_INITIAL_BODY_POSITIONS


def build_parser() -> argparse.ArgumentParser:
    """创建验收命令行解析器。"""
    parser = argparse.ArgumentParser(description="验收Grounding v2配对数据集")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true", help="只验收前两个scene的8条")
    parser.add_argument("--finalize", action="store_true", help="全量PASS后物化最终80条数据集")
    return parser


def _read_shard_table(shard: Path) -> pa.Table:
    """读取一个单episode分片的唯一数据Parquet。"""
    paths = list((shard / "data").glob("chunk-*/*.parquet"))
    if len(paths) != 1:
        raise ValueError(f"分片数据Parquet数量必须为1: {shard}")
    return pq.read_table(paths[0])


def _vector_column(table: pa.Table, name: str, dtype: Any = np.float32) -> np.ndarray:
    """把Arrow定长列表列转换为二维NumPy数组。"""
    return np.asarray(table[name].to_pylist(), dtype=dtype)


def _task_texts(shard: Path) -> list[str]:
    """读取LeRobot原生tasks元数据中的全部文本。"""
    table = pq.read_table(shard / "meta" / "tasks.parquet")
    text_column = "__index_level_0__"
    if text_column not in table.column_names:
        raise ValueError(f"tasks.parquet缺少任务文本索引列: {shard}")
    return [str(value) for value in table[text_column].to_pylist()]


def _resample_actions(actions: np.ndarray, samples: int = 100) -> np.ndarray:
    """按归一化时间线性重采样动作，便于不等长轨迹比较。"""
    if len(actions) == 1:
        return np.repeat(actions, samples, axis=0)
    source = np.linspace(0.0, 1.0, len(actions))
    target = np.linspace(0.0, 1.0, samples)
    return np.stack([np.interp(target, source, actions[:, dim]) for dim in range(actions.shape[1])], axis=1)


def _action_difference(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    """报告两条轨迹的时归一化关节差异，不施加经验硬阈值。"""
    left_r = _resample_actions(left)
    right_r = _resample_actions(right)
    joint_l2 = np.linalg.norm(left_r[:, :6] - right_r[:, :6], axis=1)
    return {
        "mean_joint_l2": float(np.mean(joint_l2)),
        "max_joint_l2": float(np.max(joint_l2)),
        "gripper_disagreement_rate": float(np.mean(np.abs(left_r[:, 6] - right_r[:, 6]) > 0.5)),
        "length_ratio": float(min(len(left), len(right)) / max(len(left), len(right))),
    }


def _read_review_status(path: Path) -> dict[int, str]:
    """读取人工复核表；缺失时返回空映射并由验收报告失败。"""
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {int(row["scene_seed"]): row["status"] for row in csv.DictReader(handle)}


def _check_montage(path: Path) -> bool:
    """确认蒙太奇可解码、非空且分辨率为512×512。"""
    try:
        frame_count = 0
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                if frame.width != 512 or frame.height != 512:
                    return False
                frame_count += 1
        return frame_count > 0
    except Exception:
        return False


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """以UTF-8 BOM写出便于Excel查看的CSV报告。"""
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _validate_records(config: Any, items: list[QueueItem]) -> dict[str, Any]:
    """执行分片、分布、配对、数值和人工复核的全部检查。"""
    progress = load_progress(config)
    validate_completed_shards(config, progress)
    completed = progress["completed"]
    errors: list[str] = []
    warnings: list[str] = []
    selected_keys = [item.queue_key for item in items]
    missing = [key for key in selected_keys if key not in completed]
    if missing:
        errors.append(f"缺少{len(missing)}个计划键")
    selected_records = {key: completed[key] for key in selected_keys if key in completed}
    if len(selected_records) != len(set(selected_records)):
        errors.append("队列键重复")

    actions_by_key: dict[str, np.ndarray] = {}
    state_rows: list[dict[str, Any]] = []
    task_counts: Counter[str] = Counter()
    frame_counts: list[int] = []
    with CleanTabletopEnv() as env:
        arm_limits = np.asarray(env.model.actuator_ctrlrange[:6], dtype=np.float64)
    for item in items:
        record = selected_records.get(item.queue_key)
        if record is None:
            continue
        shard = config.shard_root / record["shard_name"]
        table = _read_shard_table(shard)
        states = _vector_column(table, "observation.state")
        actions = _vector_column(table, "action")
        actions_by_key[item.queue_key] = actions
        frame_count = int(record["frame_count"])
        frame_counts.append(frame_count)
        task_counts[item.task_id] += 1
        if not 1 <= frame_count <= config.max_frames:
            errors.append(f"帧数越界: {item.queue_key}={frame_count}")
        if len(states) != frame_count or len(actions) != frame_count:
            errors.append(f"Parquet帧数不一致: {item.queue_key}")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            errors.append(f"存在NaN或Inf: {item.queue_key}")
        if np.any(actions[:, :6] < arm_limits[:, 0]) or np.any(actions[:, :6] > arm_limits[:, 1]):
            errors.append(f"动作越过MuJoCo actuator限位: {item.queue_key}")
        if np.any(actions[:, 6] < 0.0) or np.any(actions[:, 6] > 1.0):
            errors.append(f"夹爪动作越界: {item.queue_key}")
        if set(_task_texts(shard)) != {item.prompt}:
            errors.append(f"任务文本非唯一canonical: {item.queue_key}")
        seeds = set(int(value) for value in table["scene_seed"].to_pylist())
        if seeds != {item.scene_seed}:
            errors.append(f"scene_seed列错误: {item.queue_key}")
        poses = _vector_column(table, "cube_initial_poses")
        expected_pose = np.asarray(record["cube_initial_poses"], dtype=np.float32).reshape(14)
        if poses.shape != (frame_count, 14) or not np.array_equal(poses, np.repeat(expected_pose[None], frame_count, axis=0)):
            errors.append(f"cube_initial_poses列不一致: {item.queue_key}")
        if hash_array(states[0].astype(np.float32)) != record["initial_robot_state_sha256"]:
            errors.append(f"首帧机器人状态哈希不一致: {item.queue_key}")

    paired_rows: list[dict[str, Any]] = []
    for seed in dict.fromkeys(item.scene_seed for item in items):
        scene_items = [item for item in items if item.scene_seed == seed]
        records = [selected_records.get(item.queue_key) for item in scene_items]
        complete = all(record is not None for record in records)
        state_match = complete and len({record["initial_robot_state_sha256"] for record in records if record}) == 1
        agent_match = complete and len({record["initial_agent_raw_sha256"] for record in records if record}) == 1
        wrist_match = complete and len({record["initial_wrist_raw_sha256"] for record in records if record}) == 1
        pose_hashes = {
            stable_json_sha256(record["cube_initial_poses"])
            for record in records if record is not None
        }
        pose_match = complete and len(pose_hashes) == 1
        try:
            reference = load_initial_reference(config, seed)
            reference_match = all(
                record is not None
                and record["initial_robot_state_sha256"] == reference["initial_robot_state_sha256"]
                and record["initial_agent_raw_sha256"] == reference["initial_agent_raw_sha256"]
                and record["initial_wrist_raw_sha256"] == reference["initial_wrist_raw_sha256"]
                and np.array_equal(
                    np.asarray(record["cube_initial_poses"], dtype=np.float64),
                    reference["cube_initial_poses"],
                )
                for record in records
            )
        except ValueError:
            reference_match = False
        paired_rows.append({
            "scene_seed": seed,
            "task_count": sum(record is not None for record in records),
            "robot_state_hash_match": state_match,
            "agent_raw_hash_match": agent_match,
            "wrist_raw_hash_match": wrist_match,
            "cube_pose_match": pose_match,
            "lossless_reference_match": reference_match,
        })
        if not all((complete, state_match, agent_match, wrist_match, pose_match, reference_match)):
            errors.append(f"同scene初始条件配对失败: scene={seed}")

    action_rows: list[dict[str, Any]] = []
    item_by_pair = {(item.scene_seed, item.task_id): item for item in items}
    for seed in dict.fromkeys(item.scene_seed for item in items):
        for pad in ("blue", "yellow"):
            red = item_by_pair.get((seed, f"red_on_{pad}"))
            green = item_by_pair.get((seed, f"green_on_{pad}"))
            if red is None or green is None or red.queue_key not in actions_by_key or green.queue_key not in actions_by_key:
                continue
            metrics = _action_difference(actions_by_key[red.queue_key], actions_by_key[green.queue_key])
            action_rows.append({"scene_seed": seed, "pad": pad, **metrics})
    if action_rows:
        ordered = sorted(row["mean_joint_l2"] for row in action_rows)
        lower = ordered[max(0, int(len(ordered) * 0.1) - 1)]
        warnings.append(
            "动作差异仅报告不设硬阈值；请优先人工复核mean_joint_l2最低的配对，"
            f"当前约10%分位={lower:.6f}"
        )

    selected_scenes = list(dict.fromkeys(item.scene_seed for item in items))
    review = _read_review_status(config.work_root / REVIEW_FILENAME)
    for seed in selected_scenes:
        if review.get(seed) != "pass":
            errors.append(f"人工复核未PASS: scene={seed}, status={review.get(seed, 'missing')}")
        montage = config.work_root / "review_montages" / f"scene_{seed}.mp4"
        if not _check_montage(montage):
            errors.append(f"复核蒙太奇缺失或损坏: scene={seed}")

    _write_csv(
        config.work_root / "paired_initial_state_check.csv",
        ["scene_seed", "task_count", "robot_state_hash_match", "agent_raw_hash_match", "wrist_raw_hash_match", "cube_pose_match", "lossless_reference_match"],
        paired_rows,
    )
    _write_csv(
        config.work_root / "paired_action_difference.csv",
        ["scene_seed", "pad", "mean_joint_l2", "max_joint_l2", "gripper_disagreement_rate", "length_ratio"],
        action_rows,
    )
    expected_per_task = len(selected_scenes)
    if task_counts != Counter({task: expected_per_task for task in config.tasks}):
        errors.append(f"任务分布错误: {dict(task_counts)}")
    report = {
        "status": "pass" if not errors else "fail",
        "mode": "pilot" if len(items) == config.pilot_scene_count * 4 else "full",
        "validated_at": utc_now(),
        "validated_keys": selected_keys,
        "episode_count": len(selected_records),
        "scene_count": len(selected_scenes),
        "task_counts": dict(task_counts),
        "canonical_count": len(selected_records),
        "other_prompt_count": 0,
        "frame_count": {
            "min": min(frame_counts) if frame_counts else 0,
            "max": max(frame_counts) if frame_counts else 0,
            "mean": statistics.mean(frame_counts) if frame_counts else 0.0,
            "total": sum(frame_counts),
        },
        "errors": errors,
        "warnings": warnings,
    }
    return report


def _write_eda(config: Any, report: dict[str, Any]) -> None:
    """生成供训练任务阅读的Grounding v2数据概览Markdown。"""
    frame = report["frame_count"]
    task_lines = "\n".join(f"- `{task}`: {count}" for task, count in report["task_counts"].items())
    error_lines = "\n".join(f"- {value}" for value in report["errors"]) or "- 无"
    warning_lines = "\n".join(f"- {value}" for value in report["warnings"]) or "- 无"
    content = f"""# Grounding v2 数据验收报告

- 状态：**{report['status'].upper()}**
- 模式：{report['mode']}
- episode：{report['episode_count']}
- scene：{report['scene_count']}
- canonical：{report['canonical_count']}
- 其他prompt：{report['other_prompt_count']}
- 帧数：min={frame['min']}，max={frame['max']}，mean={frame['mean']:.2f}，total={frame['total']}

## 任务分布

{task_lines}

## 自动与人工验收错误

{error_lines}

## 动作差异说明

{warning_lines}

动作差异明细见 `paired_action_difference.csv`。该指标只用于定位近似轨迹，
不使用缺乏实证依据的统一硬阈值；最终以逐scene蒙太奇人工复核为准。
"""
    (config.work_root / "grounding_v2_eda_report.md").write_text(content, encoding="utf-8")


def _materialize_final(config: Any, report: dict[str, Any]) -> None:
    """按固定队列顺序把80个已复核分片物化为最终LeRobot数据集。"""
    if report["status"] != "pass" or report["mode"] != "full":
        raise RuntimeError("只有全量验收PASS后才能物化最终数据集")
    if config.root.exists() and any(config.root.iterdir()):
        raise FileExistsError(f"最终数据目录非空，拒绝覆盖: {config.root}")
    if config.root.exists():
        config.root.rmdir()
    temporary_root = config.root.parent / f".{config.root.name}_finalizing_{uuid.uuid4().hex}"
    progress = load_progress(config)
    writer = LeRobotEpisodeWriter(
        temporary_root,
        dataset_version=config.dataset_version,
        repo_id=config.repo_id,
        contract_extras={
            "config_sha256": config.sha256,
            "matrix_plan_sha256": stable_json_sha256([item.queue_key for item in build_plan(config)]),
            "prompt_mode": "canonical_only",
        },
    )
    try:
        for item in build_plan(config):
            record = progress["completed"][item.queue_key]
            shard = config.shard_root / record["shard_name"]
            table = _read_shard_table(shard)
            states = _vector_column(table, "observation.state")
            actions = _vector_column(table, "action")
            camera_frames: dict[str, list[np.ndarray]] = {}
            for output_name, camera in (
                ("agent", "observation.images.agent"),
                ("wrist", "observation.images.wrist"),
            ):
                video_paths = list((shard / "videos" / camera).glob("chunk-*/*.mp4"))
                if len(video_paths) != 1:
                    raise RuntimeError(f"物化时相机视频数量错误: {item.queue_key}, {camera}")
                with av.open(str(video_paths[0])) as container:
                    camera_frames[output_name] = [
                        frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)
                    ]
            snapshot = SceneSnapshot(
                scene_seed=item.scene_seed,
                cube_initial_poses=np.asarray(record["cube_initial_poses"], dtype=np.float64),
                pad_positions=np.asarray(
                    [dict(TASK_INITIAL_BODY_POSITIONS)["task_blue_pad"],
                     dict(TASK_INITIAL_BODY_POSITIONS)["task_yellow_pad"]],
                    dtype=np.float64,
                ),
            )
            frame_count = int(record["frame_count"])
            if any(len(frames) != frame_count for frames in camera_frames.values()):
                raise RuntimeError(f"物化时视频帧数漂移: {item.queue_key}")
            for frame_index in range(frame_count):
                writer.add_frame(
                    {
                        "agent": camera_frames["agent"][frame_index],
                        "wrist": camera_frames["wrist"][frame_index],
                    },
                    states[frame_index],
                    actions[frame_index],
                    snapshot,
                    item.prompt,
                )
            episode_index = writer.save_episode(
                item.task_id, "canonical", item.prompt, snapshot, frame_count,
            )
            if episode_index != item.queue_index:
                raise RuntimeError(f"最终episode顺序漂移: expected={item.queue_index}, actual={episode_index}")
        writer.close()
        writer = None
        temporary_root.replace(config.root)
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if temporary_root.exists():
            shutil.rmtree(temporary_root, ignore_errors=True)


def _directory_sha256(root: Path) -> str:
    """计算排除自引用验收文件后的稳定数据集整体SHA-256。"""
    excluded = {"dataset_validation.json", "dataset_sha256.txt"}
    digest = hashlib.sha256()
    for path in sorted((value for value in root.rglob("*") if value.is_file()), key=lambda value: value.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _validate_final_payload(config: Any, expected_frames: int) -> list[str]:
    """重新读取最终Parquet、视频和契约，避免只信任物化过程。"""
    errors: list[str] = []
    contract_path = config.root / "meta" / "collector_contract.json"
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ["最终数据集缺少或损坏collector_contract.json"]
    episodes = contract.get("episodes", [])
    if len(episodes) != config.expected_total:
        errors.append(f"最终契约episode数错误: {len(episodes)}")
    expected_plan = build_plan(config)
    if len(episodes) == len(expected_plan):
        for item, episode in zip(expected_plan, episodes, strict=True):
            if (
                int(episode.get("episode_index", -1)) != item.queue_index
                or int(episode.get("scene_seed", -1)) != item.scene_seed
                or episode.get("task_id") != item.task_id
                or episode.get("template_id") != "canonical"
                or episode.get("task") != item.prompt
            ):
                errors.append(f"最终契约episode映射错误: queue_index={item.queue_index}")
    parquet_paths = list((config.root / "data").glob("chunk-*/*.parquet"))
    try:
        parquet_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in parquet_paths)
    except Exception:
        parquet_rows = -1
    if parquet_rows != expected_frames:
        errors.append(f"最终Parquet总帧数错误: expected={expected_frames}, actual={parquet_rows}")
    for camera in ("observation.images.agent", "observation.images.wrist"):
        decoded = 0
        paths = list((config.root / "videos" / camera).glob("chunk-*/*.mp4"))
        try:
            for path in paths:
                with av.open(str(path)) as container:
                    decoded += sum(1 for _ in container.decode(video=0))
        except Exception:
            decoded = -1
        if decoded != expected_frames:
            errors.append(f"最终视频总帧数错误: camera={camera}, expected={expected_frames}, actual={decoded}")
    return errors


def _update_manifest(config: Any) -> None:
    """把动态完成记录合并进最终collection_manifest。"""
    manifest_path = config.work_root / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    progress = load_progress(config)
    manifest["completed_keys"] = [item.queue_key for item in build_plan(config)]
    manifest["episodes"] = [progress["completed"][item.queue_key] for item in build_plan(config)]
    manifest["completed_at"] = utc_now()
    atomic_write_json(manifest_path, manifest)


def validate(config_path: Path, pilot: bool, finalize: bool) -> dict[str, Any]:
    """运行pilot或全量验收，并可在全量PASS后生成最终数据集。"""
    if pilot and finalize:
        raise ValueError("--pilot不能与--finalize同时使用")
    config = load_config(config_path)
    items = plan_for_mode(config, pilot=pilot)
    report = _validate_records(config, items)
    _write_eda(config, report)
    if pilot:
        atomic_write_json(config.work_root / PILOT_VALIDATION_FILENAME, report)
        return report
    atomic_write_json(config.work_root / "dataset_validation.json", report)
    if finalize and report["status"] == "pass":
        _update_manifest(config)
        _materialize_final(config, report)
        report["errors"].extend(_validate_final_payload(config, report["frame_count"]["total"]))
        report["status"] = "pass" if not report["errors"] else "fail"
        copy_sidecars_to_final(config)
        if report["status"] == "pass":
            dataset_sha256 = _directory_sha256(config.root)
            report["dataset_sha256"] = dataset_sha256
            report["dataset_sha256_scope"] = "all files except dataset_validation.json and dataset_sha256.txt"
            (config.work_root / "dataset_sha256.txt").write_text(dataset_sha256 + "\n", encoding="ascii")
            (config.root / "dataset_sha256.txt").write_text(dataset_sha256 + "\n", encoding="ascii")
        atomic_write_json(config.work_root / "dataset_validation.json", report)
        atomic_write_json(config.root / "dataset_validation.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """解析参数、运行验收并以退出码表达PASS或FAIL。"""
    args = build_parser().parse_args(argv)
    report = validate(args.config, args.pilot, args.finalize)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
