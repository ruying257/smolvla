# collector

SmolVLA MuJoCo 专家数据采集与回放工具。本目录按版本演进组织 V1（单任务）、V2（颜色 Grounding 矩阵）和 V3（杯子双目标区域任务）三套采集、回放与验收流水线，并抽取共享基础设施至 `common/` 子包。

## 目录结构

| 路径 | 功能 |
| --- | --- |
| `__init__.py` | 包初始化文件，声明本模块为「SmolVLA MuJoCo 专家数据采集与回放工具」。 |
| `common/` | V1 与 V2 共享的采集控制、数据契约和任务定义。 |
| `v1/` | SmolVLA UR10e 第一版单任务采集与回放工具。 |
| `v2/` | SmolVLA 颜色 Grounding v2 矩阵采集与验收工具。 |
| `v3/` | 杯子双目标区域任务的 V3 数据采集、回放与验收工具。 |

## common/

V1/V2 共享的基础设施，V3 在 `v3/` 内自带同名模块（schema 与杯子环境耦合，独立维护）。

### `control.py` — UR10e 键盘末端增量控制与阻尼最小二乘 IK

- `TeleopDelta`：一次键盘采样产生的末端平移、旋转 RPY、夹爪状态和「是否有效操作」标记。
- `read_teleop_delta(viewer, gripper)`：按 ACT 键位读取 WASD/RF 平移、方向键/QE 旋转、Space 切换夹爪。
- `_rotation_from_rpy(rpy)`：XYZ 欧拉角增量转旋转矩阵。
- `DifferentialIKController`：末端增量经阻尼最小二乘逆运动学映射到 UR10e 关节目标，IK 不收敛时保持上一帧目标。

### `dataset_io.py` — LeRobot v2 数据写入、契约校验和显式续采封装

- 常量：`DATASET_VERSION="smolvla_ur10e_v1"`、`DATASET_REPO_ID="smolvla_ur10e"`、`DATASET_FPS=20`、`CONTRACT_FILENAME="collector_contract.json"`。
- `_quote_ffconcat_path(path)`：转义 FFmpeg concat 清单中的单引号。
- `concatenate_video_files_utf8(...)`：使用 UTF-8 concat 清单合并视频，修复 LeRobot 0.4.4 在中文路径下从第二个 episode 起合并乱码的问题。
- `LeRobotEpisodeWriter`：原子 episode 写入器，负责 schema、视频编码与契约校验。
- `configure_hf_datasets_cache(...)`：配置 HuggingFace datasets 缓存目录。

### `state_machine.py` — 与 GUI 无关的 episode 采集状态机

- `CollectionPhase`：`IDLE` / `RECORDING` / `PENDING_CONFIRMATION` 三阶段枚举。
- `CollectionStateMachine`：
  - `observe_action(meaningful)`：检测到有效动作自动开录，RECORDING 阶段每帧返回 True 写入缓冲区。
  - `observe_success(success)`：录制中检测到严格成功时进入人工确认阶段。
  - `confirm()` / `discard()`：确认保存或丢弃当前缓冲区。
  - `reset()`：恢复空闲零帧初始状态。

### `task_spec.py` — 四类语言条件任务及训练措辞定义

- `CollectionTask`：积木到放置区任务的稳定标识（`task_id`、`cube_color`、`pad_color`）。
- `prompt(template_id)`：按 `canonical` 或 `synonym` 模板生成英文指令。
- `TASKS`：`red_on_blue` / `red_on_yellow` / `green_on_blue` / `green_on_yellow` 四类锁定任务。
- `choose_balanced_template(task_id, counts)`：按已保存计数为指定任务选择当前较少的训练措辞，数量相等时优先 canonical。

## v1/

SmolVLA UR10e 第一版单任务采集与回放工具，Windows 与 Ubuntu 共用入口。

### `collect.py` — Windows 与 Ubuntu 共用的 MuJoCo 键盘专家数据采集入口

CLI 参数：

- `--root`：LeRobot 数据集输出目录（必填）。
- `--task`：四类任务之一；缺省时在 Viewer 启动前交互选择。
- `--seed` / `--seeds`：单一起始 seed 或显式 seed 列表。
- `--episodes`：本次需确认保存的 episode 数量。
- `--resume`：严格校验已有数据集后续采。
- `--timeout-seconds`：单次尝试超时秒数（默认 40）。

主循环依赖 `control.DifferentialIKController`、`dataset_io.LeRobotEpisodeWriter`、`state_machine.CollectionStateMachine`，超时/取消/丢弃时按相同 seed 重试，严格成功后提示 Enter 保存或 Backspace 丢弃。

### `replay.py` — 不创建 MuJoCo 环境的双相机 episode 视频回放入口

CLI 参数：`--root`（数据集目录）、`--episode-index`（默认 0）。

通过 LeRobot 原生接口读取双相机视频与 state/action 序列，合成带任务文本、scene seed、帧号、状态和动作叠加的画面后用 OpenCV 窗口播放，不重建物理环境。

## v2/

SmolVLA 颜色 Grounding v2 矩阵采集与验收工具，目标是在固定 scene 下为四个任务采集反事实配对矩阵。

### `collection_plan.py` — Grounding v2 配置、Latin square 队列与严格恢复状态

- 常量：`GROUNDING_REPO_ID="smolvla_ur10e_grounding_v2"`、`MANIFEST_FILENAME`、`QUEUE_FILENAME`、`PROGRESS_FILENAME`、`REVIEW_FILENAME`、`PILOT_VALIDATION_FILENAME`、`INITIAL_REFERENCE_DIRNAME`。
- `GroundingConfig`：保存已校验的锁定配置（root、scene_seeds、tasks、canonical_prompts、latin_square、fps、max_frames、pilot_scene_count、expected_total、strict_success_required、montage_speed、snapshot、sha256）。
  - `work_root`：与最终数据集隔离的可恢复采集工作目录 `.{root.name}_collection`。
  - `shard_root`：按队列键存储单 episode 分片的目录。
- `QueueItem`：不可重复的 scene-task-canonical 采集组合；`shard_name` 用 SHA-256 前 12 位保证 Windows 路径兼容。
- `utc_now()`：ISO 8601 UTC 时间。
- `load_config` / `build_plan` / `plan_for_mode` / `load_progress` / `record_completion` / `initialize_workspace` / `validate_completed_shards` / `atomic_write_json` 等工具函数支撑配置加载、Latin square 队列生成、断点续采和分片完整性校验。

### `collect_matrix.py` — 按 scene 优先的 Grounding v2 反事实矩阵人工采集入口

CLI 参数：

- `--config`：锁定的 Grounding v2 YAML 配置（必填）。
- `--pilot`：只开放前两个 scene 的 8 条 pilot。
- `--resume`：严格校验后恢复已有工作区。
- `--redo-key`：归档并重采一个已完成键，格式 `scene=...|task=...|prompt=canonical`。

主循环按队列键采集，每条 episode 最多 400 帧，严格成功后由 Enter 保存、Backspace 重试同键；400 帧上限或取消/丢弃时按相同 queue key 重试。

### `build_review_montages.py` — 为每个 scene 生成四任务 2×2 同步人工复核视频

CLI 参数：`--config`（必填）、`--pilot`（仅前两 scene）、`--speed`（2 至 4 倍速）。

`LAYOUT` 把 `red_on_blue` / `green_on_blue` / `red_on_yellow` / `green_on_yellow` 排成 2×2 网格，按倍速解码每个相机 MP4（`_decode_sampled_video`），合成单个 scene 的四任务同步复核视频；`VALID_REVIEW_VALUES` 定义 `pending` / `pass` / `redo_*` 复核状态枚举。

### `validate_grounding_dataset.py` — Grounding v2 配对分片、最终 LeRobot 数据集与人工复核验收

CLI 参数：`--config`（必填）、`--pilot`（仅验收 8 条 pilot）、`--finalize`（全量 PASS 后物化最终 80 条数据集）。

校验内容：

- `_read_shard_table` / `_vector_column` / `_task_texts`：读取分片 Parquet、向量列和 tasks 元数据。
- `_resample_actions`：按归一化时间线性重采样动作，便于不等长轨迹比较。
- 校验分片契约、配对一致性、人工复核状态，`--finalize` 时把分片合并为最终 LeRobot 数据集。

## v3/

杯子双目标区域任务（`mug_on_blue` / `mug_on_yellow`）的 V3 数据采集、回放与验收工具。与 V2 不同，V3 把 schema、数据 IO、task_spec 与杯子环境紧耦合，在子包内独立维护。

### `task_spec.py` — 杯子 V3 任务标识和唯一 canonical 训练指令

- `TASK_PROMPTS`：`mug_on_blue` → `"Put the mug on the blue pad."`、`mug_on_yellow` → `"Put the mug on the yellow pad."`。
- `TASK_IDS` / `PROMPT_MODE="canonical"`：唯一允许写入数据集的指令集合与模式。
- `task_prompt(task_id)`：返回与任务标识严格对应的英文 canonical 指令。

### `select_seeds.py` — 基于真实稳定杯子位置确定性筛选 4×5 覆盖 seed

- 常量：`CANDIDATE_START=0`、`CANDIDATE_STOP=10_000`、`GRID_COLUMNS=4`、`GRID_ROWS=5`、`MIN_SELECTED_DISTANCE=0.04`，默认输出 `configs/mug_v3_seed_selection.json` 与 `.csv`。
- `SeedCandidate`：通过真实 `MugTabletopEnv.reset` 检查的候选 seed，记录稳定杯子中心 (x, y)、所属网格列/行、到网格中心距离和通过原因。
- `_grid_index` / `_cell_center`：把连续坐标稳定映射到 4×5 网格索引并计算单元中心。
- 流程：扫描候选 seed，在 `MugTabletopEnv` 中真实 reset 让物体稳定，按 4×5 网格选出离单元中心最近且满足最小间距的 20 个 seed，保证后续采集的 scene 覆盖完整且可复现。

### `collection_plan.py` — 杯子 V3 锁定配置、交替任务队列和严格恢复状态

- 常量：`MUG_REPO_ID="smolvla_ur10e_mug_v3"`、`MUG_DATASET_VERSION="smolvla_ur10e_mug_v3"`，以及与 V2 同名的 manifest/queue/progress/review/pilot 文件名，新增 `REDO_KEYS_FILENAME="redo_keys.txt"`、`EXPECTED_CAMERA_FEATURES`（LeRobot 视频键到 MuJoCo 相机名映射）。
- `MugCollectionConfig`：完整校验后的 V3 配置，包含 `scene_seeds`、`pilot_scene_seeds`、`tasks`、`canonical_prompts`、`fps` / `viewer_fps`、`max_frames`、`expected_total`、`strict_success_required`、`montage_speed`、`camera_features`、`seed_selection`、`fixed_pad_positions`（蓝黄区域固定坐标）、`snapshot` 与 `sha256`。
- `QueueItem`：与 V2 同形，按 scene 配对交替任务；`shard_name` 同样用 SHA-256 前 12 位保证 Windows 路径兼容。
- 工具函数：`load_config` / `build_plan` / `plan_for_mode` / `prepare_redo` / `record_completion` / `initialize_workspace` / `validate_completed_shards` / `atomic_write_json` / `utc_now` 等。

### `dataset_io.py` — 杯子 V3 专用 LeRobot schema、原子 episode 写入和分片验签

- 常量：`DATASET_FPS=20`、`CONTRACT_FILENAME`、`CAMERA_FEATURES`（与 `collection_plan.EXPECTED_CAMERA_FEATURES` 对齐）。
- `concatenate_video_files_utf8(...)`：与 `common.dataset_io` 同语义的 UTF-8 concat 合并，适配 V3 中文路径。
- `MugEpisodeWriter`：杯子 V3 原子 episode 写入器，负责 schema、视频编码、契约校验。
- `read_shard_table` / `vector_column` / `task_texts` / `validate_episode_shard` / `decode_video` / `configure_hf_datasets_cache`：分片读写、向量列转换、任务文本提取、分片验签和视频解码工具。

### `collect_matrix.py` — 杯子 V3 按 scene 配对的人工键盘矩阵采集入口

CLI 参数：`--config`（必填）、`--pilot`（仅两 scene 的 4 条 pilot）、`--resume`（严格校验后恢复工作区）、`--redo-key`（归档并重采一个已完成键）。

主循环与 V2 同构：按队列键采集，每条最多 400 帧，严格成功后 Enter 保存 / Backspace 重试同键；400 帧上限、取消或丢弃时按相同 queue key 重试。依赖 `common.control`、`common.state_machine`、`v3.collection_plan`、`v3.dataset_io.MugEpisodeWriter` 与 `sim.mug_environment.MugTabletopEnv`。

### `replay.py` — 不创建 MuJoCo 环境的杯子 V3 双相机 episode 回放入口

CLI 参数：`--root`（V3 LeRobot 数据集或单分片目录）、`--episode-index`（默认 0）。

通过 LeRobot 原生接口读取 V3 双相机视频（`agentview` / `d435i_rgb`）与 state/action，合成带任务文本、scene seed、帧号、状态和动作叠加的双相机画面，强校验图像 shape 为 `(256, 256, 3)` 后用 OpenCV 窗口播放。

### `build_review_montages.py` — 为每个杯子 scene 生成蓝黄任务并排人工复核视频

CLI 参数：`--config`（必填）、`--pilot`（仅两 pilot scene）、`--speed`（2 至 4 倍速）。

`TASK_LAYOUT = ("mug_on_blue", "mug_on_yellow")` 把两任务横向并排，按倍速解码每路 MP4（`_decode_sampled_video`），合成单个 scene 的蓝黄并排复核视频；`VALID_REVIEW_VALUES` 定义 `pending` / `pass` / `redo_mug_on_blue` / `redo_mug_on_yellow` 复核状态。

### `validate_dataset.py` — 杯子 V3 分片、确定性动作回放、人工复核与最终数据集验收

CLI 参数：`--config`（必填）、`--pilot`（仅验收 4 条 pilot）、`--finalize`（40 条 PASS 后物化最终 LeRobot 数据集）。

校验流程：

- `_read_review_status`：读取 `review_status.csv` 中 scene seed 到状态文本的映射。
- `_montage_is_valid`：检查蓝黄并排蒙太奇可解码且分辨率正确。
- 确定性动作回放：在 `MugTabletopEnv` 中按分片 action 序列回放，要求在原始帧数内复现成功（参考项目硬约束：超时即判定动作不可复现）。
- 配对一致性、`REDO_KEYS_FILENAME` 重采记录、分片契约校验。
- `--finalize` 时合并 40 条分片为最终 LeRobot 数据集，结果写入 `dataset_validation.json`。

## 跨版本调用关系

```
common/control.py ─────┐
common/state_machine.py┼── v1/collect.py ──────── v1/replay.py
common/dataset_io.py ──┘    │
                            ├── v2/collect_matrix.py
                            │   v2/collection_plan.py ──┐
                            │   v2/build_review_montages.py
                            │   v2/validate_grounding_dataset.py
                            │
                            └── v3/collect_matrix.py
                                v3/collection_plan.py ─┐
                                v3/task_spec.py ────────┤
                                v3/dataset_io.py ───────┤
                                v3/select_seeds.py ────┤
                                v3/replay.py ──────────┤
                                v3/build_review_montages.py
                                v3/validate_dataset.py ─┘
```

## 典型用法

### V1 单任务采集与回放

```bash
# 采集单条 episode
python -m collector.v1.collect --root outputs/dataset_v1 --task red_on_blue --episodes 1

# 回放 episode 0
python -m collector.v1.replay --root outputs/dataset_v1 --episode-index 0
```

### V2 颜色 Grounding 矩阵

```bash
# pilot 阶段：前两 scene 的 8 条
python -m collector.v2.collect_matrix --config configs/grounding_v2.yaml --pilot

# 生成 2×2 复核蒙太奇
python -m collector.v2.build_review_montages --config configs/grounding_v2.yaml --speed 3

# 全量验收并物化最终 80 条数据集
python -m collector.v2.validate_grounding_dataset --config configs/grounding_v2.yaml --finalize
```

### V3 杯子双目标区域任务

```bash
# 1. 筛选 4×5 覆盖 seed
python -m collector.v3.select_seeds

# 2. pilot 阶段：两 scene 的 4 条
python -m collector.v3.collect_matrix --config configs/mug_v3.yaml --pilot

# 3. 生成蓝黄并排复核蒙太奇
python -m collector.v3.build_review_montages --config configs/mug_v3.yaml --speed 3

# 4. 全量验收并物化最终 40 条数据集
python -m collector.v3.validate_dataset --config configs/mug_v3.yaml --finalize

# 5. 回放任意 episode
python -m collector.v3.replay --root outputs/dataset_mug_v3 --episode-index 0
```
```
