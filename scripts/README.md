# scripts

SmolVLA 项目维护与验证脚本。本目录聚合三类工具：评测产物的事后分析（抖动、多 seed 统计、chunk 衔接对照）、专家数据标定与场景迁移校验（关节运动上限、ACT 资源清单与空间布局一致性）、以及跨平台运维入口（云端环境引导、训练与冒烟测试、Windows 评测环境）。所有脚本均可从项目根目录独立调用，分析类脚本只读评测产物、不重跑 MuJoCo 闭环。

## 目录结构

| 路径 | 类型 | 功能 |
| --- | --- | --- |
| `__init__.py` | Python | 包初始化文件，声明本模块为「SmolVLA 项目维护与验证脚本」。 |
| `analyze_chunk_boundary_jitter.py` | Python | 从已有 action_traces 量化 chunk 边界对轨迹抖动的影响。 |
| `analyze_multi_seed.py` | Python | 多 policy_seed 评测结果的跨 seed 聚合与 Bootstrap 置信区间分析。 |
| `summarize_chunk_blend.py` | Python | 汇总 `--chunk-blend` 对照实验：跨 K 成功率与边界抖动指标。 |
| `calibrate_motion_limits.py` | Python | 从 LeRobot 专家数据标定 UR10e 关节速度与加速度上限。 |
| `generate_asset_manifest.py` | Python | 生成 ACT 场景迁移资源的 SHA-256 逐文件清单。 |
| `verify_act_layout.py` | Python | 验证迁移场景与 ACT 源场景的编译后世界位姿完全一致。 |
| `bootstrap_cloud.sh` | Bash | 创建可复现的 Ubuntu 云环境并执行 GPU/模型/EGL 预检。 |
| `check_server_environment.sh` | Bash | 汇总新云服务器的硬件、驱动与运行依赖（只读诊断）。 |
| `smoke_test.sh` | Bash | 串联环境检查、单步训练与 checkpoint 完整性检查。 |
| `train.sh` | Bash | 从项目根目录调用配置驱动的 SmolVLA 训练入口。 |
| `setup_eval_env.ps1` | PowerShell | 在 Windows 11 上创建 smolvla-eval conda 环境并安装评测依赖。 |

## 评测产物分析脚本

### `analyze_chunk_boundary_jitter.py` — chunk 边界抖动分析

从现有评测 `action_traces` JSONL 量化 chunk 衔接处的跳变，不重跑评测。SmolVLA 是动作 chunk 策略，每 `execution_horizon` 步重新预测一段动作；新旧 chunk 在衔接处不连续会产生周期性抖动。本脚本从三个层面量化：

1. 命令层（`executed_action`）：发给 MuJoCo 的最终命令在 chunk 衔接处的跳变。
2. 模型层（`model_output`）：策略原始输出的衔接跳变（排除 postprocessor）。
3. 物理层（`observation_state`）：实际状态轨迹的步间位移。

CLI 参数：`--eval-root`（默认 `outputs/eval`）、`--output-dir`（默认 `outputs/eval/chunk_boundary_analysis`）、`--min-rollouts`（每目录最少 trace 数，默认 1）。

核心函数：

- `load_records(trace_path)`：读取一条 rollout 的 action trace。
- `infer_horizon(records)`：从 `chunk_start` 标记推断 `execution_horizon`。
- `analyze_records(records, horizon)`：计算单 rollout 的边界/内部跳变比值、方向余弦、chunk 内位置分布等指标。
- `load_manifest_horizon(dir_path)` / `load_rollout_success(run_dir)`：从 `run_manifest.json` 与 `rollouts.jsonl` 读取 horizon 与成败映射。
- `wilcoxon_sign_test(differences)` / `_normal_cdf` / `_erf`：纯 numpy 的 Wilcoxon 符号秩检验（正态近似 + 连续性修正）。
- `per_dir_ratio_stats(per_rollout, field)`：逐 rollout 边界/内部比值的 `>1` 占比、中位数与显著性检验。

输出：`summary_by_dir.csv`（每目录逐轨迹中位数）、`per_position.json`（chunk 内位置分布）、`example_timeline.csv`（首条轨迹逐帧位移）、`hypothesis_stats.json`（统计检验与成败分组）。依赖 numpy + 标准库。

### `analyze_multi_seed.py` — 多 seed 聚合与 Bootstrap 置信区间

读取 `evaluate` 闭环评测的 `rollouts.csv`（必需列 `scene_seed` / `policy_seed` / `success`），输出 per-seed 成功率、总体 scene 分层 Bootstrap 95% CI、跨 seed 聚合与 K-seed 敏感性分类。

CLI 参数：`--input`（评测目录，自动读 `rollouts.csv`）与 `--csv`（直接指定文件）二选一必填；`--output`（默认输入同目录下 `analysis` 子目录）、`--repeats`（默认 10000）、`--bootstrap-seed`（默认 20260813）。

常量：`SCENE_BOOTSTRAP_REPEATS=10_000`、`SCENE_BOOTSTRAP_SEED=20260813`，与 `evaluate.rollout.bootstrap_success_ci` 口径一致（以 `scene_seed` 为聚类单元整组重采样）。

核心函数：

- `load_rollouts(csv_path)` / `_as_bool(value)`：读取并解析 CSV 关键列。
- `scene_bootstrap_ci(rows, repeats, bootstrap_seed)`：scene 整组重采样成功率 95% CI。
- `per_seed_stats(...)`：按 `policy_seed` 分组计算成功率及各自 CI。
- `cross_seed_summary(per_seed)`：N 个 seed 成功率的均值 ± 样本标准差、min/max、极差。
- `seed_sensitivity(rows)`：按 `(scene, task, prompt)` 三元组分类为稳定成功 / seed 敏感 / 稳定失败。
- `attribute_rates(rows, attribute)`：按 `task_id` / `prompt_type` 聚合成功率。
- `write_report(...)`：生成人工可读 Markdown 报告。

输出：`multi_seed_summary.json` 与 `multi_seed_report.md`。依赖 numpy + 标准库，不加载 MuJoCo 或 LeRobot。

### `summarize_chunk_blend.py` — chunk-blend 对照实验汇总

读取 `outputs/eval/chunk_blend` 下每个 `K*` 评测目录的 `rollouts.jsonl` 与 `action_traces/`，逐 K 输出成功率（含 scene Bootstrap 95% CI）与边界抖动指标，复用 `analyze_chunk_boundary_jitter` 的逐轨迹计算。

CLI 参数：`--chunk-blend-root`（默认 `outputs/eval/chunk_blend`）。

核心函数：

- `bootstrap_success_ci(scene_results, repeats=10_000)`：按 scene 整组重采样成功率 95% CI（rng 固定 20260813，与 evaluate 口径一致）。
- `collect_run(run_dir)`：聚合一个 K 目录的成功率、完成步数中位数、`control_exception` 计数、边界抖动指标与夹爪多余切换次数。
- `median(values)`：列表中位数工具。

输出：`summary_comparison.csv`（每 K 一行）与 `report.md`（含判读说明：`exec_jump_ratio`/`model_jump_ratio` 应回落至 ≤1、`dir_cos_ratio` 应回升接近 1、`boundary_dir_flip_fraction` 处理组应显著下降）。依赖 numpy、标准库与 `scripts.analyze_chunk_boundary_jitter`。

### `calibrate_motion_limits.py` — 关节运动上限标定

从 LeRobot 专家数据按 episode 内差分计算六关节速度与加速度上限，写出可被评测器锁定引用的 JSON（参考项目约定：运动上限参数由本脚本生成并存放于 `configs/motion_limits/*.json`，逐关节独立施加速度与加速度阈值）。

CLI 参数：`--dataset-root`（必填）、`--output`（必填）、`--quantile`（绝对导数分位数，默认 0.99）、`--margin`（分位数安全裕量，默认 1.1）。

常量：`JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]`。

核心函数：

- `sha256_dataset(root, parquet_paths)`：计算元数据与 Parquet 样本的稳定内容哈希。
- `load_episode_actions(root)`：读取 `meta/info.json` 与 `data/chunk-*/*.parquet`，按 episode 与帧序重建六维专家动作轨迹，校验 action 为七维且有限、`frame_index` 连续无重复。
- `calibrate(trajectories, fps, quantile, margin)`：按差分计算速度、二阶差分计算加速度，取分位数并乘裕量得到各关节上限。

输出 JSON 含 `velocity_limits_rad_s` / `acceleration_limits_rad_s2` / `dataset_sha256` / `fps` / `episode_count` 等字段。依赖 numpy 与 pyarrow。

## 场景迁移与校验脚本

### `generate_asset_manifest.py` — ACT 资源迁移清单

生成 SmolVLA `assets/mujoco` 目录相对 ACT 项目 `mode` 目录的逐文件 SHA-256 清单，标注每个资源是 `copied`（与源逐字节一致）还是 `modified`。

CLI 参数：`--asset-root`（必填，SmolVLA `assets/mujoco`）、`--source-root`（必填，ACT `mode` 目录）、`--output`（必填，JSON 输出路径）。

核心函数：

- `_sha256(path)`：分块（1 MiB）计算文件 SHA-256。
- `generate_manifest(asset_root, source_root, output_path)`：遍历 asset 目录，按 `source_overrides` 把 `scene.xml` / `mug_scene.xml` 映射到 ACT 的 `demo_scene.xml`、`mug_5/model_smolvla.xml` 映射到 `model_new.xml`、绿色纹理映射到 `image0.png`，逐文件比对源哈希。

输出含 `schema_version`、`generated_at`、`file_count`、`total_bytes` 与逐文件 `path` / `source_path` / `size` / `sha256` / `source_sha256` / `status`。

### `verify_act_layout.py` — ACT 空间布局等价性验证

比较 ACT 源场景和 SmolVLA 迁移场景编译后的世界位姿，要求机械臂基座、桌面、四路相机（`agentview` / `topview` / `sideview` / `d435i_rgb`）的平移、旋转与视场角完全一致。

CLI 参数：`--source-root`（必填，ACT `mode` 目录）、`--tolerance`（最大允许绝对误差，默认 1e-9）。

核心函数：

- `_build_source_model(source_root)`：在内存中兼容修复 ACT `demo_scene.xml`（删除杯盘 include、清理 `#` 文本说明、修正 `meshdir`），磁盘上 ACT 项目不被修改；编译为 `mujoco.MjModel`。
- `_reset_model(model)`：设置 `ACT_INITIAL_ARM_QPOS_DEG` 初始关节角并 `mj_forward`。
- `_pose_values(model, data)`：提取基座位姿、桌面 geom 位姿与四路相机位姿/fovy。
- `verify_layout(source_root, tolerance)`：逐项比较源与迁移场景，超差抛 `AssertionError`，否则返回含 `max_error` 与逐项 `errors` 的报告。

依赖 `mujoco`、`numpy` 与 `sim.environment`（`ACT_INITIAL_ARM_QPOS_DEG` / `ARM_JOINT_NAMES` / `CleanTabletopEnv`）。

## 运维入口脚本

### `bootstrap_cloud.sh` — Ubuntu 云环境引导

创建可复现的 Ubuntu SmolVLA 虚拟环境并执行 GPU、模型与 EGL 预检。

CLI 参数：`--python PATH`（指定解释器）、`--venv PATH`（默认 `.venv-cloud`）、`--torch-index-url URL`（默认南京大学镜像 cu126）、`--install-system-packages`（apt 安装 ffmpeg/libegl1/libgl1/libglvnd0/python3-venv）、`--skip-model-download`。

流程：自动探测 Python 3.10/3.11，校验 `nvidia-smi` 可用，创建 venv，固定安装 `pip==25.1.1` / `setuptools==80.9.0`，从指定 index 安装 `torch==2.7.0` / `torchvision==0.22.0`，按 `constraints.txt` 安装 `requirements-cloud.txt`，最后以 `MUJOCO_GL=egl` 调用 `cloud.bootstrap_check`（可 `--skip-model-download`）。受 `SMOLVLA_BOOTSTRAP_PYTHON` / `TORCH_INDEX_URL` 环境变量覆盖。

### `check_server_environment.sh` — 服务器环境只读诊断

汇总新云服务器的硬件、驱动与 SmolVLA 运行依赖，便于保存并回传诊断；只查询、不安装依赖、不下载模型、不修改系统。

CLI 参数：`--python PATH`（按显式参数 → `.venv-cloud` → 系统 python 顺序解析）、`--output REPORT.txt`（保存报告，相对路径基于项目根）。

报告分节：基本信息、操作系统、CPU、内存、磁盘、NVIDIA GPU 与驱动、CUDA 工具链、Python 与机器学习依赖（torch/torchvision/torchcodec/lerobot/mujoco/transformers/datasets/accelerate/av/imageio-ffmpeg 版本与 CUDA 可用性）、视频与图形运行库（ffmpeg、`MUJOCO_GL`、`libEGL/GLX/GL/cuda.so`）、项目数据与关键文件、快速结论（NVIDIA 驱动 / FFmpeg / Python / PyTorch CUDA / 数据集根的存在性）。

### `smoke_test.sh` — 云端冒烟测试

串联环境检查、单步训练与 checkpoint 完整性检查。使用 `SMOLVLA_PYTHON` 或 `.venv-cloud/bin/python`，设 `MUJOCO_GL=egl`，从项目根目录 `exec` 调用 `cloud.smoke_test` 并透传全部参数；找不到云端 Python 时提示先运行 `bootstrap_cloud.sh`。

### `train.sh` — 训练入口

从项目根目录调用配置驱动的 SmolVLA 训练入口。使用 `SMOLVLA_PYTHON` 或 `.venv-cloud/bin/python`，`exec` 调用 `cloud.train` 并透传全部参数（如 `--config` / `--smoke` / `--resume-from` 等，具体见 `cloud/README.md`）。

### `setup_eval_env.ps1` — Windows 评测环境

在 Windows 11 台式机（RTX 4060）上创建 `smolvla-eval` conda 环境并安装 SmolVLA 闭环评测依赖。关键点：PyPI / 清华镜像上的 torch 在 Windows 下是 CPU 版，CUDA 版必须先从 PyTorch 官方源（cu126）安装，再装其余依赖。

无命令行参数。五步流程：conda 创建 `smolvla-eval`（Python 3.10）→ 从官方 cu126 源安装 `torch==2.7.0` / `torchvision==0.22.0` → 从清华镜像安装 `requirements-eval.txt` → 验证 `torch` / `mujoco` / `lerobot` 及 CUDA 可用性 → 打印后续 `evaluate\run.ps1` 用法。脚本内提供阿里云 / 腾讯云镜像备选注释。

用法（项目根目录）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_eval_env.ps1
```

## 模块调用关系

```
运维入口
  bootstrap_cloud.sh ────── cloud.bootstrap_check
  smoke_test.sh ────────── cloud.smoke_test
  train.sh ──────────────── cloud.train
  setup_eval_env.ps1 ────── (conda 环境 + requirements-eval.txt，独立)
  check_server_environment.sh ── (只读系统/Python 探针，无项目内调用)

评测产物分析（只读 outputs/eval）
  analyze_chunk_boundary_jitter.py ── action_traces/*.jsonl + run_manifest.json + rollouts.jsonl
  analyze_multi_seed.py ──────────── rollouts.csv
  summarize_chunk_blend.py ──────── scripts.analyze_chunk_boundary_jitter + rollouts.jsonl + action_traces/
  calibrate_motion_limits.py ─────── LeRobot meta/info.json + data/chunk-*/*.parquet

场景迁移校验
  generate_asset_manifest.py ── assets/mujoco ↔ ACT mode 目录
  verify_act_layout.py ──────── sim.environment.CleanTabletopEnv + ACT demo_scene.xml
```

## 典型用法

### 评测产物分析

```bash
# chunk 边界抖动分析（默认读 outputs/eval 下所有 action_traces）
python -m scripts.analyze_chunk_boundary_jitter

# 多 seed 聚合与 Bootstrap 置信区间
python -m scripts.analyze_multi_seed --input outputs/eval/formal_020000_multiseed

# chunk-blend 对照实验汇总
python -m scripts.summarize_chunk_blend

# 从专家数据标定关节运动上限
python -m scripts.calibrate_motion_limits \
    --dataset-root smolvla-data/smolvla_ur10e \
    --output configs/motion_limits/smolvla_ur10e.json
```

### 场景迁移校验

```bash
# 生成 ACT 资源迁移清单
python -m scripts.generate_asset_manifest \
    --asset-root assets/mujoco \
    --source-root /path/to/mujoco-act-robotics/mode \
    --output outputs/act_asset_manifest.json

# 验证迁移场景与 ACT 源场景空间布局一致
python -m scripts.verify_act_layout \
    --source-root /path/to/mujoco-act-robotics/mode
```

### 云端运维

```bash
# Ubuntu 云环境引导（含 GPU/模型/EGL 预检）
bash scripts/bootstrap_cloud.sh --install-system-packages

# 服务器环境只读诊断并保存报告
bash scripts/check_server_environment.sh --output server_report.txt

# 冒烟测试（环境检查 + 单步训练 + checkpoint 完整性）
bash scripts/smoke_test.sh

# 配置驱动训练
bash scripts/train.sh --config configs/cloud_train.yaml
```

### Windows 评测环境

```powershell
# 创建并安装 smolvla-eval conda 环境
powershell -ExecutionPolicy Bypass -File scripts\setup_eval_env.ps1
```
