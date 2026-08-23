# cloud

SmolVLA 云端训练与环境检查模块。本目录封装了云端 GPU 环境预检、LeRobot 训练入口编排以及端到端 smoke test 流程，目标是在正式训练前尽早暴露环境与配置问题。

## 目录结构

| 文件 | 功能 |
| --- | --- |
| `__init__.py` | 包初始化文件，声明本模块为「SmolVLA 云端训练与环境检查模块」。 |
| `common.py` | 云端训练入口共享的配置、路径与日志工具。 |
| `bootstrap_check.py` | 检查云端 GPU、依赖、公开模型下载和 MuJoCo EGL 渲染。 |
| `train.py` | 根据项目 YAML 配置调用 LeRobot SmolVLA 训练入口。 |
| `smoke_test.py` | 编排一轮真实数据云端训练并检查 checkpoint。 |

## 各脚本说明

### `common.py`

为各训练入口提供共用基础设施：

- `PROJECT_ROOT`：项目根目录常量。
- `load_yaml_config(path)`：读取 YAML 配置并要求根节点为映射，缺失或格式错误会抛出明确异常。
- `resolve_path(value, base)`：将相对路径解析为相对于项目根目录的绝对路径，支持 `~` 展开。
- `format_command(command)`：把参数列表格式化为可复制的 POSIX 命令字符串。
- `run_logged(command, log_path, cwd)`：执行子进程，并把合并后的 stdout/stderr 同时写入终端与 UTF-8 日志文件。

### `bootstrap_check.py`

在正式训练前暴露环境问题的预检脚本，入口为 `python -m cloud.bootstrap_check`。

检查项：

- Python 版本必须为 3.10 或 3.11；
- 系统中存在 `ffmpeg`（否则提示使用 `bootstrap_cloud.sh --install-system-packages`）；
- PyTorch 可用 CUDA，且 GPU 显存不低于 `--minimum-vram-gb`（默认 14 GiB）；
- 从 HuggingFace 下载 `lerobot/smolvla_base` 模型快照（可用 `--skip-model-download` 跳过）；
- 在 Linux 下设置 `MUJOCO_GL=egl`，实例化 `sim.environment.CleanTabletopEnv`，验证 headless 渲染能输出 `agent` 与 `wrist` 两个 256×256×3 相机图像。

成功时打印 JSON 环境报告（Python、torch、CUDA、GPU 名称与显存、MuJoCo/LeRobot 版本、模型缓存路径等）。

### `train.py`

依据 YAML 配置构造并执行 `lerobot-train` 命令，入口为 `python -m cloud.train`。

主要参数：

- `--dataset-root`：LeRobot 数据集目录，默认 `smolvla-data/smolvla_ur10e_grounding_v2`；
- `--config`：训练 YAML 配置，默认 `configs/cloud_train_Tencent.yaml`；
- `--output-dir` / `--job-name`：覆盖配置中的输出目录与任务名；
- `--resume-from`：从已有训练输出的 checkpoint 目录（如 `outputs/train/xxx/checkpoints/last`）恢复训练，使用 `--resume=true` 完整恢复 step、优化器与调度器状态；与 `--smoke` 互斥；
- `--smoke`：强制 1 step、batch=1 的 smoke 训练；
- `--dry-run`：仅打印最终 `lerobot-train` 命令，不执行。

#### 新训练路径：`build_train_command()`

读取配置中的 `train` / `policy` / `dataset` 段，拼装 policy 路径、设备、AMP、视频后端、batch_size、steps、save_freq 等参数，关闭 `push_to_hub` 与 `eval`，开启 `wandb.enable=true`。训练输出目录已存在时会拒绝覆盖（dry-run 除外）。训练日志写入 `output_dir.parent/<name>.train.log`。

#### 恢复训练路径：`_build_resume_command()`

当 `--resume-from` 非空时走此分支（与 `--smoke` 互斥）。恢复机制：

- 使用 `--resume=true` + `--config_path=<checkpoint>/pretrained_model/train_config.json`；
- `TrainPipelineConfig.from_pretrained` 从 checkpoint 加载完整训练配置（含策略超参、优化器与调度器段），其余 CLI 参数作为覆盖项；
- `load_training_state` 恢复 step、优化器、调度器与 RNG 状态。

**调度器衔接注意事项**：调度器会按新总步数重建后再套用已保存状态，若总步数改变其自动缩放会导致恢复点学习率跳升。必须用配置中 `resume` 段的 `scheduler_num_warmup_steps` / `scheduler_num_decay_steps` 复刻原有效调度（如原运行 10000 步配置自动缩放为 warmup 333 / decay 10000），保证学习率平滑衔接。

调用前 `main()` 会校验 `<resume_from>/pretrained_model/train_config.json` 存在，缺失则抛出 `FileNotFoundError`。

### `smoke_test.py`

串联环境检查、单步训练与 checkpoint 验收的端到端 smoke 流程，入口为 `python -m cloud.smoke_test`。

流程：

1. 校验 `--dataset-root` 存在、`--output-dir` 不存在（避免混用旧产物）；
2. 默认先调用 `cloud.bootstrap_check`（`--skip-bootstrap-check` 可跳过）；
3. 调用 `cloud.train --smoke` 在 `output_dir/train` 下完成单步训练；
4. 在 `pretrained_model`、`checkpoints/last/pretrained_model` 及按编号倒序的数字 checkpoint 中，查找首个同时包含 `config.json`、`model.safetensors`、`policy_preprocessor.json`、`policy_postprocessor.json` 的目录；
5. 找到则打印通过的 checkpoint 路径，否则抛出 `RuntimeError`。

## 脚本调用关系

```
smoke_test.py
   ├──> bootstrap_check.py   (环境预检，可跳过)
   └──> train.py             (单步 smoke 训练)
          └──> common.py     (配置/路径/日志工具，被 train 与 smoke_test 共用)
```

## 典型用法

```bash
# 1. 仅做环境预检
python -m cloud.bootstrap_check

# 2. 只打印训练命令，不执行
python -m cloud.train --dry-run

# 3. 单步 smoke 训练
python -m cloud.train --smoke

# 4. 端到端 smoke：预检 + 单步训练 + checkpoint 验收
python -m cloud.smoke_test

# 5. 从已有 checkpoint 恢复训练（与 --smoke 互斥）
python -m cloud.train --resume-from outputs/train/smolvla_ur10e_grounding_v2/checkpoints/last
```