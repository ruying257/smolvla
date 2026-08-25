# AutoDL 独立 cu126 环境部署（镜像 CUDA13.0-Torch2.7.0-Python3.11 v2）

本文适用于 AutoDL 镜像 `CUDA13.0-Torch2.7.0-Python3.11` v2（镜像地址：<https://www.autodl.art/i/NVIDIA/cuda-samples/CUDA13.0-Torch2.7.0-Python3.11/2457/2>）和 RTX 4090 24GB 实例。

**环境版本结论（镜像选择用）**：

- PyTorch `2.7.0`，CUDA 12.6 构建（官方 `cu126` wheel）；torchvision `0.22.0`、torchcodec `0.5.0`；
- LeRobot `0.4.4`、MuJoCo `3.6.0`、transformers `4.57.1`、accelerate `1.11.0`、numpy `2.2.6`（完整锁定清单见 `constraints.txt`）；
- Python 3.10 或 3.11（镜像自带 Python 3.11 即可用）。

镜像只提供 Ubuntu、Python 3.11 与 NVIDIA 驱动。**项目不复用镜像预装的 PyTorch/cu130**，而是在数据盘项目目录内创建独立的 `.venv-cloud`，安装项目锁定的 PyTorch 2.7.0/cu126 环境。该构建与本地 RTX 4060 评测机（同一 `cu126` 源安装）一致，保证 checkpoint 在云端训练与本机评测两端兼容。

## 1. 准备服务器

建议配置：

- 单张 RTX 4090 24GB；
- 数据盘至少 100GB，推荐 150GB；
- 代码、虚拟环境、Hugging Face 缓存、数据和 checkpoint 全部放在 `/root/autodl-tmp`；
- 不把上述内容放进 30GB 系统盘。

首次启动该镜像后，在默认目录执行镜像提供的 Python 初始化：

```bash
bash setup_base.sh
uenv
python --version   # 期望 Python 3.11
```

这里只借用镜像的 Python 3.11 解释器创建新环境，不使用其已有 PyTorch 参与项目训练。

> 如果 `setup_base.sh` / `uenv` 不存在、失败或镜像 Python 不可用，不要停在原地：直接跳到[第 10 节 conda 回退](#10-镜像-python-不可用时的-conda-回退)，用 Miniforge 提供干净的 Python 3.10，其余步骤完全相同。注意 conda 只解决 Python 层，驱动/GPU 问题必须先过 `nvidia-smi` 门禁。

## 2. 上传两个压缩包

本机需要上传：

```text
smolvla-autodl-cu126-code.zip    # 代码、MuJoCo 资源、配置和环境脚本（Git 拉取或整目录打包）
smolvla-mug-v1-data.zip          # mug_v1 数据集（meta + data + videos，单独打包）
```

把两个文件上传到 `/root/autodl-tmp/`，服务器端解压：

```bash
mkdir -p /root/autodl-tmp/smolvla
unzip -q /root/autodl-tmp/smolvla-autodl-cu126-code.zip \
  -d /root/autodl-tmp/smolvla
mv /root/autodl-tmp/smolvla-mug-v1-data.zip \
  /root/autodl-tmp/smolvla/smolvla-mug-v1-data.zip

cd /root/autodl-tmp/smolvla
unzip -q smolvla-mug-v1-data.zip -d .
```

检查代码和数据集：

```bash
test -f scripts/bootstrap_cloud.sh
test -f configs/train/mug_b8_s8000.yaml
test -f smolvla-data/smolvla_ur10e_mug_v1/meta/info.json
test -d smolvla-data/smolvla_ur10e_mug_v1/data
test -d smolvla-data/smolvla_ur10e_mug_v1/videos
```

注意：

- `smolvla-data/` 与 `configs/0-legacy/` 都在 `.gitignore` 里，Git 传输不会带上：数据必须单独打包；正式训练必须使用被跟踪的 `configs/train/` 配置（`configs/0-legacy/` 下的旧配置不会出现在服务器上）。
- 本地 `smolvla-data/smolvla_ur10e_mug_v1` 当前只有 `chunk-000`（单条 episode 的 smoke 子集），只用于环境与训练链路验证；正式训练前必须替换为完整 mug_v1 数据。

## 3. 设置数据盘缓存

每次开机后执行：

```bash
cd /root/autodl-tmp/smolvla
export HF_HOME=/root/autodl-tmp/hf-cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf-cache/hub
```

上述变量必须在环境初始化、smoke test 和正式训练前设置，避免模型缓存写入系统盘。

## 4. 只读检查镜像环境

创建项目环境前先保存服务器报告：

```bash
cd /root/autodl-tmp/smolvla
bash scripts/check_server_environment.sh \
  --python "$(command -v python)" \
  --output outputs/server_environment_before_bootstrap.txt
```

报告中应至少确认：Python 3.11、RTX 4090 24GB、NVIDIA 驱动可见、项目与数据目录存在（`configs/train/mug_b8_s8000.yaml`、`smolvla-data/smolvla_ur10e_mug_v1/`）。此时显示镜像的 cu130 属于正常现象，不是最终训练环境。

## 5. 创建项目独立环境

执行项目初始化脚本，并显式指定 PyTorch 官方 cu126 源：

```bash
cd /root/autodl-tmp/smolvla
bash scripts/bootstrap_cloud.sh \
  --python "$(command -v python)" \
  --torch-index-url https://download.pytorch.org/whl/cu126 \
  --install-system-packages
```

该命令会：

1. 安装 FFmpeg、EGL、OpenGL 和 `python3-venv` 系统库；
2. 创建 `/root/autodl-tmp/smolvla/.venv-cloud`；
3. 安装 PyTorch 2.7.0/cu126、torchvision 0.22.0；
4. 按 `constraints.txt` 安装 LeRobot 0.4.4、TorchCodec 0.5.0、MuJoCo 3.6.0 等依赖；
5. 检查 GPU、显存、公开模型下载和 MuJoCo EGL 双相机渲染。

如果 PyTorch 官方源访问缓慢，可改用项目默认的南京大学 cu126 镜像：

```bash
bash scripts/bootstrap_cloud.sh \
  --python "$(command -v python)" \
  --install-system-packages
```

不要在已有的 `.venv-cloud` 上反复执行初始化。首次初始化中断时，优先把完整报错发回定位；确认需要重建后再删除该环境或换一个新的 `--venv` 路径。

## 6. 验证独立环境

激活项目环境：

```bash
source /root/autodl-tmp/smolvla/.venv-cloud/bin/activate
```

检查版本：

```bash
python - <<'PY'
import torch
import torchvision

print("python torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
PY

python -m pip check
```

目标结果：

```text
torch: 2.7.0+cu126
torchvision: 0.22.0+cu126
torch CUDA: 12.6
CUDA available: True
GPU: NVIDIA GeForce RTX 4090
```

`nvidia-smi` 显示 CUDA 13.0 而 `torch.version.cuda` 显示 12.6 是正常的：前者是驱动支持能力，后者是项目 PyTorch wheel 自带的 CUDA 运行时。

## 7. 单步 smoke test

```bash
cd /root/autodl-tmp/smolvla
source .venv-cloud/bin/activate

bash scripts/smoke_test.sh \
  --dataset-root smolvla-data/smolvla_ur10e_mug_v1 \
  --output-dir outputs/smoke-autodl-cu126
```

该命令依次验证真实数据读取、1-step 训练和完整 checkpoint 保存。只有 smoke test 全部通过后才能上传完整 mug_v1 数据并开始长训练。

## 8. 正式训练

替换为完整 mug_v1 数据后，先打印最终命令：

```bash
cd /root/autodl-tmp/smolvla
source .venv-cloud/bin/activate

bash scripts/train.sh \
  --dataset-root smolvla-data/smolvla_ur10e_mug_v1 \
  --config configs/train/mug_b8_s8000.yaml \
  --dry-run
```

确认输出为 FP16 AMP、batch size 8、10,000 steps、`repo_id=smolvla_ur10e_mug_v1` 后正式执行：

```bash
bash scripts/train.sh \
  --dataset-root smolvla-data/smolvla_ur10e_mug_v1 \
  --config configs/train/mug_b8_s8000.yaml
```

说明：

- `configs/train/mug_b8_s8000.yaml` 的 `train.steps=10000`、`batch_size=8`（文件名里的 "s8000" 是腾讯云时期遗留命名，不是步数）。输出目录与任务名由配置给出（`outputs/train/smolvla_ur10e_mug_v1_b8_s8000`）；如需新命名，追加 `--output-dir` 与 `--job-name`。
- 训练过程中保留：

```text
outputs/train/smolvla_ur10e_mug_v1_b8_s8000.train.log
outputs/train/smolvla_ur10e_mug_v1_b8_s8000/checkpoints/
```

训练结束后下载完整 `pretrained_model/` 目录，不得只下载 `model.safetensors`。

## 9. 从腾讯云 checkpoint 恢复训练（应对训练中断）

腾讯云训练中断或迁移时，不要重头训练：从已有 checkpoint 完整恢复 step、优化器、调度器与 RNG。

先上传旧 checkpoint 的完整目录（`checkpoints/` 含 `last/` 与各数字 checkpoint，每个 checkpoint 内必须有完整的 `pretrained_model/`），例如放到：

```text
/root/autodl-tmp/smolvla/checkpoints/smolvla_ur10e_mug_v1_b8_s8000/
```

恢复命令（`--resume-from` 指向 `checkpoints/last`）：

```bash
cd /root/autodl-tmp/smolvla
source .venv-cloud/bin/activate

bash scripts/train.sh \
  --dataset-root smolvla-data/smolvla_ur10e_mug_v1 \
  --config configs/train/mug_b8_s12000_resume.yaml \
  --resume-from /root/autodl-tmp/smolvla/checkpoints/smolvla_ur10e_mug_v1_b8_s8000/checkpoints/last
```

要点：

- `cloud/train.py` 的 `--resume-from` 走 `--resume=true` + `--config_path=<checkpoint>/pretrained_model/train_config.json`，完整恢复 step、优化器、调度器与 RNG；
- `mug_b8_s12000_resume.yaml` 中硬编码的 `resume.checkpoint: /workspace/...` 只是说明性字段，实际路径一律以 `--resume-from` 为准；`resume.scheduler_num_warmup_steps=333` / `scheduler_num_decay_steps=10000` 用于复刻原 10000 步调度，避免恢复点学习率跳升；
- 恢复模式下 `train.steps` 表示【总步数】：`mug_b8_s12000_resume.yaml` 的 `steps=12000` 表示从恢复点续训到累计 12000 步；
- DR 微调配置 `configs/train/mug_b8_s8000plus3000_dr.yaml` 从 `--policy.path` 加载 s8000 checkpoint（优化器/调度器全新、LR 重启），需要 s8000 checkpoint 已在服务器上；
- 恢复前确认数据集与训练时一致（`repo_id=smolvla_ur10e_mug_v1`），且 `--dataset-root` 指向完整数据。

## 10. 镜像 Python 不可用时的 conda 回退

如果镜像首启的 `setup_base.sh` / `uenv` 失败，或镜像 Python 不可用，用 Miniforge 提供干净的 Python 3.10：

先确认驱动层可用（conda 无法修复驱动/GPU 问题）：

```bash
nvidia-smi
```

检查是否已有 conda；无则安装 Miniforge 到数据盘：

```bash
command -v conda || ls /root/miniconda3/bin/conda /opt/conda/bin/conda 2>/dev/null

cd /root/autodl-tmp
wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /root/autodl-tmp/miniforge3
```

创建项目 Python 环境：

```bash
source /root/autodl-tmp/miniforge3/etc/profile.d/conda.sh
conda create -y -n smolvla-cloud python=3.10 pip
```

用与第 5 节相同的 bootstrap 脚本，`--python` 指向 conda 环境解释器：

```bash
cd /root/autodl-tmp/smolvla
bash scripts/bootstrap_cloud.sh \
  --python /root/autodl-tmp/miniforge3/envs/smolvla-cloud/bin/python \
  --install-system-packages
```

bootstrap 会在该 Python 之上创建 `.venv-cloud` 并安装 cu126 依赖；此后第 6-9 节的验证、smoke、训练与恢复命令完全不变（脚本默认使用 `.venv-cloud/bin/python`）。

## 11. 再次开机

环境创建成功后不需要重复安装。每次 AutoDL 重新开机只需：

```bash
cd /root/autodl-tmp/smolvla
export HF_HOME=/root/autodl-tmp/hf-cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf-cache/hub
source .venv-cloud/bin/activate
```

随后继续执行 smoke test 或训练命令即可。
