# AutoDL conda 环境部署（RTX 4090 / PyTorch 2.7.0 cu126 / SmolVLA）

本文基于本次 AutoDL 实例（`autodl-container-a5904787c3-a9362940`，Ubuntu 22.04，RTX 4090 24GB）的实际环境报告编写，目标是装好项目锁定环境并跑通 smoke 与正式训练。

## 0. 版本结论与服务器现状

**版本锁定（与本地评测机一致，来自 `constraints.txt`）**：

- PyTorch `2.7.0` / CUDA 12.6（官方 `cu126` wheel）；torchvision `0.22.0`、torchcodec `0.5.0`；
- LeRobot `0.4.4`、MuJoCo `3.6.0`、transformers `4.57.1`、accelerate `1.11.0`、numpy `2.2.6`；
- Python 3.10（用 conda 新建环境）。

**本次服务器报告要点**：

| 项目 | 状态 |
| --- | --- |
| GPU / 驱动 | RTX 4090 24GB，驱动 580.76.05（CUDA 13.0），compute 8.9 ✔ |
| conda | `/root/miniconda3` 存在；**base 为 Python 3.12.3**，自带 torch 2.8.0+cu128（**项目不使用**） |
| 项目代码 | 已上传：`/root/autodl-tmp/smolvla`，git `main` @ `737007b` ✔ |
| 数据集 | `smolvla-data/smolvla_ur10e_mug_v1`（meta/data/videos）**未上传** ✘ |
| ffmpeg / EGL | 缺失，需 apt 安装（bootstrap 会装） |
| nvcc / CUDA_HOME | 缺失 / 未设置，**无需处理**（cu126 wheel 自带 CUDA 运行时） |

结论：镜像的 conda base 是 Python 3.12（项目只支持 3.10/3.11），且 base 里的 torch 是 cu128 构建（与本地 cu126 评测环境不一致），不能直接复用。方案：**用 conda 新建 Python 3.10 环境，再在其上创建项目独立 `.venv-cloud`（cu126）**，与本地评测环境完全一致。

## 1. 准备服务器

- 数据盘 `/dev/md0` 当前 50GB、几乎全空；代码、虚拟环境、Hugging Face 缓存、数据和 checkpoint 全部放 `/root/autodl-tmp`，不放 30GB 系统盘；
- 50GB 足够 smoke 与 mug_v1 正式训练；磁盘紧张时在控制台扩容数据盘即可，不影响已装环境。

## 2. 上传数据集（代码已在）

代码已在 `/root/autodl-tmp/smolvla`，当前只缺数据集。在本机项目根目录把 `smolvla-data/smolvla_ur10e_mug_v1/`（`meta` + `data` + `videos`）打包（zip 顶层为 `smolvla_ur10e_mug_v1/`）：

```powershell
# 本机 PowerShell，在 F:\桌面\smolvla 下执行
Compress-Archive -Path smolvla-data\smolvla_ur10e_mug_v1 -DestinationPath smolvla-mug-v1-data.zip
```

上传到 `/root/autodl-tmp/` 后解压到 `smolvla/smolvla-data/` 并校验：

```bash
cd /root/autodl-tmp
mkdir -p smolvla/smolvla-data
unzip -q smolvla-mug-v1-data.zip -d smolvla/smolvla-data/

cd /root/autodl-tmp/smolvla
test -f smolvla-data/smolvla_ur10e_mug_v1/meta/info.json
test -d smolvla-data/smolvla_ur10e_mug_v1/data
test -d smolvla-data/smolvla_ur10e_mug_v1/videos
```

> 如果 zip 顶层带 `smolvla-data/` 前缀（例如用 `zip -r smolvla-mug-v1-data.zip smolvla-data/smolvla_ur10e_mug_v1` 打包），解压目标改为 `-d smolvla/`，然后执行同样的三条校验。

注意：本地 `smolvla_ur10e_mug_v1` 目前只有 `chunk-000`（1 条 episode 的 smoke 子集），只用于环境与训练链路验证；正式训练前必须替换为完整数据。

## 3. 用 conda 创建 Python 3.10 环境

镜像自带 conda（`/root/miniconda3`）。创建项目专用环境：

```bash
# 可选：conda 下载慢时先配置清华镜像
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/

conda create -y -n smolvla-cloud python=3.10 pip
```

确认环境解释器路径（后续 bootstrap 直接用它，不需要激活环境）：

```bash
/root/miniconda3/envs/smolvla-cloud/bin/python --version   # 期望 Python 3.10.x
```

> 不要使用镜像 base（Python 3.12 / torch 2.8.0+cu128）：项目锁定的依赖与本地评测均按 Python 3.10/3.11 + cu126 验证。
> 若系统盘空间紧张，可用前缀方式把环境建到数据盘：`conda create -y -p /root/autodl-tmp/conda-envs/smolvla-cloud python=3.10 pip`，后续命令中的环境 python 路径对应改为 `/root/autodl-tmp/conda-envs/smolvla-cloud/bin/python`。

## 4. 设置数据盘缓存

```bash
cd /root/autodl-tmp/smolvla
export HF_HOME=/root/autodl-tmp/hf-cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf-cache/hub
```

每个会话都要设置；环境初始化、smoke test 和正式训练前都必须已设置，避免模型缓存写入系统盘。

## 5. 创建项目独立环境（.venv-cloud，cu126）

PyPI 依赖下载慢时，先设置清华 pip 镜像（只影响普通依赖，torch 安装由 `--torch-index-url` 单独指定）：

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

然后执行 bootstrap（在 conda Python 3.10 之上完成全部安装）：

```bash
cd /root/autodl-tmp/smolvla
bash scripts/bootstrap_cloud.sh \
  --python /root/miniconda3/envs/smolvla-cloud/bin/python \
  --install-system-packages
```

该命令会：

1. apt 安装 ffmpeg、libegl1、libgl1、libglvnd0、python3-venv（当前是 root，脚本直接 apt-get）；
2. 创建 `/root/autodl-tmp/smolvla/.venv-cloud`；
3. 安装 PyTorch 2.7.0/cu126、torchvision 0.22.0（默认南京大学 cu126 镜像；官方源更慢时用下面命令）；
4. 按 `constraints.txt` 安装 LeRobot 0.4.4、TorchCodec 0.5.0、MuJoCo 3.6.0 等依赖；
5. 检查 GPU、显存、公开模型下载和 MuJoCo EGL 双相机渲染。

PyTorch 官方源安装（cu126）：

```bash
bash scripts/bootstrap_cloud.sh \
  --python /root/miniconda3/envs/smolvla-cloud/bin/python \
  --torch-index-url https://download.pytorch.org/whl/cu126 \
  --install-system-packages
```

不要在已有的 `.venv-cloud` 上反复执行初始化。首次初始化中断时，优先把完整报错发回定位；确认需要重建后再删除该环境或换一个新的 `--venv` 路径。

> 不要删除 conda 环境 `smolvla-cloud`：`.venv-cloud` 的 Python 解释器指向它（Linux venv 以符号链接方式复用基础解释器）。

## 6. 验证独立环境

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

`nvidia-smi` 显示 CUDA 13.0 而 `torch.version.cuda` 显示 12.6 是正常的：前者是驱动支持能力，后者是项目 PyTorch wheel 自带的 CUDA 运行时。`nvcc` 缺失、`CUDA_HOME` 未设置均无需处理。

## 7. 单步 smoke test（需要数据已上传）

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

> **WandB 提示**：训练固定开启 `wandb.enable=true`。如果服务器上未登录 wandb，训练可能卡在 "Create a W&B account" 之类的交互提示处，任选其一解决：
> - `wandb login` 后粘贴 API Key；
> - 或 `export WANDB_MODE=offline`（只在本地记录，不同步云端）。

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

## 10. conda 缺失时的 Miniforge 安装（本实例不需要）

本次实例已有 `/root/miniconda3`，跳过本节约 30 分钟。若换到没有 conda 的实例，才需要安装 Miniforge 到数据盘：

```bash
nvidia-smi   # 先确认驱动层可用（conda 无法修复驱动/GPU 问题）

command -v conda || ls /root/miniconda3/bin/conda /opt/conda/bin/conda 2>/dev/null

cd /root/autodl-tmp
wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /root/autodl-tmp/miniforge3
```

随后把第 3 节的 conda 命令换成 `/root/autodl-tmp/miniforge3/bin/conda`，环境 python 路径换成 `/root/autodl-tmp/miniforge3/envs/smolvla-cloud/bin/python`，其余步骤（第 4-9 节）不变。

## 11. 再次开机

环境创建成功后不需要重复安装。每次 AutoDL 重新开机只需：

```bash
cd /root/autodl-tmp/smolvla
export HF_HOME=/root/autodl-tmp/hf-cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf-cache/hub
source .venv-cloud/bin/activate
```

随后继续执行 smoke test 或训练命令即可。若实例被重装（系统盘 conda 被清空），按第 3、5 节重建环境。
