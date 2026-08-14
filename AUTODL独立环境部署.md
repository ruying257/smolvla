# AutoDL 独立 cu126 环境部署

本文适用于 AutoDL 镜像 `CUDA13.0-Torch2.7.0-Python3.11` v2 和 RTX 4090 24GB实例。镜像只提供 Ubuntu、Python 3.11与NVIDIA驱动；项目不会复用镜像预装的PyTorch/cu130，而是在数据盘项目目录内创建独立的`.venv-cloud`，安装项目锁定的PyTorch 2.7.0/cu126环境。

## 1. 准备服务器

建议配置：

- 单张RTX 4090 24GB；
- 数据盘至少100GB，推荐150GB；
- 代码、虚拟环境、Hugging Face缓存、数据和checkpoint全部放在`/root/autodl-tmp`；
- 不把上述内容放进30GB系统盘。

首次启动v2镜像后，在默认目录执行镜像提供的Python初始化：

```bash
bash setup_base.sh
uenv
python --version
```

这里只借用镜像的Python 3.11解释器创建新环境，不使用其已有PyTorch参与项目训练。

## 2. 上传两个压缩包

本机需要上传：

```text
smolvla-autodl-cu126-code.zip    # 代码、MuJoCo资源、配置和环境脚本
smolvla-data.zip                 # 当前4条episode的smoke数据
```

把两个文件上传到：

```text
/root/autodl-tmp/
```

服务器端解压：

```bash
mkdir -p /root/autodl-tmp/smolvla
unzip -q /root/autodl-tmp/smolvla-autodl-cu126-code.zip \
  -d /root/autodl-tmp/smolvla
mv /root/autodl-tmp/smolvla-data.zip \
  /root/autodl-tmp/smolvla/smolvla-data.zip

cd /root/autodl-tmp/smolvla
unzip -q smolvla-data.zip -d .
```

检查代码和smoke数据：

```bash
test -f scripts/bootstrap_cloud.sh
test -f configs/cloud_train.yaml
test -f smolvla-data/smolvla_ur10e/meta/info.json
test -d smolvla-data/smolvla_ur10e/data
test -d smolvla-data/smolvla_ur10e/videos
```

`smolvla-data.zip`当前只有4条episode，只用于环境与训练链路验证。正式训练前必须替换成完整的80条专家示范数据。

## 3. 设置数据盘缓存

每次开机后执行：

```bash
cd /root/autodl-tmp/smolvla
export HF_HOME=/root/autodl-tmp/hf-cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf-cache/hub
```

上述变量必须在环境初始化、smoke test和正式训练前设置，避免模型缓存写入系统盘。

## 4. 只读检查镜像环境

创建项目环境前先保存服务器报告：

```bash
cd /root/autodl-tmp/smolvla
bash scripts/check_server_environment.sh \
  --python "$(command -v python)" \
  --output outputs/server_environment_before_bootstrap.txt
```

报告中应至少确认：Python 3.11、RTX 4090 24GB、NVIDIA驱动可见、项目与数据目录存在。此时显示镜像的cu130属于正常现象，不是最终训练环境。

## 5. 创建项目独立环境

执行项目初始化脚本，并显式指定PyTorch官方cu126源：

```bash
cd /root/autodl-tmp/smolvla
bash scripts/bootstrap_cloud.sh \
  --python "$(command -v python)" \
  --torch-index-url https://download.pytorch.org/whl/cu126 \
  --install-system-packages
```

该命令会：

1. 安装FFmpeg、EGL、OpenGL和`python3-venv`系统库；
2. 创建`/root/autodl-tmp/smolvla/.venv-cloud`；
3. 安装PyTorch 2.7.0/cu126、torchvision 0.22.0；
4. 按`constraints.txt`安装LeRobot 0.4.4、TorchCodec 0.5.0、MuJoCo 3.6.0等依赖；
5. 检查GPU、显存、公开模型下载和MuJoCo EGL双相机渲染。

如果PyTorch官方源访问缓慢，可改用项目默认的南京大学cu126镜像：

```bash
bash scripts/bootstrap_cloud.sh \
  --python "$(command -v python)" \
  --install-system-packages
```

不要在已有的`.venv-cloud`上反复执行初始化。首次初始化中断时，优先把完整报错发回定位；确认需要重建后再删除该环境或换一个新的`--venv`路径。

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

`nvidia-smi`显示CUDA 13.0而`torch.version.cuda`显示12.6是正常的：前者是驱动支持能力，后者是项目PyTorch wheel自带的CUDA运行时。

## 7. 单步 smoke test

```bash
cd /root/autodl-tmp/smolvla
source .venv-cloud/bin/activate

bash scripts/smoke_test.sh \
  --dataset-root smolvla-data/smolvla_ur10e \
  --output-dir outputs/smoke-autodl-cu126
```

该命令依次验证真实数据读取、1-step训练和完整checkpoint保存。只有smoke test全部通过后才能上传80条正式数据并开始长训练。

## 8. 正式训练

替换为完整80条数据后，先打印最终命令：

```bash
cd /root/autodl-tmp/smolvla
source .venv-cloud/bin/activate

bash scripts/train.sh \
  --dataset-root smolvla-data/smolvla_ur10e \
  --config configs/cloud_train.yaml \
  --output-dir outputs/train/smolvla_ur10e \
  --dry-run
```

确认输出为FP16 AMP、batch size 1和20,000 steps后正式执行：

```bash
bash scripts/train.sh \
  --dataset-root smolvla-data/smolvla_ur10e \
  --config configs/cloud_train.yaml \
  --output-dir outputs/train/smolvla_ur10e
```

训练过程中保留：

```text
outputs/train/smolvla_ur10e.train.log
outputs/train/smolvla_ur10e/checkpoints/
```

训练结束后下载完整`pretrained_model/`目录，不得只下载`model.safetensors`。

## 9. 再次开机

环境创建成功后不需要重复安装。每次AutoDL重新开机只需：

```bash
cd /root/autodl-tmp/smolvla
export HF_HOME=/root/autodl-tmp/hf-cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf-cache/hub
source .venv-cloud/bin/activate
```

随后继续执行smoke test或训练命令即可。
