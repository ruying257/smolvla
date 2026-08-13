# SmolVLA UR10e MuJoCo 数据采集项目

本项目以 ACT 的 `mode/demo_scene.xml`、键盘控制和数据采集流程为参考，在项目内独立提供 UR10e 双积木语言任务环境、20 Hz专家采集和LeRobot v3数据回放。杯子和盘子已删除，运行时不依赖ACT项目路径。

## 场景布局

- 机械臂基座：`[1.0, 0.0, 0.8]`
- 桌面body：`[0.0, 0.0, 0.0]`
- 桌面geom中心：`[0.5, 0.0, 0.4]`
- 桌面半尺寸：`[1.0, 0.7, 0.4]`，上表面高度为 `0.8 m`
- 机械臂初始关节角：`[0°, -90°, 90°, -90°, -90°, 90°]`
- 蓝色区域：`[0.55, -0.22, 0.8005]`
- 黄色区域：`[0.55, 0.22, 0.8005]`

两个积木边长均为 `0.05 m`、质量均为 `0.05 kg`，并具有自由关节。每次 `reset(scene_seed)` 只随机积木的平面位置：`x=[0.25,0.42]`、`y=[-0.35,0.35]`，中心距离至少 `0.12 m`；同一seed严格复现。放置区域完整尺寸为 `0.16 × 0.12 × 0.002 m`，位置固定且只用于半透明视觉提示。

严格成功要求指定积木完整进入目标区域内缩5毫米后的范围、连续稳定0.5秒且夹爪已经释放。错误积木、错误区域、掉落越界、超时和控制异常分别分类。

## 环境准备

推荐从项目环境定义创建或更新采集环境。FFmpeg是LeRobot视频编码的必需依赖：

```powershell
conda env create -f environment-collector.yml
conda activate smolvla-collector
```

已有环境可执行：

```powershell
conda install -n smolvla-collector -c conda-forge ffmpeg
conda activate smolvla-collector
python -m pip install -r requirements-collector.txt
```

本机环境包含MuJoCo、LeRobot 0.4.4、OpenCV、PyAV和FFmpeg，不安装SmolVLA训练侧依赖。

## 打开场景

```powershell
python view_scene.py
```

主视角以及 `agentview`、`d435i_rgb`、`sideview` 三路固定相机均由MuJoCo直接渲染到同一GLFW窗口，不创建OpenCV窗口。关闭Viewer后程序正常退出。

短时GUI冒烟测试：

```powershell
python view_scene.py --max-seconds 5
```

只显示主视角：

```powershell
python view_scene.py --no-camera-panel
```

指定可复现积木布局：

```powershell
python view_scene.py --scene-seed 7
```

## 采集专家数据

采集一条红积木到蓝色区域的episode：

```powershell
python -m collector.collect `
  --root smolvla-data\smolvla_ur10e `
  --task red_on_blue `
  --seed 0 `
  --episodes 1
```

对已有数据集显式续采，并循环使用指定seed：

```powershell
python -m collector.collect `
  --root smolvla-data\smolvla_ur10e `
  --task green_on_yellow `
  --seeds 3,7,11 `
  --episodes 3 `
  --resume
```

控制键：

| 按键 | 功能 |
| --- | --- |
| `W/S`、`A/D`、`R/F` | 末端前后、左右、上下平移 |
| 方向键、`Q/E` | 末端旋转 |
| 空格 | 切换夹爪开合 |
| `Z` | 取消当前episode并使用同一seed重试 |
| `Enter` | 严格成功后确认保存 |
| `Backspace` | 严格成功后丢弃并使用同一seed重试 |
| `Esc` | 退出采集器；未确认缓冲不会保存 |

首次有效操作自动开始录制。控制与数据固定为20 Hz，Viewer保持60 Hz。每类任务的两种训练措辞由采集器按已保存数量自动均衡；未见措辞不会写入训练集。

## 回放数据

回放第0条episode：

```powershell
python -m collector.replay --root smolvla-data\smolvla_ur10e --episode-index 0
```

回放窗口同步显示第三方和腕部视频，以及任务、seed、7维状态和7维动作。空格暂停，左右方向键逐帧，`Q`或`Esc`退出；回放不会创建MuJoCo环境。

## Headless检查

```powershell
python view_scene.py --headless --steps 10 --scene-seed 9
```

输出包含模型规模、机器人与桌面布局、四个任务元素的状态，以及三路 `256×256×3 uint8` 图像摘要。加入两个free joint后，模型规模应为 `nq=28`、`nv=26`、`nu=7`。

## 自动验证

```powershell
python -m unittest discover -s tests -v
python -m scripts.verify_act_layout `
  --source-root ..\code_learn\mujoco-act-robotics\mode
```

位姿验证器只读取ACT的 `demo_scene.xml`，在内存中删除杯盘并修复资源路径，不修改参考项目。机器人基座、桌面和四路模型相机的世界位姿及视场角误差上限为 `1e-9`。

## 资源说明

- `assets/mujoco/`：项目独立运行所需的XML、mesh和纹理。
- `assets/licenses/SOURCE_AND_LICENSE.md`：资源来源和许可证说明。
- `assets/licenses/asset_manifest.json`：逐文件大小、SHA-256和复制状态。
- `sim/`：随机任务环境、严格成功判定及MuJoCo内部多相机Viewer。
- `collector/`：IK遥操作、采集状态机、LeRobot写入和视频回放。
- `scripts/`：资源清单和ACT位姿等价性验证。
- `tests/`：模型、任务物体、稳定性、相机和注释回归测试。

所有路径均由项目文件位置推导，不引用ACT项目的运行时路径，也没有符号链接。

## 云端 SmolVLA 训练（P4）

当前云端目标环境为Ubuntu 24.04、Python 3.11、Tesla T4 15 GiB。依赖锁定为PyTorch 2.7.0/cu126、torchvision 0.22.0、TorchCodec 0.5.0、LeRobot 0.4.4和MuJoCo 3.6.0。云端只负责环境检查与训练；训练完成后的checkpoint下载到本笔记本评测。

### 必须上传的内容

推荐通过Git上传完整项目，不手工挑选代码文件：

```text
/workspace/smolvla/
├── assets/                 # 完整XML、mesh、纹理和许可证
├── sim/                    # UR10e MuJoCo环境
├── collector/              # 数据契约与任务定义
├── cloud/                  # 云端训练和环境检查
├── evaluate/               # 本机闭环评测代码、入口和文档
├── scripts/                # Ubuntu入口脚本
├── configs/                # 云端训练和本机评测配置
├── tests/
├── requirements-cloud.txt
├── constraints.txt
└── README.md
```

数据目录放在`smolvla`项目文件夹内，但仍与Git代码分开传输，且不得单独挑选Parquet或视频：

```text
/workspace/smolvla/
└── smolvla-data/
    └── smolvla_ur10e/
        ├── meta/           # 包括collector_contract.json和LeRobot metadata
        ├── data/           # 全部Parquet
        └── videos/         # 两路相机的全部视频
```

smoke test上传包含1至4条episode的完整数据集；正式训练再替换为P3产出的80条数据。本阶段按已确认边界不做manifest/SHA-256或schema前置验签，训练入口只检查数据目录存在；不兼容数据会在LeRobot加载或模型前向阶段失败。

首次运行不需要上传基座模型、本机虚拟环境、Hugging Face缓存、本机`outputs/`、临时编码目录、checkpoint或Token文件。`lerobot/smolvla_base`由初始化脚本下载；公开模型无需Token，如确有需要只使用云端环境变量。

### 初始化云端环境

进入一台新服务器后，先运行只读环境查询，并把报告完整回传：

```bash
cd /workspace/smolvla
bash scripts/check_server_environment.sh \
  --output outputs/server_environment.txt
```

如果要查询指定Python（例如尚未创建项目虚拟环境时的镜像Python）：

```bash
bash scripts/check_server_environment.sh \
  --python "$(command -v python)" \
  --output outputs/server_environment_before_bootstrap.txt
```

该脚本不安装依赖、不下载模型、不修改系统，只汇总操作系统、CPU、内存、磁盘、GPU、驱动、CUDA、Python/PyTorch、FFmpeg、EGL库以及项目数据目录状态。未指定`--output`时只打印到终端。

确认硬件报告后，再初始化正式环境：

```bash
cd /workspace/smolvla
bash scripts/bootstrap_cloud.sh --install-system-packages
source .venv-cloud/bin/activate
```

初始化脚本安装锁定依赖，并检查`nvidia-smi`、PyTorch CUDA、显存、FFmpeg、公开模型下载和MuJoCo EGL双相机渲染。系统依赖已由云平台提供时可省略`--install-system-packages`。如需覆盖官方wheel源：

```bash
bash scripts/bootstrap_cloud.sh \
  --torch-index-url https://download.pytorch.org/whl/cu126
```

初始化脚本接受Python 3.10或3.11，并优先检测`python3.11`。如需指定当前镜像的解释器，可传入`--python "$(command -v python)"`；脚本不会擅自添加第三方APT源。

脚本不会安装或修改NVIDIA驱动。GPU、模型下载或EGL检查失败时必须先处理根因。

### 一键 smoke test

```bash
cd /workspace/smolvla
bash scripts/smoke_test.sh \
  --dataset-root smolvla-data/smolvla_ur10e \
  --output-dir outputs/smoke
```

该命令依次完成环境检查、真实数据加载、1-step训练，并检查生成的checkpoint是否包含模型配置、权重和策略前后处理器。闭环评测不在云端smoke中执行。

### 正式训练

先查看最终命令：

```bash
bash scripts/train.sh \
  --dataset-root smolvla-data/smolvla_ur10e \
  --config configs/cloud_train.yaml \
  --output-dir outputs/train/smolvla_ur10e \
  --dry-run
```

确认后移除`--dry-run`执行。训练从`lerobot/smolvla_base`初始化，输入和输出特征由数据集推断为两路相机、7维状态和7维动作，同时增加一路masked empty camera。Tesla T4配置固定为FP16 AMP和batch size 1；默认关闭WandB和Hub上传，不覆盖已有输出目录。15 GiB只保证优先验证P4 smoke链路，正式长训练如发生OOM仍需进一步冻结或改用24 GiB GPU。

## 本机 SmolVLA 闭环评测

评测固定在本笔记本的`smolvla-eval` Conda环境执行。正式矩阵为10个未见场景、4类任务、canonical/synonym/unseen三种措辞和2个policy seed，共240条；每条最多400步。评测代码和PowerShell入口统一位于`evaluate/`，YAML配置保留在`configs/`。

```powershell
conda activate smolvla-eval
.\evaluate\run.ps1 `
  --checkpoint outputs\train\smolvla_ur10e `
  --config configs\eval_standard.yaml `
  --output-dir outputs\eval\formal_020000 `
  --resume
```

入口同时固定场景seed与SmolVLA采样使用的policy seed，每条完成后即时写入JSONL并支持严格断点续跑。输出包含运行manifest、JSONL、CSV、汇总JSON、Markdown报告、视频保留清单和审计视频。完整实验设计、Bootstrap口径、冒烟及预实验命令见[本机模型效果评测文档](evaluate/README.md)。

### 从训练服务器下载checkpoint

不得只传`model.safetensors`，必须上传完整目录：

```text
pretrained_model/
├── config.json
├── model.safetensors
├── train_config.json
├── policy_preprocessor.json
├── policy_postprocessor.json
└── policy_*_processor.safetensors
```

评测入口会拒绝缺少配置、权重或处理器定义的checkpoint，避免使用错误归一化统计执行闭环。
