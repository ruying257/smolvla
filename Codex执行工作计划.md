
## 1. 目标与职责划分

本计划由当前 Codex 执行。目标是在 `F:\桌面\smolvla` 建立一个独立的 SmolVLA 项目：本机完成代码、可视化 MuJoCo 数据采集、数据质检和闭环效果评测；云端 Ubuntu 完成环境安装与 SmolVLA 微调。用户负责实际键盘遥操作采集，并在云端执行训练命令，将完整输出和checkpoint带回本机分析。

项目必须达到以下交付状态：将 `smolvla` 代码项目和已质检数据传到云端后，云端可执行环境初始化与训练；完整checkpoint下载后可在本机`smolvla-eval`环境执行闭环评测。两端均不依赖`mujoco-act-robotics`父目录。

## 2. 已确认边界

| 项目项 | 已确认决策 |
| --- | --- |
| 新项目路径 | `F:\桌面\smolvla` |
| 参考项目 | `F:\桌面\code_learn\mujoco-act-robotics` |
| 原项目改动 | 不修改 `mujoco-act-robotics` 的任何文件 |
| 资源复用 | 复制最小必要的 UR10e、夹爪、桌面、相机 XML/mesh/纹理和工具代码，不做路径引用或符号链接 |
| 本机采集 | Windows 或 Ubuntu 均可运行；首轮以本机带 GUI 的 MuJoCo 键盘遥操作采集为主 |
| 云端 | Ubuntu、可联网、具备 24GB 级 NVIDIA GPU；负责 SmolVLA 训练 |
| 模型 | 仅微调 `lerobot/smolvla_base`，不做 ACT 性能对比 |
| 任务 | 红/绿积木放到蓝/黄长方形底板，共 4 类英文组合指令 |
| 数据 | 至少 80 条专家示范；本机采集、回放和质检后再上传 |
| 评测 | 本机`smolvla-eval`执行；10个未见scene seed x 4类任务 x 3种措辞 x 固定policy seed 20260，共120次rollout |
| Seen场景对照 | 使用训练已见seed 0-5，执行4类canonical任务 x 固定policy seed 20260，共24次rollout；只与正式结果的canonical子集比较 |
| 采样规格 | 20 Hz；第三方与腕部两路 RGB 均为 256 x 256 |
| 观测状态 | 7 维：6 个当前关节角 + 1 个当前夹爪状态 |
| 策略动作 | 7 维：6 个绝对关节目标角 + 1 个夹爪指令 |
| 人工职责 | 用户负责实时键盘采集及成功 episode 的最终确认 |
| 云端协作 | Codex 提供脚本与命令；用户在云端运行并回传完整输出 |
| 效果口径 | 不预设首版成功率硬指标；工程链路完成不等于策略效果达标 |
| 不纳入范围 | ROS2、RL 后训练、真机、中文指令、多阶段任务、对原 ACT 项目的改造 |

## 3. 交付架构

```text
参考仓库 mujoco-act-robotics
        |
        | 复制最小必要资源和可复用逻辑，保留来源、许可证和修改说明
        v
本机 smolvla 项目
  |- 跨平台 collector：Windows/Ubuntu GUI 采集、回放、质检
  |- cloud：Ubuntu 环境检查和训练
  |- evaluate：Windows本机闭环rollout、结果汇总和文档
  |- scripts：数据打包/校验、云端初始化和训练
        |
        | Git：代码、资源、配置、脚本、依赖锁定
        | 人工传输：一个已校验的数据包及其 manifest
        v
云端 Ubuntu
  |- bootstrap_cloud.sh
  |- 验证数据包
  |- train.sh
  |- 下载完整checkpoint
        v
本机 Windows（smolvla-eval）
  |- python -m evaluate
  |- evaluate/run.ps1
```

### 3.1 代码与数据传输规则

- 代码、复制后的资源、配置、脚本和依赖文件使用 Git 同步。
- 专家数据、checkpoint 和结果视频不进入 Git。
- 数据可由用户手工传输，但**不得手工挑选单个数据文件**。本机必须先运行打包脚本，生成一个完整的数据归档文件和 `manifest.json`；云端上传后先运行校验脚本。
- `manifest.json` 至少包含数据集版本、文件清单、文件大小、SHA-256、episode 数、任务类别数量、LeRobot 版本和打包时间。
- 校验失败时训练脚本必须拒绝启动。

这样保留人工传输的操作方式，同时避免漏传视频、索引、metadata 或使用错误数据集版本。

## 4. 目标目录结构

```text
F:\桌面\code_learn\smolvla/
  assets/
    licenses/                   # 复制资产的来源、许可证和修改说明
    mujoco/                     # UR10e、夹爪、桌面、相机 XML/mesh/纹理
  sim/
    environment.py              # 双积木双底板 MuJoCo 环境
    task_spec.py                # 任务、语言模板、随机化和成功判定
    control.py                  # UR10e 关节控制、安全限位
  collector/
    collect.py                  # Windows/Ubuntu GUI 键盘采集入口
    replay.py                   # episode 回放
    validate_dataset.py         # 数据完整性和分布检查
  cloud/
    train.py                    # SmolVLA 云端训练入口
  evaluate/
    rollout.py                  # 本机无 GUI 的闭环 rollout
    common.py                   # checkpoint、动作和结果工具
    run.ps1                     # smolvla-eval 统一入口
    README.md                   # 评测框架、命令与指标
  scripts/
    package_dataset.py          # 生成归档与 manifest
    verify_dataset.py           # 云端校验数据归档
    bootstrap_cloud.sh          # Ubuntu 环境安装与预检
    train.sh
  configs/
    collection_windows.yaml
    collection_ubuntu.yaml
    cloud_train.yaml
    eval.yaml
    eval_standard.yaml
  requirements-collector.txt
  requirements-cloud.txt
  constraints.txt
  README.md
```

实际文件名可按现有仓库风格调整，但职责边界不可混淆：采集器不依赖云端路径；云端训练不依赖 GUI；所有路径经配置文件或 `pathlib` 解析，不写死 `F:\...`。

## 5. 阶段 P0：参考实现审计与项目脚手架

### 工作

- 阅读参考仓库的 `README.md`、`collect_data.py`、`deploy.py`、`mujoco_env/` 和 `mode/` 资源。
- 明确复制清单：UR10e、Robotiq 夹爪、桌面、D435i/相机及其所有间接依赖的 mesh/纹理/XML。
- 创建独立 `smolvla` Git 项目与上文目录骨架。
- 记录复制资源的来源路径、许可证和修改说明。
- 在本机 Windows 上新建采集器独立环境；不安装正式 SmolVLA 训练依赖。

### 验收

- 新项目不含到 `mujoco-act-robotics` 的相对路径、绝对路径、符号链接或运行时 import。
- `README.md` 说明参考资源来源、项目功能和独立运行边界。
- 本机可在新项目根目录运行基础依赖检查。

## 6. 阶段 P1：可独立运行的 MuJoCo 场景

### 工作

- 基于复制后的模型资源建立 UR10e、桌面、双相机、两个正方体积木和两个薄长方形底板场景。
- 积木颜色固定为 red/green，底板颜色固定为 blue/yellow；尺寸、质量和摩擦参数保持对称。
- 蓝、黄放置区域固定为 `[0.55, -0.22, 0.8005]` 和 `[0.55, 0.22, 0.8005]`；只随机红、绿积木的平面位置。
- 积木采样范围固定为 `x=[0.25,0.42]`、`y=[-0.35,0.35]`，中心间距至少12厘米；实现可复现 scene seed 和积木初始位姿记录。
- 实现严格成功判定：指定积木位于指定底板内缩区域、积木稳定、夹爪释放。
- 实现失败类别：抓错积木、放错底板、掉落/越界、超时、控制或仿真异常。

### 验收

- Windows viewer 可打开，显示 UR10e、桌面、双积木、双底板和双相机图像。
- 给定同一 seed 时场景一致；不同 seed 时位置分布满足非重叠约束。
- 不依赖参考仓库仍可启动场景。

## 7. 阶段 P2：跨平台采集子模块

### 工作

- 在 `collector/` 实现 Windows/Ubuntu 共用的 GUI 采集入口、键盘控制、episode 保存、取消重置和回放。
- 控制与数据采样固定为 20 Hz，Viewer 显示为 60 Hz；第三方 RGB 和腕部 RGB 均为 256 x 256。
- 观测固定为第三方 RGB、腕部 RGB、7 维当前状态和英文任务文本；7 维当前状态明确为 6 个当前关节角和 1 个当前夹爪状态。
- 动作固定为 7 维目标命令：6 个绝对关节目标角和 1 个夹爪指令。
- 机械臂形态与控制方式参考 ACT 项目：键盘产生末端位姿增量，经 IK 求解为关节目标角，再由 MuJoCo 执行；保存给策略学习的是上述 7 维目标动作。
- 任务固定为 4 个组合：

```text
Put the red cube on the blue pad.
Put the red cube on the yellow pad.
Put the green cube on the blue pad.
Put the green cube on the yellow pad.
```

- 每类任务使用两种等义训练表达并均衡分配：canonical 模板 `Put the {cube_color} cube on the {pad_color} pad.`，等义模板 `Place the {cube_color} cube onto the {pad_color} pad.`。
- 未见措辞模板 `Move the {cube_color} cube to the {pad_color} pad.` 不得进入训练集，仅保留给本机语言泛化评测。
- 采集器必须写入 task、scene seed、对象初始位姿、episode 标识和数据集版本。
- 用户负责实时键盘操作；采集器负责严格成功检测，并在保存前要求用户做最终确认。

### 验收

- Windows 可通过键盘完成至少一个完整 episode，保存后可回放。
- 同一采集代码在 Ubuntu GUI 环境下不应包含 Windows 专属路径或 API。
- 数据读取器能读取全部 feature key、shape、dtype 和 task 文本。

## 8. 阶段 P3：数据质检与可传输数据包

### 工作

- 实现全量回放检查、坏 episode 剔除、任务/模板/初始位姿分布统计。
- 正式训练集只接受通过严格成功判定且经用户确认的专家 episode；取消、失败和超时 episode 可保存在独立诊断目录，但不得进入正式训练数据包。
- 实现数据包打包脚本：只接受通过质检的数据集输入。
- 生成数据归档文件和 `manifest.json`；内容包括 SHA-256、文件清单、episode 数、4 类任务计数和版本信息。
- 实现云端校验脚本：解包后逐文件验签并检查 LeRobot metadata、图像数据和 episode 数。

### 正式数据目标

- 至少 80 条有效专家示范。
- 40 条来自 10 个共享 scene seed，每个 seed 执行全部 4 条指令。
- 40 条来自独立随机场景，每类任务各 10 条。
- 每类任务总计 20 条，两个训练模板均衡。

### 验收

- 数据集全量可回放。
- 打包后在本机解包并校验成功。
- 缺少任意受控文件、篡改文件或任务计数不平衡时，校验脚本给出明确失败原因。

## 9. 阶段 P4：云端可运行性准备

### 工作

- 编写 `requirements-cloud.txt` 和 `constraints.txt`，锁定 Python、PyTorch、CUDA 兼容的 LeRobot/SmolVLA 依赖。
- 编写 `bootstrap_cloud.sh`：创建虚拟环境、安装依赖、检查 GPU、MuJoCo、模型下载和 headless 渲染。
- 编写 `train.sh`：从 `lerobot/smolvla_base` 初始化，读取数据根目录和训练配置。
- 编写本机`evaluate/`入口：在`smolvla-eval`中读取checkpoint，运行无GUI rollout并导出CSV、JSON和视频。
- 所有脚本均接收相对路径或显式命令行参数；不得包含本机绝对路径。
- Codex 负责编写并解释云端命令；用户负责在云端实际运行，并将完整日志和输出回传给 Codex，供其继续定位或调整。

### 云端 smoke test

在正式上传 80 条数据前，使用 1 至 4 条采集数据验证：

1. 云端可安装依赖并加载 `lerobot/smolvla_base`；
2. 数据包可解包和校验；
3. 一批数据可完成模型前向；
4. 模型输出可转换为 7 维 UR10e 命令；
5. checkpoint可完整保存，并包含模型配置、权重和策略前后处理器；
6. checkpoint下载后可由本机评测入口重新加载和执行。

### 验收

- 云端只依赖 `smolvla` 目录、数据归档、网络和公开模型 ID。
- `bootstrap_cloud.sh`、`train.sh`与本机`evaluate/run.ps1`的命令和输出路径写入README。
- 任何数据校验失败、模型下载失败、GPU 不可用或 headless 渲染失败都应在 smoke test 阶段暴露，而非正式训练中途。

## 10. 阶段 P5：正式云端训练与本机评测

### 工作

- 手工传输经 P3 打包和校验的数据归档至云端；先运行校验脚本。
- 在 24GB 级 GPU 上从 `smolvla_base` 微调。先采用小 batch、梯度累积和保守的模块冻结策略；精确参数由 P4 smoke test 决定。
- 保存数据版本、训练配置、随机 seed、checkpoint、日志和 rollout 视频。
- 下载完整checkpoint，在本机使用10个未训练scene seed、4类任务、canonical/synonym/unseen三种措辞和固定`policy_seed=20260`执行120次闭环rollout。
- 每条最多400步；同时固定scene seed与Flow Matching采样使用的policy seed。
- 每条结果即时写入JSONL并支持manifest严格校验的断点续跑；正式结果使用scene分层Bootstrap计算95%置信区间。

### 输出指标

- 总体和分任务严格成功率、95% 置信区间；
- 抓错积木率、放错底板率、执行失败率；
- 平均执行步数；
- 平均/P95 推理延迟；
- 同场景不同指令下的目标遵从率与代表性视频。

### 完成与效果判定

- 首版不预设人为成功率硬指标，以固定矩阵上的120次评测建立真实基线。
- 数据、训练、checkpoint 重载和闭环评测链路完整可复现时，可判定工程链路完成；如果策略成功率较低，不得表述为策略效果达标。
- 策略效果不足时，根据分任务指标和失败分类进行一轮有边界的补采或调参，不得通过改变成功判定、scene seed 或统计口径制造更高成功率。

## 11. 必须防范的风险

| 风险 | 强制措施 |
| --- | --- |
| 原仓库资源未复制完整 | P1 必须在参考仓库不可见的情况下启动新场景 |
| Windows GUI 或键盘控制异常 | P2 前先验证 viewer、键盘事件、双相机和单 episode 保存 |
| Windows/Ubuntu 数据 schema 漂移 | 采集/云端共享同一数据 contract；打包 manifest 记录版本并由云端校验 |
| 人工传输漏文件或数据错版 | 只传单一数据归档和 manifest；云端验签失败即停止 |
| 本机无窗口渲染失败 | 在`smolvla-eval`中执行真实短rollout，不只import MuJoCo |
| SmolVLA 输入与数据不匹配 | P4 用真实小样本完成前向、短训练、checkpoint 加载和 rollout |
| 语言被视觉位置捷径替代 | 训练数据加入共享 scene seed 下四指令配对；评测使用反事实指令切换 |
| 数据不足或泛化差 | 先小数据打通，再补采失败位姿和反事实场景；不得只增加训练步数 |
| 任务范围膨胀 | 首版不加入 ROS2、RL、真机或多阶段操作 |

## 12. Codex 汇报与停止规则

当前 Codex 每阶段必须报告：修改/新增文件、执行命令、验证结果、遗留风险和下一阶段前提。

出现下列任一情况，停止推进并先定位根因：

- P1 新场景仍引用或依赖 `mujoco-act-robotics`；
- P2 采集数据无法回放或 feature schema 不完整；
- P3 数据包在本机自校验失败；
- 本机`smolvla-eval`无法完成一次真实前向和无窗口短rollout；
- 同一场景切换指令后策略行为不随对象/目标语义改变。

## 13. 非目标与表达边界

- 不把仿真 rollout 写成真机部署。
- 不把模型加载、loss 下降或单条成功视频写成 VLA 任务成功。
- 不声称 SmolVLA 优于 ACT，因为本项目不做该对比。
- 项目最终定位为：预训练 VLA 在 UR10e 仿真场景的语言条件策略适配、数据采集和可复现闭环验证。
