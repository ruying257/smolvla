# SmolVLA Mug（杯子）多 Seed 统计 + Bootstrap 置信区间实验

本实验在台式机（RTX 4060 / Windows 11 / `smolvla-eval` Conda 环境）上运行，对 **mug
（杯子）抓取放置任务** 的未见场景评测矩阵，用多个 `policy_seed` 重复闭环评测，并通过
**Scene 分层 cluster Bootstrap** 给出总体成功率的 95% 置信区间，同时量化 Flow
Matching 采样随机性对结果的影响。

> 评测对象是 `MugTabletopEnv`（UR10e + 2F85 夹爪抓取 mug 放到蓝/黄放置区），
> 不是双积木 cube 场景。

配套文件：

- `configs/eval_multi_seed.yaml`：N=3 个 seed 的 mug 评测配置（120 条 rollout）；
- `scripts/analyze_multi_seed.py`：跨 seed 聚合分析脚本，输出 `multi_seed_report.md`
  与 `multi_seed_summary.json`。

## 1. 背景与动机

笔记本 GTX 1650 在 FP16 推理下产生非确定性错误，单次 rollout 的成功/失败不可信。
即使硬件行为确定，SmolVLA 的 Flow Matching 在每次生成 Action Chunk 时都会采样随机
噪声，`policy_seed` 只固定该噪声的生成种子。因此单次评测结果同时受两类随机性影响：

1. **场景布局随机性**：`scene_seed` 决定杯子的初始位置（在桌面上落下稳定）；
2. **采样随机性**：`policy_seed` 决定 Flow Matching 噪声，进而影响动作轨迹。

本实验把 `policy_seed` 从"固定为 20260 的复现字段"升级为正式实验变量：同一场景、
任务在 N 个 seed 下重复执行，用 Bootstrap 把两类随机性一起纳入成功率区间估计。

> 说明：`policy_seed` 只固定采样噪声种子，不消除硬件 FP16 非确定性。若需评估硬件
> 层确定性，可对同一 seed 重复执行同条件推理。本实验通过统计手段量化总体波动，
> 不单独诊断硬件层。

## 2. 实验设计

### 2.1 评测矩阵

与 mug 未见场景基线（`configs/eval/mug_v1_unseen.yaml`）完全同构，仅扩展 policy seed
维度：

```text
20 个未见 scene seed（未参与训练，4×5 空间网格筛选）
× 2 个目标任务（mug_on_blue / mug_on_yellow）
× 1 种措辞（canonical，mug 环境唯一支持的训练措辞）
× N 个 policy seed
```

| 档位 | N | 总 rollout | 适用场景 |
| --- | ---: | ---: | --- |
| 快速档 | 3 | 120 | 先跑通全链路、预算有限（推荐先做） |
| 严谨档 | 5 | 200 | 需要更窄的区间与更强的统计结论 |

配置默认 N=3（`configs/eval_multi_seed.yaml`）；扩到严谨档只需把 `policy_seeds`
改为 `20260, 20261, 20262, 20263, 20264`，并换新输出目录。

### 2.2 Seed 选择

- `scene_seeds`：9201、3132、6943、2277、6343、6043、8846、3174、2076、5724、
  8320、9165、2721、9162、4175、8848、7149、7586、1323、8780，与存档 unseen
  基线完全一致，全部未见于训练；
- `policy_seeds`：20260 起连续 N 个。**20260 与存档 unseen 40 条基线
  （`outputs/eval/存档/mug_v1_s8000_h25_unseen_canonical`，成功率 85%）直接可比**，
  可作为该模型与历史结果的直接对照；
- 其余维度与基线完全一致：`fps=20`、`max_steps=360`、`execution_horizon=25`
  （每次预测 50 步、只执行 25 步后重规划）、成功条件（杯子中心进入目标区内缩 1 cm
  边界、保持直立稳定 0.5 秒并释放夹爪）。

### 2.3 实验键

```text
scene=<scene_seed>|task=<task_id>|prompt=canonical|policy=<policy_seed>
```

120 条全部唯一；`--resume` 按该键断点续跑。

## 3. 统计口径

### 3.1 总体成功率

```text
总体成功率 = 全部成功 rollout / 全部 rollout（120 或 200 条）
```

### 3.2 Scene 分层 cluster Bootstrap 95% CI（第一主指标）

- **聚类单元 = `scene_seed`**：同一 scene 的全部 rollout（含多个 seed、两个任务）
  作为一个 cluster 整体重采样，避免把同一布局的多条结果当作独立样本，否则会系统性
  低估区间宽度；
- 每次重采样按 scene 整组抽取（可重复），把选中 scene 的全部 rollout 合并求成功率；
- B = 10000 次，随机种子固定为 `20260813`（与 `evaluate/rollout.py` 的
  `bootstrap_success_ci` 完全一致，保证存档基线 40 条结果与本次 120 条结果可对比）；
- 取 2.5% / 97.5% 百分位作为 95% 置信区间。

该口径同时纳入场景布局随机性与采样随机性（cluster 内部包含不同 seed 的 rollout）。

### 3.3 Per-seed 统计

对每个 policy seed 单独计算成功率及其各自的 scene Bootstrap CI。用于观察：

- 不同采样种子下成功率是否稳定；
- 单个 seed 的结果是否落在总体区间内。

### 3.4 跨 seed 聚合

对 N 个 seed 的成功率计算：

```text
均值 ± 样本标准差（ddof=1）、min、max、极差（max - min）
```

标准差/极差量化 **Flow Matching 采样随机性**对成功率的扰动幅度。

### 3.5 Seed 敏感性（K-seed 三分法）

对每个 `(scene, task)` 组合统计 N 个 seed 下的成功数：

| 分类 | 条件 | 含义 |
| --- | --- | --- |
| 稳定成功 | N/N 成功 | 采样不敏感，结果可靠 |
| Seed 敏感 | 部分 seed 成功 | 采样噪声决定成败，需谨慎解读单次结果 |
| 稳定失败 | 0/N 成功 | 非采样问题，模型在该条件下系统性失败 |

> 现有 `evaluate/rollout.py` 的 `stability` 统计只支持 2-seed 三分法，N>2 时不计数；
> 本实验的 K-seed 敏感性由 `scripts/analyze_multi_seed.py` 独立计算补齐。

## 4. 台式机运行步骤

### 4.0 前置检查

```powershell
conda activate smolvla-eval
nvidia-smi                    # 确认 RTX 4060 与驱动版本
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**必须确认输出为 `2.7.0+cu126` 且 `True`**。PyPI/清华镜像在 Windows 下只有 torch
的 CPU 版（版本号无 `+cu126` 后缀、`cuda_available` 为 False）；CUDA 版必须从官方源
单独安装。若当前是 CPU 版，先修复环境（`environment-eval.yml` 与
`scripts/setup_eval_env.ps1` 已更新，重新执行脚本或在现有环境中执行）：

```powershell
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-eval.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

checkpoint 必须完整（缺一不可），从训练服务器/笔记本拷贝：

```text
pretrained_model/
├── config.json
├── model.safetensors
├── policy_preprocessor.json
└── policy_postprocessor.json
```

本实验使用 **008000 步 checkpoint**（与存档 unseen 基线同权重）：

```text
outputs/train/smolvla_ur10e_mug_v1_b8_s8000/checkpoints/008000/pretrained_model
```

### 4.1 冒烟（1 条 2 步）

```powershell
.\evaluate\run.ps1 `
  --checkpoint outputs\train\smolvla_ur10e_mug_v1_b8_s8000\checkpoints\008000\pretrained_model `
  --config configs\eval_multi_seed.yaml `
  --output-dir outputs\eval\multi_seed_smoke `
  --max-rollouts 1 --max-steps 2
```

只验证 checkpoint 加载、CUDA 前向、随机种子、视频与 manifest，不评价效果。

> 视频保留策略：评测入口默认**保留全部视频**（不再有 `--keep-all-videos` 参数）。
> 如需裁剪只保留失败视频与每个任务的首条成功视频，加 `--prune-videos`。本实验为了
> GitHub 主页展示保留全部视频，命令不加该参数即可。

### 4.2 计时外推

冒烟通过后跑 **4 条预实验**（1 个 scene × 2 任务 × 1 措辞 × 1 个 seed），用实际耗时
估算总时长：

```powershell
.\evaluate\run.ps1 `
  --checkpoint outputs\train\smolvla_ur10e_mug_v1_b8_s8000\checkpoints\008000\pretrained_model `
  --config configs\eval_multi_seed.yaml `
  --output-dir outputs\eval\multi_seed_pilot `
  --max-rollouts 4
```

```text
预计总时长 ≈ 4条实际耗时 / 4 × 120（或 200）
```

RTX 4060 上单条 rollout（360 步上限）通常为 1–3 分钟量级，120 条约数小时，建议整批
后台运行。预实验目录与正式目录分开，不并入正式结果。

### 4.3 正式运行（120 条 / N=3）

```powershell
.\evaluate\run.ps1 `
  --checkpoint outputs\train\smolvla_ur10e_mug_v1_b8_s8000\checkpoints\008000\pretrained_model `
  --config configs\eval_multi_seed.yaml `
  --output-dir outputs\eval\mug_v1_s8000_h25_unseen_canonical_multiseed
```

**断点续跑**：中断后使用完全相同的 checkpoint、配置、输出目录，追加 `--resume`：

```powershell
.\evaluate\run.ps1 `
  --checkpoint outputs\train\smolvla_ur10e_mug_v1_b8_s8000\checkpoints\008000\pretrained_model `
  --config configs\eval_multi_seed.yaml `
  --output-dir outputs\eval\mug_v1_s8000_h25_unseen_canonical_multiseed `
  --resume
```

运行开始后不得修改 checkpoint、seed、措辞、超时或成功条件；checkpoint、代码、
环境或配置变化时必须使用新输出目录。

### 4.4 跨 seed 聚合分析

```powershell
python scripts\analyze_multi_seed.py `
  --input outputs\eval\mug_v1_s8000_h25_unseen_canonical_multiseed `
  --output outputs\eval\mug_v1_s8000_h25_unseen_canonical_multiseed\analysis
```

产物：

```text
analysis/
├── multi_seed_report.md     # 人工可读：per-seed 表、总体 CI、敏感性、分任务
└── multi_seed_summary.json  # 机器可读：全部统计字段
```

## 4.5 GitHub 主页视频展示

评测默认保留全部 rollout 视频（`outputs\eval\...\videos\*.mp4`），但项目 `.gitignore`
同时忽略 `outputs/` 与 `*.mp4`，**视频默认不会进入 Git 仓库**。若要用于 GitHub 主页
展示，三选一：

1. **精选视频入库（推荐少量）**：把要展示的成功/失败样例复制到非忽略目录（如
   `assets/demo/`），并修改 `.gitignore` 允许该目录的 mp4：

   ```gitignore
   # 允许主页演示视频（放在 .gitignore 末尾）
   !assets/demo/
   assets/demo/*
   !assets/demo/*.mp4
   ```

   然后在 README 中以相对链接引用：

   ```markdown
   ## Demo

   - [成功：mug_on_blue](assets/demo/mug_on_blue_success.mp4)
   - [失败：mug_on_yellow](assets/demo/mug_on_yellow_failure.mp4)
   ```

   GitHub 的 blob 页面支持直接预览 mp4，点击链接即可播放。

2. **GitHub Releases**：把完整 `videos/` 打包 zip 传到 Release 附件，README 给出下载
   链接，适合保留全部视频且不撑大仓库。

3. **外链托管**：上传到 YouTube/图床，README 用普通链接或 `<video>` 嵌入。

> 注意：每次 `git push` 前先确认 `.gitignore` 规则，避免误把 GB 级视频推入仓库。

## 5. 结果检查

```powershell
$rows = Import-Csv outputs\eval\mug_v1_s8000_h25_unseen_canonical_multiseed\rollouts.csv
$rows.Count                                   # 应为 120
($rows.rollout_key | Sort-Object -Unique).Count   # 应为 120
$rows.policy_seed | Sort-Object -Unique       # 应为 20260,20261,20262
Get-Content outputs\eval\mug_v1_s8000_h25_unseen_canonical_multiseed\summary.json -Encoding UTF8
Get-Content outputs\eval\mug_v1_s8000_h25_unseen_canonical_multiseed\analysis\multi_seed_report.md -Encoding UTF8
```

正式结果必须同时满足：120 条、120 个唯一实验键、20 个 scene seed、2 个目标任务、
canonical 措辞、3 个 policy seed、无 `control_exception`。

## 6. 结果记录模板

将 `analysis\multi_seed_report.md` 的数值填入下表（示例列，跑完替换）：

### 6.1 总体与跨 seed

| Checkpoint | Rollout | 总体成功率 | Scene Bootstrap 95% CI | 跨 seed 均值±std | min–max | 极差 |
| --- | ---: | ---: | --- | --- | --- | ---: |
| 008000 | 120 | `__%` | `[__%, __%]` | `__% ± __%` | `[__%, __%]` | `__%` |

> 存档基线（40 条、单 seed）：成功率 85.00%，Scene Bootstrap 95% CI [72.5%, 95.0%]。
> 本次 20260 seed 的 40 条子集应与其一致（同权重、同矩阵、同 horizon），用于跨机器
> 一致性校验（笔记本 GTX 1650 vs 台式机 RTX 4060）。

### 6.2 Per-seed 成功率

| policy seed | 成功数 | 总数 | 成功率 | Scene Bootstrap 95% CI |
| ---: | ---: | ---: | ---: | ---: |
| 20260 |  | 40 | `__%` | `[__, __]` |
| 20261 |  | 40 | `__%` | `[__, __]` |
| 20262 |  | 40 | `__%` | `[__, __]` |

### 6.3 Seed 敏感性

| 稳定成功（N/N） | Seed 敏感 | 稳定失败（0/N） | 不完整分组 |
| ---: | ---: | ---: | ---: |
| `__` | `__` | `__` | 0 |

### 6.4 分任务（跨 seed 聚合）

| 任务 | 成功数 | 总数 | 成功率 |
| --- | ---: | ---: | ---: |
| mug_on_blue |  | 60 | `__%` |
| mug_on_yellow |  | 60 | `__%` |

### 6.5 结论填写（示例句式）

- 模型在 20 个未见场景上的杯子放置总体严格成功率为 `__%`（95% CI `[__%, __%]`，
  Scene 分层 Bootstrap，B=10000）；
- 3 个采样 seed 下成功率波动 `__%`（极差），说明 Flow Matching 采样随机性
  [可忽略 / 不可忽略]；
- 稳定失败条件 `__` 个（`__%`），采样不敏感条件下模型[具备 / 不具备]可靠完成能力；
- 与存档基线（85%，单 seed）相比，多 seed 区间 [是否] 覆盖基线点估计，说明
  [采样随机性对结论的影响程度]。

## 7. 结论边界

- 结果只代表当前 checkpoint 在本机 MuJoCo 仿真闭环中的能力，不代表真实 UR10e 成功率；
- Bootstrap 区间反映场景布局与采样随机性，**不消除硬件 FP16 非确定性**；
- 120/200 条结果不能与缺少基座或其他 checkpoint 对照时宣称"训练带来提升"；
- 杯子任务只有 canonical 措辞，本实验不涉及语言泛化（synonym/unseen）维度；
- unseen 只代表未见布局泛化，不代表新任务组合泛化。

## 8. 文件清单（本次新增/修正）

| 文件 | 说明 |
| --- | --- |
| `configs/eval_multi_seed.yaml` | mug 未见场景 N=3 多 seed 评测配置 |
| `scripts/analyze_multi_seed.py` | 跨 seed 聚合 + Bootstrap 分析脚本（通用，cube/mug 均适用） |
| 本文档 | mug 实验设计与操作手册 |
