# SmolVLA 多 Seed 统计 + Bootstrap 置信区间实验

本实验在台式机（RTX 4060 / Windows 11 / `smolvla-eval` Conda 环境）上运行，用多个
`policy_seed` 对同一正式评测矩阵重复闭环评测，并通过 **Scene 分层 cluster Bootstrap**
给出总体成功率的 95% 置信区间，同时量化 Flow Matching 采样随机性对结果的影响。

配套文件：

- `configs/eval_multi_seed.yaml`：N=3 个 seed 的评测配置（360 条 rollout）；
- `scripts/analyze_multi_seed.py`：跨 seed 聚合分析脚本，输出 `multi_seed_report.md`
  与 `multi_seed_summary.json`。

## 1. 背景与动机

笔记本 GTX 1650 在 FP16 推理下产生非确定性错误，单次 rollout 的成功/失败不可信。
即使硬件行为确定，SmolVLA 的 Flow Matching 在每次生成 Action Chunk 时都会采样随机
噪声，`policy_seed` 只固定该噪声的生成种子。因此单次评测结果同时受两类随机性影响：

1. **场景布局随机性**：`scene_seed` 决定积木/杯子初始位置；
2. **采样随机性**：`policy_seed` 决定 Flow Matching 噪声，进而影响动作轨迹。

本实验把 `policy_seed` 从"固定为 20260 的复现字段"升级为正式实验变量：同一场景、
任务、措辞在 N 个 seed 下重复执行，用 Bootstrap 把两类随机性一起纳入成功率区间估计。

> 说明：`policy_seed` 只固定采样噪声种子，不消除硬件 FP16 非确定性。若需评估硬件
> 层确定性，可对同一 seed 重复执行同条件推理（既有 conditioning 诊断已做过同条件
> 重复推理的确定性验证）。本实验通过统计手段量化总体波动，不单独诊断硬件层。

## 2. 实验设计

### 2.1 评测矩阵

与正式 120 条实验完全同构，仅扩展 policy seed 维度：

```text
10 个未见 scene seed（10000-10009）
× 4 类任务（red_on_blue / red_on_yellow / green_on_blue / green_on_yellow）
× 3 种措辞（canonical / synonym / unseen）
× N 个 policy seed
```

| 档位 | N | 总 rollout | 适用场景 |
| --- | ---: | ---: | --- |
| 快速档 | 3 | 360 | 先跑通全链路、预算有限（推荐先做） |
| 严谨档 | 5 | 600 | 需要更窄的区间与更强的统计结论 |

配置默认 N=3（`configs/eval_multi_seed.yaml`）；扩到严谨档只需把 `policy_seeds`
改为 `20260, 20261, 20262, 20263, 20264`，并换新输出目录。

### 2.2 Seed 选择

- `scene_seeds`：10000–10009，与正式实验一致，全部未见于训练；
- `policy_seeds`：20260 起连续 N 个。**20260 与既有正式 120 条实验完全可比**
  （`eval_standard.yaml` 固定为 20260），可作为该模型与历史结果的直接对照；
- 其余维度（fps=20、max_steps=400、成功条件）与正式实验完全一致。

### 2.3 实验键

```text
scene=<scene_seed>|task=<task_id>|prompt=<prompt_type>|policy=<policy_seed>
```

360 条全部唯一；`--resume` 按该键断点续跑。

## 3. 统计口径

### 3.1 总体成功率

```text
总体成功率 = 全部成功 rollout / 全部 rollout（360 或 600 条）
```

### 3.2 Scene 分层 cluster Bootstrap 95% CI（第一主指标）

- **聚类单元 = `scene_seed`**：同一 scene 的全部 rollout（含多个 seed、任务、措辞）
  作为一个 cluster 整体重采样，避免把同一布局的 12/N 条结果当作独立样本，否则会
  系统性低估区间宽度；
- 每次重采样按 scene 整组抽取（可重复），把选中 scene 的全部 rollout 合并求成功率；
- B = 10000 次，随机种子固定为 `20260813`（与 `evaluate/rollout.py` 的
  `bootstrap_success_ci` 完全一致，保证历史 120 条结果与本次 360 条结果可对比）；
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

对每个 `(scene, task, prompt)` 三元组统计 N 个 seed 下的成功数：

| 分类 | 条件 | 含义 |
| --- | --- | --- |
| 稳定成功 | N/N 成功 | 采样不敏感，结果可靠 |
| Seed 敏感 | 部分 seed 成功 | 采样噪声决定成败，需谨慎解读单次结果 |
| 稳定失败 | 0/N 成功 | 非采样问题，模型在该条件下系统性失败 |

> 现有 `evaluate/rollout.py` 的 `stability` 统计只支持 2-seed 三分法，N>2 时不计数；
> 本实验的 K-seed 敏感性由 `scripts/analyze_multi_seed.py` 独立计算补齐。

### 3.6 语言泛化口径（沿用既有定义）

```text
seen 成功率 = (canonical + synonym) / 2
语言泛化差距 = seen 成功率 - unseen 成功率
```

`unseen` 措辞 `Move the {cube_color} cube to the {pad_color} pad.` 未见于训练。

## 4. 台式机运行步骤

### 4.0 前置检查

```powershell
conda activate smolvla-eval
nvidia-smi                    # 确认 RTX 4060 与驱动版本
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

checkpoint 必须完整（缺一不可），从训练服务器/笔记本拷贝：

```text
pretrained_model/
├── config.json
├── model.safetensors
├── policy_preprocessor.json
└── policy_postprocessor.json
```

### 4.1 冒烟（1 条 2 步）

```powershell
.\evaluate\run.ps1 `
  --checkpoint <checkpoint目录> `
  --config configs\eval_multi_seed.yaml `
  --output-dir outputs\eval\multi_seed_smoke `
  --max-rollouts 1 --max-steps 2 --keep-all-videos
```

只验证 checkpoint 加载、CUDA 前向、随机种子、视频与 manifest，不评价效果。

### 4.2 计时外推

冒烟通过后跑 **12 条预实验**（1 个 scene × 4 任务 × 3 措辞 × 1 个 seed，对应矩阵
最前面 12 条），用实际耗时估算总时长：

```powershell
.\evaluate\run.ps1 `
  --checkpoint <checkpoint目录> `
  --config configs\eval_multi_seed.yaml `
  --output-dir outputs\eval\multi_seed_pilot `
  --max-rollouts 12 --keep-all-videos
```

```text
预计总时长 ≈ 12条实际耗时 / 12 × 360（或 600）
```

RTX 4060 上单条 rollout（400 步上限）通常为 1–3 分钟量级，360 条约数小时，建议
整批后台运行。预实验目录与正式目录分开，不并入正式结果。

### 4.3 正式运行（360 条 / N=3）

```powershell
.\evaluate\run.ps1 `
  --checkpoint <checkpoint目录> `
  --config configs\eval_multi_seed.yaml `
  --output-dir outputs\eval\formal_020000_multiseed
```

**断点续跑**：中断后使用完全相同的 checkpoint、配置、输出目录，追加 `--resume`：

```powershell
.\evaluate\run.ps1 `
  --checkpoint <checkpoint目录> `
  --config configs\eval_multi_seed.yaml `
  --output-dir outputs\eval\formal_020000_multiseed `
  --resume
```

运行开始后不得修改 checkpoint、seed、措辞、超时或成功条件；checkpoint、代码、
环境或配置变化时必须使用新输出目录。

### 4.4 跨 seed 聚合分析

```powershell
python scripts\analyze_multi_seed.py `
  --input outputs\eval\formal_020000_multiseed `
  --output outputs\eval\formal_020000_multiseed\analysis
```

产物：

```text
analysis/
├── multi_seed_report.md     # 人工可读：per-seed 表、总体 CI、敏感性、分任务/措辞
└── multi_seed_summary.json  # 机器可读：全部统计字段
```

## 5. 结果检查

```powershell
$rows = Import-Csv outputs\eval\formal_020000_multiseed\rollouts.csv
$rows.Count                                   # 应为 360
($rows.rollout_key | Sort-Object -Unique).Count   # 应为 360
$rows.policy_seed | Sort-Object -Unique       # 应为 20260,20261,20262
Get-Content outputs\eval\formal_020000_multiseed\summary.json -Encoding UTF8
Get-Content outputs\eval\formal_020000_multiseed\analysis\multi_seed_report.md -Encoding UTF8
```

正式结果必须同时满足：360 条、360 个唯一实验键、10 个 scene seed、4 类任务、
3 种措辞、3 个 policy seed、无 `control_exception`。

## 6. 结果记录模板

将 `analysis\multi_seed_report.md` 的数值填入下表（示例列，跑完替换）：

### 6.1 总体与跨 seed

| Checkpoint | Rollout | 总体成功率 | Scene Bootstrap 95% CI | 跨 seed 均值±std | min–max | 极差 |
| --- | ---: | ---: | --- | --- | --- | ---: |
| `<路径>` | 360 | `__%` | `[__%, __%]` | `__% ± __%` | `[__%, __%]` | `__%` |

### 6.2 Per-seed 成功率

| policy seed | 成功数 | 总数 | 成功率 | Scene Bootstrap 95% CI |
| ---: | ---: | ---: | ---: | ---: |
| 20260 |  | 120 | `__%` | `[__, __]` |
| 20261 |  | 120 | `__%` | `[__, __]` |
| 20262 |  | 120 | `__%` | `[__, __]` |

### 6.3 Seed 敏感性

| 稳定成功（N/N） | Seed 敏感 | 稳定失败（0/N） | 不完整分组 |
| ---: | ---: | ---: | ---: |
| `__` | `__` | `__` | 0 |

### 6.4 分措辞（跨 seed 聚合）

| 措辞 | 成功率 | 与训练关系 |
| --- | ---: | --- |
| canonical | `__%` | 已见 |
| synonym | `__%` | 已见 |
| unseen | `__%` | 未见 |
| **语言泛化差距** | `__%` | seen − unseen |

### 6.5 结论填写（示例句式）

- 模型在 10 个未见场景上的总体严格成功率为 `__%`（95% CI `[__%, __%]`，Scene 分层
  Bootstrap，B=10000）；
- 3 个采样 seed 下成功率波动 `__%`（极差），说明 Flow Matching 采样随机性
  [可忽略 / 不可忽略]；
- 稳定失败条件 `__` 个（`__%`），采样不敏感条件下模型[具备 / 不具备]可靠完成能力。

## 7. 结论边界

- 结果只代表当前 checkpoint 在本机 MuJoCo 仿真闭环中的能力，不代表真实 UR10e 成功率；
- Bootstrap 区间反映场景布局与采样随机性，**不消除硬件 FP16 非确定性**；
- 360/600 条结果不能与缺少基座或其他 checkpoint 对照时宣称"训练带来提升"；
- `unseen` 只代表措辞泛化，不代表新任务组合泛化。

## 8. 文件清单（本次新增）

| 文件 | 说明 |
| --- | --- |
| `configs/eval_multi_seed.yaml` | N=3 多 seed 评测配置 |
| `scripts/analyze_multi_seed.py` | 跨 seed 聚合 + Bootstrap 分析脚本 |
| 本文档 | 实验设计与操作手册 |
