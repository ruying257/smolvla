# SmolVLA 本机评测与诊断

本目录提供 SmolVLA 在 MuJoCo 中的标准闭环评测、失败归因和视觉鲁棒性评测。正式结论应来自固定实验矩阵上的重复闭环成功率，不能用训练 loss、单条视频或静态特征差异代替。

## 1. 工具结构

| 文件 | 作用 | 是否推进物理 |
| --- | --- | --- |
| `rollout.py` | 积木/杯子标准闭环评测、恢复、汇总和视频管理 | 是 |
| `rollout_robustness.py` | 视觉鲁棒性专用 rollout，支持图像变换和按参数整理产物 | 是 |
| `diagnose_language.py` | 检查颜色词经过真实 preprocessor 后的 Token 差异 | 否 |
| `diagnose_conditioning.py` | 检查语言差异是否传播到 VLM 特征和 Action Chunk | 否 |
| `diagnose_visual_counterfactual.py` | 用位置交换/颜色中性化检查视觉—语言空间绑定 | 否 |
| `diagnose_mug_visual_robustness.py` | 杯子外观和全图像素扰动下的闭环成功率评测 | 是 |
| `common.py` | 配置、路径、checkpoint 和 JSON 公共工具 | — |
| `run.ps1` | 检查 Conda 环境后启动 `python -m evaluate` | — |

推荐流程：

```text
标准闭环评测
  ├─ 语言异常 → Token诊断 → 条件传播诊断
  ├─ 目标选择异常 → 视觉反事实诊断
  └─ 换皮/成像异常 → Mug视觉鲁棒性评测
```

## 2. 环境与入口

已验证环境：Windows、Python 3.11、PyTorch 2.7.0+cu126、LeRobot 0.4.4、MuJoCo 3.6.0、GTX 1650。

```powershell
cd F:\桌面\smolvla
conda activate smolvla-eval
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

标准评测有两个等价入口：

```powershell
python -m evaluate --help
.\evaluate\run.ps1 --help
```

入口默认启用 Hugging Face 离线模式。checkpoint 可以传训练输出根目录，也可以传具体的 `pretrained_model`：

- 训练输出根目录：自动选择 `checkpoints\last\pretrained_model`；
- 具体 checkpoint：严格使用指定步数，不自动切换；
- 不同步数或配置必须使用不同输出目录。

checkpoint 至少包含：

```text
pretrained_model/
├── config.json
├── model.safetensors
├── policy_preprocessor.json
└── policy_postprocessor.json
```

开发或修改评测逻辑后运行：

```powershell
python -m unittest discover -s tests -v
```

## 3. 标准闭环评测

### 3.1 固定随机性与成功口径

- `scene_seed`：控制物体初始布局；
- `policy_seed`：控制 Flow Matching 的 Action Chunk 采样噪声；
- 当前正式配置统一固定 `policy_seed=20260`，主要比较 scene、task、prompt 或视觉条件；
- 评测频率为 20 Hz，积木默认最多 400 步，杯子配置最多 360 步。

严格成功要求目标物体完整进入目标区域、连续稳定 0.5 秒且夹爪已释放。杯子任务还要求杯子保持直立，并使用目标区内缩边界。

### 3.2 现有评测矩阵

| 配置 | 矩阵 | 用途 |
| --- | ---: | --- |
| `configs/eval_standard.yaml` | 10 未见 scene × 4 task × 3 prompt = 120 | 积木正式泛化评测 |
| `configs/eval_seen.yaml` | 6 已见 scene × 4 task = 24 | 积木已见布局对照 |
| `configs/eval/mug_v1.yaml` | 20 已见 scene × 2 task = 40 | 杯子 seen 评测 |
| `configs/eval/mug_v1_unseen.yaml` | 20 未见 scene × 2 task = 40 | 杯子 unseen 评测 |
| `configs/eval/mug_v1_seen_green.yaml` | 6 已见 scene × 2 task = 12 | 绿白杯探索 |
| `configs/eval/mug_v1_seen_green_2seeds.yaml` | 2 已见 scene × 2 task = 4 | 绿白杯冒烟 |
| `configs/eval/motion_limiter/*.yaml` | 20 已见 scene × 2 task = 40/组 | h25 限制器对照 |

积木三种 prompt：

| 类型 | 示例 | 训练状态 |
| --- | --- | --- |
| `canonical` | Put the red cube on the blue pad. | 已见 |
| `synonym` | Place the red cube onto the blue pad. | 已见 |
| `unseen` | Move the red cube to the blue pad. | 未见 |

杯子目前只支持 `canonical`，任务为 `mug_on_blue` 和 `mug_on_yellow`。

### 3.3 冒烟与正式运行

先用独立输出目录执行短冒烟，只验证模型加载、CUDA、场景、视频和日志链路：

```powershell
.\evaluate\run.ps1 `
  --checkpoint outputs\train\smolvla_ur10e_mug_v1_b8_s8000\checkpoints\008000\pretrained_model `
  --config configs\eval\mug_v1.yaml `
  --output-dir outputs\eval\mug_v1_smoke `
  --max-rollouts 1 `
  --max-steps 2
```

冒烟通过后运行完整矩阵：

```powershell
.\evaluate\run.ps1 `
  --checkpoint outputs\train\smolvla_ur10e_mug_v1_b8_s8000\checkpoints\008000\pretrained_model `
  --config configs\eval\mug_v1.yaml `
  --output-dir outputs\eval\mug_v1_s8000_seen_canonical
```

只复核一个 scene 时使用 `--scene-seed`；它保留该 scene 下配置中的全部任务、措辞和 policy seed，并应使用新输出目录。

### 3.4 恢复与失败重跑

标准评测首次运行不加 `--resume`。中断后使用完全相同的 checkpoint、配置、参数和输出目录：

```powershell
.\evaluate\run.ps1 `
  --checkpoint <与首次相同> `
  --config <与首次相同> `
  --output-dir <与首次相同> `
  --resume
```

`--resume` 会校验 manifest、checkpoint SHA-256、源码、配置、环境、实验键和视频策略；合法结果会跳过，`control_exception` 或带 `error` 的轨迹会重跑。

成功判据修订后，可只覆盖已有失败项：

```powershell
.\evaluate\run.ps1 `
  --checkpoint outputs\train\smolvla_ur10e_mug_v1_b8_s8000\checkpoints\008000\pretrained_model `
  --config configs\eval\mug_v1.yaml `
  --output-dir outputs\eval\mug_v1_s8000_seen_canonical `
  --rerun-failures
```

`--rerun-failures` 不能与 `--resume`、`--scene-seed` 或 `--max-rollouts` 组合。该模式混合旧成功轨迹与新失败重跑，适合修订结果，不代替全量同版本复跑。

### 3.5 动作执行选项

#### Execution horizon

`--execution-horizon K` 表示每次仍预测完整 Action Chunk，但只执行前 K 步后根据最新观测重规划。它属于 manifest 身份，改变时必须使用新输出目录；K 越小，闭环反馈更频繁，推理开销也越高。

#### Chunk blend

`--chunk-blend K` 使用旧 chunk 尾帧作为锚点，对新 chunk 前 K 帧做角度回卷后的线性插值；夹爪维度保持透传。`K=0` 为关闭，建议从 2–4 开始。

008000 杯子 checkpoint、h25、40 条/组的现有对照：

| K | 成功率 | exec jump ratio | 边界方向翻转率 |
| ---: | ---: | ---: | ---: |
| 0 | 77.5% | 1.39 | 0.111 |
| 2 | 80.0% | 0.92 | 0.000 |
| 4 | 85.0% | 0.54 | 0.000 |

K=4 显著降低边界抖动；成功率区间仍有重叠，不应解读为已证明统计显著提升。

#### Motion limiter

限制器只作用于六个关节的评测执行动作，不修改模型、训练数据或夹爪指令。现有对照配置：

```text
configs/eval/motion_limiter/mug_v1_seen_h25_baseline.yaml
configs/eval/motion_limiter/mug_v1_seen_h25.yaml
```

逐步日志会同时记录模型输出、反归一化动作、范围裁剪动作、最终执行动作、参考速度和真实关节状态。

### 3.6 视频保留策略

当前默认保留全部视频。只有显式增加 `--prune-videos`，才会保留全部失败视频及每个 `task_id × prompt_type` 的首条成功视频。视频策略属于 manifest 身份，续跑时不能改变。

## 4. 专项诊断

### 4.1 语言 Token 诊断

固定零图像和零状态，只加载真实 preprocessor，检查红绿与蓝黄文本是否被正确分词：

```powershell
python -m evaluate.diagnose_language `
  --checkpoint outputs\train\smolvla_ur10e_b8_s15000_r2\checkpoints\010000\pretrained_model `
  --output-dir outputs\eval\diagnostics\language_tokens_010000
```

通过只能排除 Tokenizer/preprocessor 合并颜色词，不能证明模型实际使用了颜色信息。

### 4.2 条件传播诊断

固定 scene 图像、状态和 `(1, 50, 32)` Flow Matching 噪声，检查语言差异是否传播到 VLM 特征、prefix KV cache 和 Action Chunk：

```powershell
python -m evaluate.diagnose_conditioning `
  --checkpoint outputs\train\smolvla_ur10e_b8_s15000_r2\checkpoints\010000\pretrained_model `
  --config configs\diagnose_conditioning.yaml `
  --output-dir outputs\eval\diagnostics\conditioning_010000
```

冒烟可增加 `--max-scenes 1 --prompt-types canonical`。该工具不推进 MuJoCo，不代表闭环成功率。

### 4.3 视觉反事实诊断

固定机器人状态和采样噪声，比较 `original`、`swap_positions`、`neutralize_red`、`neutralize_green`，定位视觉编码、跨模态绑定、动作专家或空间方向映射的最早异常环节：

```powershell
python -m evaluate.diagnose_visual_counterfactual `
  --checkpoint outputs\train\smolvla_ur10e_b8_s15000_r2\checkpoints\010000\pretrained_model `
  --config configs\diagnose_visual_counterfactual.yaml `
  --output-dir outputs\eval\diagnostics\visual_counterfactual_010000
```

正式矩阵为 `6 scene × 2 instruction × 4 variant = 48` 个条件，并包含 original 重复控制。`--max-scenes 1` 只用于链路冒烟；特征或 Action Chunk 差异不能单独证明抓取成功。

### 4.4 Mug 视觉鲁棒性评测

该工具固定 `scene_seed × task × canonical × policy_seed=20260`，只改变视觉条件：

- 环境级外观：`original`、`green_white`、`changed`；
- 全图像素扰动：亮度、对比度、Gamma、高斯噪声、高斯模糊、JPEG；
- 两路相机同时扰动，视频记录策略实际看到的扰动后图像；
- 当前不包含 Mug ROI 或 pad ROI 遮挡。

默认配置为 `5 scene × 2 task × (3 外观 + 18 像素档) = 210` 条，即每个视觉条件 10 条。扩为 20 个 scene 后是 840 条，即每条件 40 条。

当前应显式传入实际配置路径：

```powershell
python -m evaluate.diagnose_mug_visual_robustness `
  --checkpoint outputs\train\smolvla_ur10e_mug_v1_b8_s8000\checkpoints\008000\pretrained_model `
  --config configs\eval\mug_robustness\diagnose_mug_robustness.yaml `
  --output-dir outputs\eval\mug_robustness `
  --device cuda
```

亮度单 scene 冒烟：

```powershell
python -m evaluate.diagnose_mug_visual_robustness `
  --checkpoint outputs\train\smolvla_ur10e_mug_v1_b8_s8000\checkpoints\008000\pretrained_model `
  --config configs\eval\mug_robustness\diagnose_mug_robustness.yaml `
  --output-dir outputs\eval\mug_robustness_smoke `
  --max-scenes 1 `
  --perturbations brightness
```

高斯噪声使用条件派生的确定性随机种子。崩溃阈值定义为成功率首次跌破 `max(0.5, baseline - 20pp)` 的配置档位，baseline 为 `original` 无扰动。

## 5. 输出与追溯

### 5.1 标准评测产物

```text
outputs/eval/<run>/
├── run_manifest.json
├── rollouts.jsonl
├── rollouts.csv
├── summary.json
├── report.md
├── action_clipping_summary.json
├── action_clipping_by_dimension.csv
├── motion_metrics_by_rollout.csv
├── motion_metrics_summary.json
├── motion_metrics_report.md
├── stage_metrics_by_rollout.csv
├── stage_metrics_summary.json
├── video_retention.json
├── rollout_update.json        # 仅 --rerun-failures 模式生成
├── action_traces/
└── videos/
```

每条 rollout 使用稳定实验键：

```text
scene=<scene_seed>|task=<task_id>|prompt=<prompt_type>|policy=<policy_seed>
```

`rollouts.jsonl` 每条完成后立即刷新；CSV、summary、report 和各类统计文件在整批完成后生成。`control_exception` 是评测有效性异常，不能当作普通策略失败。

### 5.2 鲁棒性评测产物

```text
outputs/eval/<robustness-run>/
├── run_manifest.json
├── rollouts.jsonl
├── rollout_detail.csv
├── robustness_aggregate.csv
├── summary.json
├── report.md
├── curves/
├── action_traces/
│   ├── appearance/<variant>/
│   └── pixel/<扰动名>/<强度>/
└── videos/
    ├── appearance/<variant>/
    └── pixel/<扰动名>/<强度>/
```

目录强度使用简洁稳定格式，例如 `pixel/brightness/0.5/`、`pixel/jpeg/30/`。视频与动作日志使用相同参数层级，历史产物不会自动迁移。

## 6. 统计与验收

标准闭环评测至少报告：

- 总体严格成功率和按 scene 整组 Bootstrap 95% 区间；
- 分任务、分 scene、分 prompt 成功率；
- 失败类型、成功步数、推理延迟；
- 动作裁剪、运动平滑度和 Mug 阶段分布。

积木正式配置验收：120 条结果、120 个唯一实验键、10 个 scene、4 个任务、3 种 prompt、固定 `policy_seed=20260`、无 `control_exception`。

杯子 seen/unseen 配置分别验收：40 条结果、40 个唯一实验键、20 个 scene、2 个任务、canonical、固定 `policy_seed=20260`、无 `control_exception`。

快速检查：

```powershell
$rows = Import-Csv outputs\eval\<run>\rollouts.csv
$rows.Count
($rows.rollout_key | Sort-Object -Unique).Count
Get-Content outputs\eval\<run>\summary.json -Encoding UTF8
Get-Content outputs\eval\<run>\report.md -Encoding UTF8
```

所有结果仅代表指定 checkpoint 在当前 MuJoCo 仿真环境中的能力，不代表真实 UR10e 成功率；缺少基线或其他 checkpoint 对照时，也不能单独宣称训练带来提升。
