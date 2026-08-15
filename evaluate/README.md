# SmolVLA 本机模型效果评测

本文档定义 SmolVLA 在本笔记本 `smolvla-eval` Conda环境中的MuJoCo闭环评测框架。正式结论来自固定实验矩阵上的重复闭环成功率，而不是训练loss、单个视频或单次随机rollout。

## 1. Scene seed与固定policy seed

评测包含两个相互独立的随机来源：

- `scene_seed`：控制两个积木的初始位置；同一seed复现相同场景。
- `policy_seed`：控制SmolVLA Flow Matching生成Action Chunk时的采样噪声。

本轮不把policy seed作为模型效果的主要实验变量，所有配置统一固定为`20260`，仅用于复现Flow Matching采样。模型效果主要通过不同`scene_seed`、任务和措辞进行比较；结果中仍记录policy seed，便于追溯。

## 2. 正式实验矩阵

`configs/eval_standard.yaml`锁定：

```text
10个未见scene seed（10000-10009）
× 4类任务
× 3种措辞
× 1个固定policy seed（20260）
= 120条rollout
```

三种措辞为：

| 类型 | 示例 | 训练状态 |
| --- | --- | --- |
| `canonical` | Put the red cube on the blue pad. | 已见 |
| `synonym` | Place the red cube onto the blue pad. | 已见 |
| `unseen` | Move the red cube to the blue pad. | 未见 |

训练数据使用的scene seed为`0-9`与`100-139`，正式测试使用`10000-10009`，不存在布局seed重合。专家示范最长351步，因此正式超时锁定为400步，即20 Hz下20秒，并提供约14%余量。

严格成功必须满足：指定积木完整进入目标区域内缩5 mm后的范围、连续稳定0.5秒且夹爪已经释放。

## 3. 环境与checkpoint

已验证环境为Windows、Python 3.11、PyTorch 2.7.0+cu126、LeRobot 0.4.4、MuJoCo 3.6.0和GTX 1650。评测不使用云端`.venv-cloud`或EGL。

```powershell
cd F:\桌面\smolvla
conda activate smolvla-eval
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python view_scene.py --headless --steps 10 --scene-seed 10000
```

统一入口默认启用Hugging Face离线模式。本机缓存必须包含checkpoint引用的`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`配置、处理器与权重。

checkpoint至少包含：

```text
pretrained_model/
├── config.json
├── model.safetensors
├── policy_preprocessor.json
└── policy_postprocessor.json
```

## 3.1 Seen场景对照实验

`configs/eval_seen.yaml`用于验证新训练模型在训练已见布局中的效果：

```text
6个训练已见scene seed（0-5）
× 4类任务
× 1种训练已见canonical措辞
× 1个policy seed（20260）
= 24条rollout
```

这6个scene seed在四类任务的训练示范中都出现过。除场景集合、措辞范围和只采用`policy_seed=20260`外，评测仍使用与正式实验相同的checkpoint、20 Hz、400步超时和成功条件。执行命令：

```powershell
.\evaluate\run.ps1 `
  --checkpoint outputs\train\smolvla_ur10e_b8_s15000_r2 `
  --config configs\eval_seen.yaml `
  --output-dir outputs\eval\seen_canonical
```

中断后在同一命令末尾增加`--resume`。分析时只能将seen实验与正式120条结果中的`prompt_type=canonical`子集比较，不能把24条seen结果与包含三种措辞的总体成功率直接相减。该对照用于描述已见布局与未见布局的性能差距，不等同于独立测试集上的泛化结论。

### 3.2 Execution horizon=10诊断实验

SmolVLA checkpoint保持`chunk_size=50`。增加`--execution-horizon 10`后，每次仍生成50步动作，但只执行前10步，随后使用最新图像和状态重新生成chunk。该模式属于Receding Horizon，不对重叠chunk求平均，也不是RTC。

不能向旧的50步run续跑，因为execution horizon属于manifest身份。请使用新输出目录：

```powershell
conda activate smolvla-eval
.\evaluate\run.ps1 `
  --checkpoint outputs\train\smolvla_ur10e_b8_s15000_r2 `
  --config configs\eval_seen.yaml `
  --output-dir outputs\eval\seen_canonical_h10 `
  --execution-horizon 10
```

建议与原`seen_canonical`结果成对比较成功率、抓取阶段轨迹、推理耗时和动作裁剪率。horizon=10会把模型推理频率提高约5倍，因此不能只比较成功率而忽略计算开销。

### 3.3 红绿积木语言Token诊断

当红绿任务产生近似轨迹时，先用checkpoint真实preprocessor排除任务文本、换行、Tokenizer、padding和attention mask问题。该诊断固定相同零图像与七维零状态，覆盖`2种积木颜色 × 2种底板颜色 × 3种措辞 = 12条`，只在CPU加载预处理器，不加载策略权重、不运行MuJoCo控制：

```powershell
conda activate smolvla-eval
python -m evaluate.diagnose_language `
  --checkpoint outputs\train\smolvla_ur10e_b8_s15000_r2\checkpoints\010000\pretrained_model `
  --output-dir outputs\eval\diagnostics\language_tokens_010000
```

输出包含：

```text
language_tokens_010000/
├── run_manifest.json          # checkpoint、preprocessor哈希、Tokenizer和依赖版本
├── token_records.jsonl        # 12条文本的原始、换行、完整及有效Token记录
├── pairwise_comparison.csv    # 6组红绿和6组蓝黄控制配对
├── summary.json               # 机器可读判定及后续分流
└── report.md                  # 人工可读Token与差异位置表
```

通过条件为：所有红绿与蓝黄配对的attention mask相同、均无截断、完整preprocessor与直接Tokenizer结果一致，并且每对只在一个有效颜色Token位置发生预期差异。通过只能排除语言输入处理阶段把颜色合并，不能证明冻结VLM或动作专家实际使用颜色信息；下一步应固定图像、状态和Flow Matching噪声，比较VLM语言特征与action chunk。

### 3.4 固定条件的VLM特征与Action Chunk诊断

Token诊断通过后，使用`configs/diagnose_conditioning.yaml`检查红绿语言差异是否继续传到冻结VLM和动作专家。实验覆盖6个训练已见场景、4类任务和3种措辞，共72个条件。每个条件使用完全相同的输入重复推理一次，因此共产生144个action chunk。

每个场景只执行一次`reset`，不推进MuJoCo物理。场景内所有任务严格复用两路图像和七维状态；全部条件严格复用由唯一`policy_seed=20260`生成的`(1, 50, 32)` Flow Matching噪声。该实验不是第二组policy seed对照，重复推理只用于确认数值确定性。

```powershell
conda activate smolvla-eval
python -m evaluate.diagnose_conditioning `
  --checkpoint outputs\train\smolvla_ur10e_b8_s15000_r2\checkpoints\010000\pretrained_model `
  --config configs\diagnose_conditioning.yaml `
  --output-dir outputs\eval\diagnostics\conditioning_010000
```

先做单场景、canonical真实模型冒烟时使用新输出目录：

```powershell
conda activate smolvla-eval
python -m evaluate.diagnose_conditioning `
  --checkpoint outputs\train\smolvla_ur10e_b8_s15000_r2\checkpoints\010000\pretrained_model `
  --config configs\diagnose_conditioning.yaml `
  --output-dir outputs\eval\diagnostics\conditioning_010000_smoke `
  --max-scenes 1 `
  --prompt-types canonical
```

诊断链路为：

```text
颜色Token embedding
→ VLM最终上下文化语言特征
→ 动作专家实际使用的16层prefix KV cache
→ 归一化50×7 action chunk
→ 反归一化物理动作
→ 限位裁剪动作
```

主要产物包括`run_manifest.json`、`fixed_noise.npy`、`fixed_inputs/`、`condition_records.jsonl`、`language_features.npz`、`action_chunks.npz`、两类比较CSV、`summary.json`、`report.md`和两张差异曲线。`action_comparison.csv`同时报告前10步和完整50步，并用蓝黄目标颜色差异作为红绿源颜色差异的控制组。

结果按传播阶段定位：VLM特征不同而动作接近底噪，说明动作专家可能没有有效使用语言条件；物理动作不同但裁剪后趋同，说明执行限位抹除了部分条件差异；红绿差异明显弱于蓝黄差异，说明模型可能更依赖目标底板。上述结果都只证明模型对文本变化的敏感程度，不能证明它正确定位或抓取了相应颜色积木；空间正确性仍需结合物体位置和闭环轨迹验证。

### 3.5 视觉反事实因果诊断

当固定图像诊断表明红绿语言特征能够传播，但闭环仍无法选择正确积木时，使用`configs/diagnose_visual_counterfactual.yaml`直接改变积木视觉条件。实验固定机器人状态、Flow Matching噪声、blue目标、canonical措辞和`policy_seed=20260`，不推进MuJoCo动力学，也不修改正式rollout。

四种视觉版本为：

- `original`：原始红绿积木位置与颜色；
- `swap_positions`：交换两个free joint的完整七维qpos，颜色随积木移动；
- `neutralize_red`：红块保持位置和几何，只把RGBA改成中性灰；
- `neutralize_green`：绿块保持位置和几何，只把RGBA改成中性灰。

正式实验包含`6 scenes × 2 instructions × 4 variants = 48`个条件，并对每个scene的两条original条件各做一次完全相同的重复推理，共60个Action Chunk：

```powershell
conda activate smolvla-eval
python -m evaluate.diagnose_visual_counterfactual `
  --checkpoint outputs\train\smolvla_ur10e_b8_s15000_r2\checkpoints\010000\pretrained_model `
  --config configs\diagnose_visual_counterfactual.yaml `
  --output-dir outputs\eval\diagnostics\visual_counterfactual_010000
```

GTX 1650真实冒烟只运行第一个scene，即8个条件和2个重复控制：

```powershell
conda activate smolvla-eval
python -m evaluate.diagnose_visual_counterfactual `
  --checkpoint outputs\train\smolvla_ur10e_b8_s15000_r2\checkpoints\010000\pretrained_model `
  --config configs\diagnose_visual_counterfactual.yaml `
  --output-dir outputs\eval\diagnostics\visual_counterfactual_010000_smoke `
  --max-scenes 1
```

单场景冒烟的统计标签为`insufficient_scenes`，只验收CUDA、特征截取、确定性重复和产物写出；因果故障定位必须使用完整6个scene。

诊断首先保存两路RGB和基于MuJoCo geom ID生成的红绿精确mask，再将256×256 mask按真实512输入与connector结构聚合为8×8、共64个视觉Token的ROI权重。某块积木因视角遮挡而在单路相机中不可见时，工具保留真实的全零mask和零ROI权重；只有两路真实相机都看不到该颜色时才判为无效，不会伪造ROI。启动阶段还会拒绝状态变化、错误位置交换、非目标像素异常漂移和非有限特征。

跨模态和动作阶段使用同一因果选择指数：

```text
CSI = (D_target - D_distractor) / (D_target + D_distractor + epsilon)
```

例如red指令中，`D_target`是original与neutralize_red的距离，`D_distractor`是original与neutralize_green的距离。CSI为正表示目标积木的视觉干预影响更大；单个scene的正值不能当作稳定grounding证据。报告以scene为独立单位，给出中位数、正向scene数、按scene Bootstrap 95%区间以及`consistent`、`mixed`或`opposite_or_insensitive`标签。

工具按最早异常环节给出定位：

1. RGB变化但视觉Token不响应：视觉编码器或connector不敏感；
2. 视觉Token响应但VLM-CSI不为正：颜色词—图像区域grounding不足；
3. VLM-CSI成立但Action-CSI不成立：动作专家没有有效使用视觉绑定；
4. Action-CSI成立但位置跟随余弦不稳定：动作到空间方向的映射不足。

产物包括`run_manifest.json`、`fixed_noise.npy`、`counterfactual_inputs/`、`condition_records.jsonl`、`visual_features.npz`、`conditioning_features.npz`、`action_chunks.npz`、三个比较CSV、`summary.json`、`report.md`和两张诊断曲线。`counterfactual_inputs/`中的干预审计JSON用于核验两路相机像素变化没有越出允许ROI。

FK只把50步绝对关节命令投影为attachment-site的运动学轨迹，用于判断交换目标位置后动作方向是否跟随目标；它不推进动力学、不模拟接触，也不等同于闭环抓取成功。视觉Token、VLM特征或Action Chunk存在差异，也都不能单独证明模型正确识别并抓取了目标积木。

## 4. 分阶段执行

### 4.1 自动测试

```powershell
conda activate smolvla-eval
python -m unittest discover -s tests -v
```

### 4.2 两步链路冒烟

只验证checkpoint加载、CUDA前向、随机种子、视频、manifest和JSONL，不评价模型效果：

```powershell
.\evaluate\run.ps1 `
  --checkpoint outputs\train\smolvla_ur10e `
  --config configs\eval_standard.yaml `
  --output-dir outputs\eval\smoke_reproducible `
  --max-rollouts 4 `
  --max-steps 2 `
  --keep-all-videos
```

### 4.3 12条预实验

使用正式配置的第一个scene seed。由于矩阵顺序为`scene → task → prompt → policy`，前12条刚好覆盖：

```text
1 scene × 4 tasks × 3 prompts × 1 fixed policy seed
```

```powershell
.\evaluate\run.ps1 `
  --checkpoint outputs\train\smolvla_ur10e `
  --config configs\eval_standard.yaml `
  --output-dir outputs\eval\pilot_020000 `
  --max-rollouts 12 `
  --keep-all-videos
```

预实验用于检查显存、耗时、失败分类和恢复机制，不并入正式120条结果。

### 4.4 正式120条实验

正式运行开始后不得根据中间结果修改checkpoint、seed、措辞、超时或成功条件。

```powershell
.\evaluate\run.ps1 `
  --checkpoint outputs\train\smolvla_ur10e `
  --config configs\eval_standard.yaml `
  --output-dir outputs\eval\formal_020000
```

运行中断后使用完全相同的命令并增加`--resume`：

```powershell
.\evaluate\run.ps1 `
  --checkpoint outputs\train\smolvla_ur10e `
  --config configs\eval_standard.yaml `
  --output-dir outputs\eval\formal_020000 `
  --resume
```

默认在整批完成后保留全部失败视频，并为每个`task_id × prompt_type`保留按seed排序的第一条成功视频。若需要保留全部视频，首次启动和每次续跑都必须增加`--keep-all-videos`；视频策略属于manifest身份，不能在同一run中途改变。

## 5. 可恢复性与追溯

每条轨迹以以下字符串作为唯一实验键：

```text
scene=<scene_seed>|task=<task_id>|prompt=<prompt_type>|policy=<policy_seed>
```

每条完成后立即落盘到`rollouts.jsonl`并刷新磁盘。`--resume`会：

1. 校验checkpoint、权重SHA-256、评测源码SHA-256、环境、配置、步数、实验键和视频策略；
2. 拒绝损坏JSONL、重复实验键和缺失的保留视频；
3. 跳过合法完成项；
4. 移除并重新执行`control_exception`或带`error`的轨迹。

checkpoint、代码、环境或配置变化时必须创建新输出目录，不能向旧run混写。

## 6. 输出结构

```text
outputs/eval/formal_020000/
├── run_manifest.json       # checkpoint、代码、配置、环境和120个实验键
├── rollouts.jsonl          # 每条完成后即时追加的恢复日志
├── rollouts.csv            # 最终逐条结果
├── action_clipping_summary.json       # 总体及七个维度的裁剪统计
├── action_clipping_by_dimension.csv   # 便于排序检查的逐维裁剪表
├── summary.json            # 机器可读统计
├── report.md               # 人工可读报告
├── video_retention.json    # 成功视频清理与保留清单
├── action_traces/          # 每个rollout独立的逐步动作JSONL
└── videos/
```

每条结果包含`scene_seed`、`policy_seed`、`rollout_key`、任务与措辞、成功和失败类型、步数、推理延迟、动作裁剪率、动作日志路径、checkpoint SHA-256、视频状态、错误和完成时间。

每条动作日志按控制步记录：

```text
model_output      模型归一化空间的原始七维输出
physical_action   policy postprocessor反归一化后的物理动作
executed_action   经过UR10e关节范围和夹爪[0, 1]限位后的实际执行动作
clipped_mask      七个维度分别是否发生裁剪
clip_amount       physical_action - executed_action
chunk_start       当前步是否为新动作chunk的第一步
```

`action_clipping_by_dimension.csv`分别报告`shoulder_pan`、`shoulder_lift`、`elbow`、三个腕关节和`gripper`的裁剪步数、裁剪率及越界量。`clipped_trace_step_rate`表示“至少一维被裁剪的控制步比例”，`clipped_action_element_rate`表示全部`控制步×7维`元素中的裁剪比例，两者不能混用。

## 7. 统计口径

第一主指标为总体严格成功率，同时报告：

- 按`scene_seed`整组重采样10000次得到的Bootstrap 95%置信区间；
- 四任务等权宏平均成功率；
- canonical、synonym、unseen成功率；
- `seen=(canonical+synonym)/2`与unseen的语言泛化差距；
- 分任务、分场景结果；policy seed固定为20260，仅作为复现字段；
- `wrong_cube`、`wrong_pad`、`dropped_or_out_of_bounds`、`timeout`和`control_exception`分布；
- 成功轨迹步数中位数和P90；
- 发生裁剪的轨迹比例、总裁剪步数占比、逐维裁剪率和归一化越界量；
- 推理延迟中位数和P95。

`control_exception`属于评测有效性异常，入口返回非零状态；修复原因后通过`--resume`重跑，不能把它当作普通模型失败解释。

## 8. 结果检查

```powershell
$rows = Import-Csv outputs\eval\formal_020000\rollouts.csv
$rows.Count
($rows.rollout_key | Sort-Object -Unique).Count
Get-Content outputs\eval\formal_020000\summary.json -Encoding UTF8
Get-Content outputs\eval\formal_020000\report.md -Encoding UTF8
```

正式结果必须同时满足：120条结果、120个唯一实验键、10个scene seed、4类任务、3种措辞、固定`policy_seed=20260`、无`control_exception`。

本结果只代表当前checkpoint在本机MuJoCo仿真闭环中的能力，不代表真实UR10e成功率，也不能在缺少基座或其他checkpoint对照时宣称“训练带来提升”。
