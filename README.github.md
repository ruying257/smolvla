# SmolVLA：UR10e 语言条件机器人操作的 MuJoCo 仿真闭环

> 一个完整的 **VLA（Vision-Language-Action）模型工程化闭环** 实践：从仿真环境与数据采集，到云端微调 SmolVLA，再到部署侧轨迹优化与统计严谨的闭环评测。所有结论均来自固定实验矩阵上的定量指标。

## 核心指标速览

| 指标 | 数值 |
| --- | --- |
| 未见场景严格成功率 | **85.8%**（20 未见场景 × 2 任务 × 3 policy seed = 120 条 rollout） |
| 成功率 95% 置信区间（Scene 分层 Bootstrap，B=10000） | **[76.7%, 93.3%]** |
| 部署优化：边界跳跃比 | **1.084 → 0.896**（速度/加速度限制器） |
| 部署优化：末端加加速度 Jerk P95 | **43.93 → 34.64 m/s³（-21.1%）** |
| 部署优化：闭环成功率 | **77.5% → 85.0%**（ChunkBlend 边界平滑） |
| 推理延迟 | ≈ 11 ms/步（GTX 1650 / FP16） |

**技术栈**：Python · PyTorch 2.7 · HuggingFace LeRobot 0.4.4 · SmolVLA（Flow Matching 动作生成）· MuJoCo 3.6 · PyAV / FFmpeg

---

## 流水线总览

```text
① 数据采集        ② 模型训练          ③ 部署推理优化         ④ 评测与诊断
┌──────────┐   ┌─────────────┐   ┌──────────────────┐   ┌──────────────────────┐
│MuJoCo 仿真│ → │ SmolVLA 微调 │ → │ 限制器 + ChunkBlend│ → │ 闭环 rollout + Bootstrap│
│键盘遥操作  │   │RTX 4090     │   │轨迹平滑            │   │统计 + 失败归因         │
└──────────┘   └─────────────┘   └──────────────────┘   └──────────────────────┘
```

本文按这 4 个阶段展开，其中 **部署推理优化** 与 **评测与诊断** 是本项目的核心亮点。

---

## 阶段一：数据采集（Data Collection）

**解决的问题**：VLA 策略的质量上限由数据决定，而「评测能不能信」由数据契约与成功口径决定。

- 在 MuJoCo 中构建 UR10e + Robotiq 2F85 仿真环境：第三视角 + 腕部两路 RGB 相机（256×256），物体布局由 `scene_seed` 完全可复现；
- 20 Hz 键盘遥操作采集（IK 末端增量控制 → 7 维绝对关节目标动作），动作/观测契约固定：6 关节角 + 1 夹爪指令；
- **严格成功判定**：目标物中心进入目标区内缩边界、保持直立稳定 0.5 秒、夹爪已释放——口径先于评测定义，杜绝「宽松判定刷成功率」；
- 产出 LeRobot v3 格式数据集（40 条 mug 专家数据：20 共享场景 × 2 任务 `mug_on_blue` / `mug_on_yellow`），元数据完整记录 scene seed 与初始位姿。

---

## 阶段二：模型训练（Training）

**解决的问题**：让预训练 VLA 适配 UR10e 的具体操作技能，而不是从零训练大模型。

- 从 HuggingFace `lerobot/smolvla_base` 初始化微调（公开预训练权重，Flow Matching 生成动作）；
- 输入：两路 256×256 图像 + 7 维当前状态 + 英文指令；输出：7 维绝对关节目标动作（50 步 Action Chunk）；
- RTX 4090 上训练 8000 步（batch 8，FP16 AMP），checkpoint 完整保存模型配置、权重与策略前后处理器（归一化统计），保证下载后可在本机无头闭环加载。

```bash
bash scripts/train.sh \
  --dataset-root smolvla-data/smolvla_ur10e_mug_v1 \
  --config configs/train/mug_b8_s8000.yaml \
  --output-dir outputs/train/smolvla_ur10e_mug_v1_b8_s8000
```

---

## 阶段三：部署推理优化（Deployment Optimization）⭐

**核心问题**：SmolVLA 按 Action Chunk 逐帧执行、队列耗尽时重新预测。重预测边界处新旧 chunk 在输出空间不连续，产生**周期性方向突变**——用户能直接看到的「每 25 步一次点头」。分析确认：新 chunk 首帧的模型输出跳变是 chunk 内部的 **1.3–2.1 倍**（Wilcoxon 检验 p<0.001）。本节两项优化分别从**执行层**与**输出层**压平这一不连续。

### 3.1 优化前后轨迹对比

![优化前对比](assets/demo/before.gif)

> [待补充：优化前对比视频] —— 6 个不同未见场景的 agent 视角成功轨迹拼接。**红框处可见明显的重预测边界抖动（每 25 步一次「点头」）**。

![优化后对比](assets/demo/after.gif)

> [待补充：优化后对比视频] —— 完全相同 6 个场景，部署优化后 agent 视角。边界衔接连续，轨迹丝滑。

### 3.2 速度/加速度限制器（执行层渐进轨迹）

**原理**：策略输出的绝对关节目标常「一步到位」，其瞬时速度/加速度超过机械臂物理极限。限制器将单步目标拆解为满足 `velocity_limits` / `acceleration_limits` 的渐进参考轨迹：先计算期望速度与期望加速度，分别做二阶 `clip`，再积分出本步参考位置；并做**跨目标检测**——积分参考点越过模型目标时精确停靠，避免振荡。第 7 维夹爪指令原样透传。

```python
# evaluate/common.py — JointMotionLimiter.limit（核心节选）
target = np.clip(action[:6], self._ranges[:, 0], self._ranges[:, 1])
desired_velocity = (target - self._reference_position) / self._dt
desired_acceleration = (desired_velocity - self._reference_velocity) / self._dt
acceleration = np.clip(desired_acceleration,
                       -self._acceleration_limits, self._acceleration_limits)
velocity = np.clip(self._reference_velocity + acceleration * self._dt,
                   -self._velocity_limits, self._velocity_limits)
next_position = self._reference_position + velocity * self._dt
# 防止积分后的参考点跨越目标，跨越后精确停在当前模型目标
crossed = (target - self._reference_position) * (target - next_position) <= 0.0
next_position = np.where(crossed, target, next_position)
velocity = np.where(crossed, 0.0, velocity)
```

**效果指标**（同 checkpoint、同 40 条矩阵，仅限制器开关对照）：

| 指标 | 优化前 | 优化后 |
| --- | ---: | ---: |
| 末端加加速度 Jerk P95（m/s³） | 43.93 | **34.64（↓ 21.1%）** |
| 边界跳跃比（Boundary Jump Ratio） | 1.084 | **0.896** |

边界跳跃比 < 1 意味着：重预测边界处的步长跳动**已小于 chunk 内部普通步长**，边界不再被「放大」。

### 3.3 ChunkBlend 边界平滑（输出层带回卷插值）

**原理**：重预测时以**旧 chunk 尾帧为锚点**（即边界时刻实际到达的目标），对新 chunk 前 K 帧做线性插值，让新 chunk 从旧 chunk 的终点出发。两个关键保护：

- **角度回卷**：关节角是循环量，插值前把角度差回卷到 `[-π, π)`，避免跨 π 时多转一整圈（如 +3.0 与 -3.0 直插绕远路）；
- **夹爪透传**：第 7 维是离散开/合语义，不参与插值，避免产生「半开半合」的无效中间夹持力。

```python
# evaluate/rollout.py — ChunkBlendPolicy._blend_chunk（核心节选）
def _blend_chunk(self, chunk):
    if self._k <= 0 or self._prev_chunk is None:
        return chunk
    anchor = self._prev_chunk[-1]      # 旧 chunk 尾帧 = 边界时实际到达的目标
    k = min(self._k, chunk.shape[0])
    blended = chunk.copy()
    for t in range(k):
        weight = 0.5 if k == 1 else (t + 1) / k   # t=0 小幅前进，t=k-1 贴合新 chunk
        for i in range(6):             # 关节角：角度回卷后向锚点靠拢
            delta = _wrap_angle(chunk[t, i] - anchor[i])
            blended[t, i] = anchor[i] + weight * delta
        blended[t, 6] = chunk[t, 6]    # 夹爪：透传新 chunk 值，不插值
    return blended
```

**效果指标**（同 checkpoint、同 40 条矩阵，K=0 关闭 vs K=4）：

| 指标 | K=0（关闭） | K=4 |
| --- | ---: | ---: |
| 闭环成功率 | 77.5% | **85.0%** |
| 边界跳跃比（中位数） | 1.057 | **0.651** |
| 边界方向翻转率 | 0.111 | **0.000** |

> 说明：本实验在已见场景矩阵上做严格开关对照；成功率区间存在重叠，不宣称统计显著提升，但边界方向翻转率从 11.1% 降至 0、跳跃比降至 0.65，抖动形态的改善是直接可测的。

---

## 阶段四：评测与诊断（Evaluation & Diagnosis）⭐

### 4.1 闭环评测协议

- **双随机性显式建模**：`scene_seed` 控制物体布局随机性，`policy_seed` 控制 Flow Matching 的 Action Chunk 采样噪声；
- 20 Hz 闭环执行：每次预测 50 步、只执行前 25 步后重规划（execution horizon），每条最多 360 步；
- **严格成功口径**：目标物进入内缩边界 + 直立稳定 0.5 秒 + 夹爪已释放；
- 断点续跑：结果按稳定实验键（`scene|task|prompt|policy`）即时落盘，支持 SHA-256 校验的 `--resume`。

### 4.2 Scene-level Bootstrap：给成功率一个诚实的置信区间

**问题陈述**：单点成功率 85.8% 不可靠。评测矩阵中同一场景下有多条轨迹（多个 `policy_seed`），它们**共享同一布局随机性**——若把同场景下的轨迹当作独立样本直接算区间，会犯 **Pseudo-replication（假重复）** 统计错误，区间会被显著低估、高估显著性。

**解决方案**：以 `scene_seed` 为**聚类单元**做有放回重采样（B=10000）：每次从 20 个场景中整组抽取场景，**同一场景内的全部轨迹（含全部 policy_seed）整体进出**，重算成功率；最终置信区间取重采样分布的分位数——即 **Percentile Method（分位数法）**。

| 维度 | 配置 |
| --- | --- |
| 评测规模 | 20 个未见场景 × 2 个任务 × 3 个 policy_seed = **120 条 rollout** |
| 总体严格成功率 | **85.8%** |
| 95% 置信区间（Scene 分层 cluster Bootstrap） | **[76.7%, 93.3%]** |
| 重采样 | B=10000，聚类单元 = scene_seed |

```python
# evaluate/rollout.py — bootstrap_success_ci（口径核心）
for _ in range(repeats):                        # repeats = 10_000
    chosen = rng.choice(scenes, size=len(scenes), replace=True)  # 场景整组有放回抽样
    samples[i] = np.concatenate([grouped[s] for s in chosen]).mean()
return [np.percentile(samples, 2.5), np.percentile(samples, 97.5)]  # 分位数法
```

**附加说明**：该区间同时纳入「场景布局随机性」与「Flow Matching 采样随机性」，但不消除硬件 FP16 非确定性——本项目已在 RTX 4060 上稳定运行，作为结果边界如实披露。

### 4.3 自动化阶段失败分类器

**解决的问题**：评测 120 条 rollout 后，人工逐条回看视频归因不可扩展。评测器内置**自动化失败分类器**，结束后直接产出失败归因分布，无需人工回看视频：

| 分类维度 | 含义 |
| --- | --- |
| `grasp_failure` | 抓取失败（未抓住 / 抓取后脱落） |
| `transport_failure` | 搬运掉落（运输途中目标脱离夹爪） |
| `place_failure` | 放置偏移 / 未稳定（未达内缩边界或未满足稳定条件） |
| `timeout` | 超时（达到步数上限仍未完成） |
| `control_exception` | 控制异常（仿真/控制有效性异常，不计为策略失败） |

该功能已落地：评测结束即输出逐条归因、按任务/场景交叉分布与阶段进展统计（S1–S5 各阶段到达率与失败阶段），支撑「先归因、后补采/调参」的迭代闭环。

---

## 快速复现

```powershell
# 1. 进入评测环境（本机 Windows + smolvla-eval Conda 环境）
conda activate smolvla-eval

# 2. 120 条多 seed 闭环评测（20 未见场景 × 2 任务 × 3 policy_seed）
.\evaluate\run.ps1 `
  --checkpoint outputs\train\smolvla_ur10e_mug_v1_b8_s8000\checkpoints\008000\pretrained_model `
  --config configs\0-legacy\eval_multi_seed.yaml `
  --output-dir outputs\eval\multiseed_repro

# 3. Scene 分层 Bootstrap 置信区间与跨 seed 敏感性分析
python scripts\analyze_multi_seed.py --input outputs\eval\multiseed_repro
```

已见场景 40 条对照入口：`--config configs\eval\mug_v1.yaml`。完整实验设计见 `MultiSeed_Bootstrap_实验设计.md`。

---

## 项目结构

| 目录 | 职责 |
| --- | --- |
| `sim/` | UR10e MuJoCo 仿真环境、严格成功判定与失败分类 |
| `collector/` | 键盘遥操作采集、LeRobot v3 数据集写入与质检 |
| `cloud/` | 云端训练入口、环境预检与 smoke test |
| `evaluate/` | 闭环 rollout、Bootstrap 统计、阶段检测与诊断工具 |
| `configs/` | 评测 / 训练 / 运动限制 / 鲁棒性配置 |
| `scripts/` | 跨 seed 聚合分析、云端入口脚本 |
| `assets/` | MuJoCo 模型资源、许可证与演示视频 |
| `tests/` | 场景、采集、评测与诊断的单元测试 |

---

## 结果边界

- 所有成功率仅代表指定 checkpoint 在本机 **MuJoCo 仿真**中的闭环能力，**不代表真实 UR10e 成功率**（Sim2Real 仍需真机迁移验证）；
- 置信区间为 Scene 分层 Bootstrap 的**分位数法**结果，反映场景布局与采样随机性，不消除硬件 FP16 非确定性；
- 不预设「成功率达标」为项目成功的唯一标准：工程链路完整可复现、统计口径可信、失败可归因，是本项目工程化闭环的验收口径。
