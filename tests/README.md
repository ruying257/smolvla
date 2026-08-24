# tests

SmolVLA 自动测试包。以 pytest 组织，覆盖四类回归：MuJoCo 仿真场景结构（积木与杯子环境）、专家数据采集流水线（V1 采集状态机、Grounding v2 矩阵与恢复、杯子 V3 计划与原子写入）、闭环评测与策略（评测契约与 rollout 产物、chunk 衔接平滑、杯子视觉鲁棒性扰动）、云端训练命令构建。项目根目录无 `conftest.py` / `pytest.ini`，测试 fixture 与共享环境在各测试模块内定义并通过导入复用。

## 目录结构

| 路径 | 被测域 | 功能 |
| --- | --- | --- |
| `__init__.py` | — | 测试包初始化文件，声明本模块为「SmolVLA 自动测试包」。 |
| `test_scene.py` | sim | ACT 双积木桌面场景回归（模型维度、seed 复现、headless 渲染）。 |
| `test_mug_scene.py` | sim | 杯子双放置区场景回归（模型维度、稳定落桌、headless 渲染）。 |
| `test_collector.py` | collector | P2 采集控制、状态机、数据契约与 IK 回归。 |
| `test_grounding_collection.py` | collector (v2) | Grounding v2 采集计划、恢复、配对与蒙太奇。 |
| `test_mug_collector_v3.py` | collector (v3) | 杯子 V3 计划、schema、resume 与原子 episode 写入。 |
| `test_evaluate.py` | evaluate | 评测配置契约、CLI 参数、本地 rollout 产物与动作标定。 |
| `test_chunk_blend.py` | evaluate | `ChunkBlendPolicy` chunk 衔接平滑与夹爪保护。 |
| `test_mug_visual_robustness.py` | evaluate | 杯子视觉鲁棒性像素扰动工具与注入式 rollout。 |
| `test_cloud.py` | cloud | 云端训练命令构建（smoke / resume / 互斥 / 默认数据集）。 |

## 仿真场景回归测试

依赖 MuJoCo 渲染与物理仿真，通过共享环境实例降低重复加载成本。

### `test_scene.py` — 双积木桌面场景

`CleanTabletopEnvTest` 使用共享 `CleanTabletopEnv`，`tearDownClass` 释放渲染资源；导入 `sim.environment` 的常量与 `CleanTabletopEnv`。

- `test_model_dimensions_objects_and_cameras`：验证模型维度、积木/区域 body 与相机集合。
- seeded reset：验证同 seed 可复现、不同 seed 产生变化、积木初始不覆盖区域。
- `test_headless_step_and_camera_render`：推进环境并渲染三路非空 RGB 图像（headless 渲染）。

### `test_mug_scene.py` — 杯子双放置区场景

`MugTabletopEnvTest` 使用共享 `MugTabletopEnv`，`tearDownClass` 释放渲染资源；导入 `sim.mug_environment` 的常量与 `MugTabletopEnv`。

- `test_model_dimensions_objects_and_cameras`：验证杯子场景模型维度、杯子/区域 body 与相机集合。
- seeded reset：验证可复现、不同 seed 变化、杯子初始不覆盖 pad，以及稳定后直立落桌（依赖物理仿真）。
- `test_headless_cameras_are_non_empty_rgb`：推进环境并渲染三路非空 RGB 图像。

## 采集流水线测试

### `test_collector.py` — P2 采集控制、状态机与数据契约

导入 `collector.collect`、`collector.control`、`collector.dataset_io`、`collector.state_machine`、`collector.task_spec` 与 `sim.environment`。

- `CollectionStateMachineTest`：首次有效动作启动录制、成功确认、discard 清帧、ASCII 状态提示。
- `TaskTemplateTest`：canonical/synonym 任务模板与平衡选择。
- `DatasetContractTest`：LeRobot feature schema、视频合并、中文路径、丢弃清理、resume 契约拒绝。
- `DifferentialIKControllerTest`：IK 控制动作合法性与不收敛保持上一帧。

### `test_grounding_collection.py` — Grounding v2 矩阵采集

导入 `collector` 采集矩阵、计划、LeRobot 数据集与场景快照模块。

- `GroundingPlanTest`：80 个唯一 queue key、canonical prompt、Latin square 均衡、pilot 场景与帧数限制。
- `GroundingResumeTest`：非空工作区拒绝、损坏 progress、配置 prompt 漂移、缺失 shard 文件、初始 reference 往返。
- `GroundingPairingAndMontageTest`：同 scene 初始条件 hash 约束、蒙太奇布局与非空 H.264 视频产物。

### `test_mug_collector_v3.py` — 杯子 V3 采集

导入 `collector.v3` 计划、数据集、seed 选择与验证模块，以及 `sim.mug_environment`。

- `MugV3PlanAndSchemaTest`：40 个 paired keys、任务顺序、pilot、V3 schema、camera sources 与 seed 报告距离约束。
- `MugV3ResumeAndAtomicityTest`：resume 拒绝覆盖、配置漂移、完成键唯一、redo 归档、内存写入器 schema/lifecycle、真实 LeRobot shard round trip。

## 评测与策略测试

### `test_evaluate.py` — 评测契约与本地 rollout

导入 `evaluate.common`、`evaluate.rollout`、`scripts.calibrate_motion_limits`；定义 `workspace_temp_dir` fixture 供本模块与 `test_mug_visual_robustness` 复用。

- `EvaluationContractTests`：默认评测配置、场景 seed 过滤、源码哈希、执行 horizon 校验等 CLI 参数与契约。
- `LocalRolloutTests.test_fake_policy_writes_extended_result_and_video`：本机 CPU 短 rollout 验证结果、视频、action trace、gripper filter、chunk start、stage detection 等输出产物。

### `test_chunk_blend.py` — chunk 衔接平滑

导入 `evaluate.rollout` 中的 `ChunkBlendPolicy` 相关实现。`ChunkBlendPolicyTest` 覆盖：

- `test_wrap_angle` / `test_angle_wrap_prevents_long_way`：角度回卷走最短路径。
- `test_blend_reduces_first_frame_jump`：blend 削减首帧跳变。
- `test_gripper_dimension_not_blended`：夹爪维度透传不参与 blend。
- `test_blend_only_affects_leading_frames` / `test_k_zero_is_pass_through` / `test_horizon_truncation`：前 K 帧 blend、k=0 透传、chunk 截断。
- `test_reset_clears_state` / `test_tensor_to_numpy_action_chunk`：reset 状态清理与张量转 NumPy。

### `test_mug_visual_robustness.py` — 杯子视觉鲁棒性

导入 `evaluate.rollout_robustness`、`evaluate.diagnose_mug_visual_robustness`、`evaluate.rollout`，并复用 `tests.test_evaluate.workspace_temp_dir`。

- `PixelPerturbationTests`：覆盖 brightness / contrast / gamma / noise / blur / jpeg 等像素扰动，验证格式保持、可复现性与未知扰动报错。
- `ThinCopyRolloutIntegrationTests.test_image_transform_is_invoked_and_preserves_semantics`：本机 CPU 短 rollout 验证扰动评测注入的 `image_transform` 每步调用且语义保持。

## 云端训练测试

### `test_cloud.py` — 训练命令构建

导入 `cloud.common` 与 `cloud.train`。`CloudTrainingCommandTests` 覆盖：

- `test_smoke_command_locks_dataset_features_and_single_step`：smoke 模式配置、数据集 feature 锁定与单步训练。
- `test_resume_command_restores_state`：resume 命令状态恢复。
- `test_resume_with_smoke_raises`：resume 与 smoke 互斥校验。
- `test_default_dataset_is_inside_project`：默认数据集路径位于项目内。

## 测试依赖分类

| 类别 | 涉及文件 | 说明 |
| --- | --- | --- |
| 需要 MuJoCo 渲染/物理仿真 | `test_scene.py`、`test_mug_scene.py` | 创建真实 `CleanTabletopEnv` / `MugTabletopEnv`，渲染 RGB、推进物理步；建议 `MUJOCO_GL=egl`。 |
| 需要本机 CPU 短 rollout | `test_evaluate.py`、`test_mug_visual_robustness.py` | 以 fake policy 跑极短闭环，写出结果/视频/action trace。 |
| 纯逻辑/契约（无 MuJoCo） | `test_collector.py`、`test_grounding_collection.py`、`test_mug_collector_v3.py`、`test_chunk_blend.py`、`test_cloud.py` | 校验计划、状态机、schema、命令构建与策略纯函数。 |

## 运行方式

```bash
# 运行全部测试（需要 MuJoCo 的用例建议设置 EGL 后端）
MUJOCO_GL=egl python -m pytest tests/

# 按域运行单个文件
python -m pytest tests/test_scene.py
python -m pytest tests/test_mug_collector_v3.py

# 仅运行某个测试类或用例
python -m pytest tests/test_chunk_blend.py::ChunkBlendPolicyTest::test_wrap_angle
```
