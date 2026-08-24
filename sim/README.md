# sim

SmolVLA 本地 MuJoCo 仿真模块。封装参考 ACT 演示布局的 UR10e 桌面场景，提供积木（双目标区域）与杯子（双放置区）两套独立环境，统一管理场景加载、确定性 reset、物理推进、固定相机渲染、机器人状态/动作接口、严格任务判定与单上下文交互式 Viewer。两套环境共享机器人控制、相机渲染与 Viewer 能力，但模型、reset 状态与任务判定各自独立，互不修改。

## 目录结构

| 路径 | 功能 |
| --- | --- |
| `__init__.py` | 包初始化文件，声明本模块为「SmolVLA 本地 MuJoCo 仿真模块」，导出 `CleanTabletopEnv` 与 `MugTabletopEnv`。 |
| `environment.py` | 参照 ACT 演示布局的双积木 MuJoCo 环境封装。 |
| `mug_environment.py` | 独立的 ACT 杯子双放置区 MuJoCo 环境，继承积木环境。 |
| `mujoco_viewer.py` | 与 ACT 相同渲染结构的单上下文 MuJoCo/GLFW Viewer。 |

## `environment.py` — 双积木桌面环境

参照 ACT 演示布局构建的双积木 MuJoCo 环境封装。被 `collector.common.control`、`scripts.verify_act_layout` 引用，并向 `mug_environment.py` 提供基类。

### 常量

- 机器人：`ARM_JOINT_NAMES`（六关节）、`ACT_INITIAL_ARM_QPOS_DEG = [0, -90, 90, -90, -90, 90]`、`GRIPPER_RELEASED_QPOS = 0.1`。
- 相机：`DISPLAY_CAMERA_NAMES = ("agentview", "d435i_rgb", "sideview")`。
- 任务对象：`TASK_CUBE_BODY_NAMES` / `TASK_CUBE_GEOM_NAMES`（红、绿积木）、`TASK_PAD_BODY_NAMES` / `TASK_PAD_GEOM_NAMES`（蓝、黄放置区）、`CUBE_HALF_SIZE` / `PAD_HALF_SIZE`、`TASK_INITIAL_BODY_POSITIONS`（四对象固定坐标）。
- 随机化：`CUBE_SAMPLE_X_RANGE` / `CUBE_SAMPLE_Y_RANGE`、`CUBE_MIN_CENTER_DISTANCE = 0.12`、`CUBE_INITIAL_Z = 0.825`、`MAX_RESET_ATTEMPTS = 100`。
- 任务与判定：`TASK_IDS`（`red_on_blue` / `red_on_yellow` / `green_on_blue` / `green_on_yellow`）、`TASK_OBJECTS`（任务到目标/干扰积木与区域的映射）、`PLACEMENT_INSET = 0.005`、`STABLE_LINEAR_SPEED` / `STABLE_ANGULAR_SPEED` / `STABLE_DURATION_SECONDS = 0.5`。
- 类型别名：`RgbImage = NDArray[np.uint8]`。

### 数据类

- `SceneSnapshot`：reset 后的任务场景快照，含 `scene_seed`、`cube_initial_poses`（红绿积木 `xyz+quaternion`，形状 `(2, 7)`）、`pad_positions`（蓝黄固定坐标，形状 `(2, 3)`）。
- `TaskEvaluation`：严格成功状态与失败分类，含 `success`、`failure_mode`（`success` / `in_progress` / `wrong_cube` / `wrong_pad` / `dropped_or_out_of_bounds` / `timeout` / `control_exception`）、`metrics`。

### 模块级函数

- `_load_model_from_asset_bundle(asset_root)`：把主 XML、include XML、mesh 与纹理读成内存资源包再交给 MuJoCo 编译，规避 MuJoCo 3.6.0 Windows 原生接口无法稳定读取中文绝对路径的问题，不重写 XML。
- `_require_object_id(model, object_type, name)`：按名称查询 MuJoCo 对象编号，缺失时抛 `ValueError`。

### `CleanTabletopEnv` — 双积木环境主类

- 构造：`__init__(project_root=None, image_size=(256, 256))`，自动定位 `assets/mujoco/scene.xml`，加载模型并 `reset(scene_seed=0)`；支持 `with` 上下文管理（`__enter__` / `__exit__` / `close`）。
- 场景重置：`reset(scene_seed=0) -> SceneSnapshot`，红绿积木只随机平面位置（姿态恒为单位四元数、中心距离 ≥ 12 cm、无非法初始接触），蓝黄区固定；100 次仍不合法抛 `RuntimeError`。`scene_snapshot()` 返回防御性副本。
- 物理推进：`step(steps=1)`，每步 `mj_step`，出现 NaN/Inf 抛 `RuntimeError`。
- 相机渲染：`capture_cameras()`（前视/腕部/侧视三路）、`capture_camera(name)`（单路）、`capture_training_images()`（`agent`/`wrist` 两路训练图像），均输出 `256×256×3` RGB uint8。
- 状态与动作：`get_state()`（六关节角 + 夹爪 0/1，七维 float32）、`get_end_effector_position()`、`get_arm_qvel()`、`apply_joint_action(action, physics_steps=0)`（校验七维有限、夹爪 ∈ [0,1]、关节角在 ctrlrange 内）。
- 任务判定：`evaluate_task(task_id, elapsed_seconds=None, timeout_seconds=None) -> TaskEvaluation`，目标积木完整位于区内缩边界、稳定 0.5 s、松爪且无干扰积木/错区/越界时判成功。
- 布局查询：`spatial_layout()`（机械臂与桌面编译后世界位姿）、`task_layout()`（积木与区域只读快照）。
- 交互 Viewer：`run(max_seconds=None, display_hz=60.0, show_camera_panel=True)`，60 Hz 显示、500 Hz 物理累加推进，返回显示时长/帧数/平均 FPS。

## `mug_environment.py` — 杯子双放置区环境

独立的 ACT 杯子双放置区 MuJoCo 环境。`MugTabletopEnv` 继承 `CleanTabletopEnv` 已验证的机器人控制、相机渲染、空间布局与 Viewer 能力，但使用独立杯子模型、reset 状态与任务判定，不修改积木环境或原场景。被 `collector.v3.*` 与 `scripts`（经 collector 间接）引用。

### 常量

- 杯子对象：`MUG_BODY_NAME = "body_obj_mug_5"`、`MUG_BOTTOM_SITE_NAME` / `MUG_TOP_SITE_NAME`、`MUG_PAD_BODY_NAMES` / `MUG_PAD_GEOM_NAMES`。
- 任务：`MUG_TASK_IDS = ("mug_on_blue", "mug_on_yellow")`、`MUG_TASK_PADS`（任务到目标/干扰区映射）。
- 外观：`MUG_TEXTURE_ASSET_KEY`、`MUG_APPEARANCE_TEXTURES`（`original` / `green_white` / `changed` 三种纹理变体）。
- 随机化：`MUG_SAMPLE_X_RANGE` / `MUG_SAMPLE_Y_RANGE`、`MUG_INITIAL_Z = 0.86`、`MUG_SETTLE_STEPS = 250`、`MUG_MAX_RESET_ATTEMPTS = 100`、`MUG_FOOTPRINT_RADIUS = 0.071`、`MUG_INITIAL_PAD_CLEARANCE = 0.01`。
- 判定（与积木环境同语义）：`PAD_HALF_SIZE`、`PLACEMENT_INSET = 0.01`、`STABLE_LINEAR_SPEED` / `STABLE_ANGULAR_SPEED` / `STABLE_DURATION_SECONDS = 0.5`。

### 数据类

- `MugSceneSnapshot`：`scene_seed`、`mug_initial_pose`（稳定后 `xyz+quaternion`，形状 `(7,)`）、`pad_positions`（蓝黄固定坐标 `(2, 3)`）。
- `MugTaskEvaluation`：`success`、`failure_mode`（`success` / `in_progress` / `wrong_pad` / `dropped_or_out_of_bounds` / `timeout` / `control_exception`）、`metrics`。

### 模块级函数

- `resolve_mug_texture_path(asset_root, appearance_variant)`：解析杯子外观变体对应的纹理路径，校验存在性。
- `_load_mug_model(asset_root, appearance_variant="original")`：从内存资源包加载独立杯子场景（杯子目录递归入包，确保 32 个碰撞网格可解析），XML 始终引用原始资源键、仅替换内存纹理字节。

### `MugTabletopEnv` — 杯子环境主类（继承 `CleanTabletopEnv`）

- 构造：`__init__(project_root=None, image_size=(256, 256), appearance_variant="original")`，加载 `assets/mujoco/mug_scene.xml` 与指定外观纹理。
- 场景重置：`reset(scene_seed=0) -> MugSceneSnapshot`，每次候选从干净动力学开始，杯子以单位四元数从 `z=0.86` 落下并推进 250 步等待稳定；校验位置/接触/数值合法，稳定后精确恢复 ACT 关节角、清零速度、仿真时间归零。100 次仍不合法抛 `RuntimeError`。
- 任务判定：`evaluate_task(task_id, ...) -> MugTaskEvaluation`，杯子中心在目标区内、底面落桌高度正确、保持直立、稳定 0.5 s 且松爪时判成功；落在另一区域判 `wrong_pad`。
- 布局查询：`task_layout()`（杯子位姿/质量/速度与上下边界 site，区域尺寸/颜色/碰撞掩码）。
- 私有校验：`_mug_ids`（经根 body 的 `body_jntadr` 定位未命名 free joint）、`_has_mug_robot_contact`、`_too_close_to_pad`、`_settled_scene_error`、`_is_mug_inside_pad`（中心/落桌高度/直立三条件）。
- 继承自 `CleanTabletopEnv`：`capture_cameras` / `capture_training_images` / `get_state` / `apply_joint_action` / `run` / `step` 等在杯子模型上直接可用。

## `mujoco_viewer.py` — 单上下文 MuJoCo Viewer

与 ACT 渲染结构一致的单上下文 MuJoCo/GLFW Viewer。主视角与三路固定相机共用同一个 `MjrContext`，直接把固定相机渲染到子 viewport，省去 `mjr_readPixels → CPU → mjr_drawPixels` 往返复制。被 `CleanTabletopEnv.run` 与采集入口（`collector.v3.collect_matrix` 等）引用。

### `EmbeddedCameraViewer`

- 构造：`__init__(model, data, title="SmolVLA ACT Clean Tabletop", width=1400, height=1000, show_fixed_cameras=True)`，创建 GLFW 窗口、`MjvOption` / `MjvPerturb` / `MjvCamera` / `MjvScene` / `MjrContext`，初始化与 ACT `y_env.init_viewer` 一致的自由相机参数（`azimuth=170`、`distance=2.0`、`elevation=-30`、`lookat=[0.01, 0.11, 0.5]`），定位 `agentview` / `d435i_rgb` / `sideview` 三路固定相机；支持 `with` 上下文管理。
- 交互输入：`is_key_down(key)`（持续按下查询）、`consume_key_press(key)`（一次性按键事件消费，供采集 Enter/Backspace/Z 判定）、`set_status(title, text)`（左下角状态叠加）、鼠标拖拽旋转/平移、滚轮缩放、ESC 关闭。
- 渲染：`render()` 刷新主场景与三路固定相机子 viewport（`sideview` 左下、`agentview` 右下、`d435i_rgb` 右上），叠加视图标签与状态文本；`_render_camera` 关闭天空盒（与 ACT `black_sky=True` 对应）。
- 训练图像采集：`capture_training_images(image_size=256)`，在当前 GLFW 上下文读取 `agent` / `wrist` 两路 RGB uint8 图像（`mjr_readPixels` + `np.flipud`）。
- 生命周期：`is_running()`、`close()`（释放 `MjrContext` 并销毁 GLFW 窗口）。

## 模块调用关系

```
mujoco_viewer.py ── EmbeddedCameraViewer ──┐
                                           │
environment.py ── CleanTabletopEnv ────────┤  (CleanTabletopEnv.run 引用 EmbeddedCameraViewer)
        │                                  │
        └── mug_environment.py             │
            └── MugTabletopEnv(extends) ───┘

__init__.py ── 导出 CleanTabletopEnv / MugTabletopEnv

外部引用：
  collector.common.control ─── sim.environment (ARM_JOINT_NAMES, CleanTabletopEnv)
  collector.v3.* ────────────── sim.mug_environment (MugTabletopEnv, MugSceneSnapshot)
  scripts.verify_act_layout ── sim.environment (CleanTabletopEnv, ACT_INITIAL_ARM_QPOS_DEG, ARM_JOINT_NAMES)
```

## 典型用法

```python
from sim import CleanTabletopEnv, MugTabletopEnv

# 双积木环境：随机化场景并渲染训练图像
with CleanTabletopEnv() as env:
    snapshot = env.reset(scene_seed=9)
    images = env.capture_training_images()           # {"agent": ..., "wrist": ...}
    state = env.get_state()                          # 七维 float32
    env.apply_joint_action(action, physics_steps=1)
    result = env.evaluate_task("red_on_blue")
    env.run(max_seconds=10.0)                        # 打开交互式 Viewer

# 杯子环境：等待杯子稳定并评估放置任务
with MugTabletopEnv(appearance_variant="original") as env:
    snapshot = env.reset(scene_seed=3)
    result = env.evaluate_task("mug_on_yellow")
```
