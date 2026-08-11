"""与 ACT 相同渲染结构的单上下文 MuJoCo Viewer。"""

from __future__ import annotations

from collections import OrderedDict

import glfw
import mujoco
import numpy as np


class EmbeddedCameraViewer:
    """在一个 MuJoCo/GLFW 窗口内渲染主视角和固定相机。

    ACT GUI路径使用同一个 ``MjrContext`` 完成主场景、固定相机和RGB叠加。
    本实现保留这一关键结构，但直接把固定相机画到子viewport，省去原实现
    中 ``mjr_readPixels -> CPU图像 -> mjr_drawPixels`` 的往返复制。

    Args:
        model: 需要展示的MuJoCo模型。
        data: 与模型关联的MuJoCo运行时数据。
        title: GLFW窗口标题。
        width: 初始窗口宽度。
        height: 初始窗口高度。
        show_fixed_cameras: 是否显示三路固定相机子viewport。
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        title: str = "SmolVLA ACT Clean Tabletop",
        width: int = 1400,
        height: int = 1000,
        show_fixed_cameras: bool = True,
    ) -> None:
        """创建GLFW窗口、MuJoCo场景和单一渲染上下文。

        Args:
            model: 需要展示的MuJoCo模型。
            data: 与模型关联的MuJoCo运行时数据。
            title: GLFW窗口标题。
            width: 初始窗口宽度。
            height: 初始窗口高度。
            show_fixed_cameras: 是否显示三路固定相机。

        Raises:
            RuntimeError: GLFW初始化或窗口创建失败时抛出。
        """
        self.model = model
        self.data = data
        self.show_fixed_cameras = show_fixed_cameras
        self._closed = False
        self._left_pressed = False
        self._right_pressed = False
        self._last_cursor = (0.0, 0.0)
        self._pressed_keys: set[int] = set()
        self._key_press_events: set[int] = set()
        self.status_title = ""
        self.status_text = ""

        if not glfw.init():
            raise RuntimeError("GLFW初始化失败，无法创建MuJoCo Viewer")
        self.window = glfw.create_window(width, height, title, None, None)
        if self.window is None:
            glfw.terminate()
            raise RuntimeError("GLFW窗口创建失败")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor_move)
        glfw.set_scroll_callback(self.window, self._on_scroll)

        self.option = mujoco.MjvOption()
        self.perturb = mujoco.MjvPerturb()
        self.main_camera = mujoco.MjvCamera()
        self.scene = mujoco.MjvScene(model, maxgeom=10_000)
        self.context = mujoco.MjrContext(
            model,
            mujoco.mjtFontScale.mjFONTSCALE_150.value,
        )

        # 与ACT y_env.init_viewer保持一致的自由相机参数。
        self.main_camera.azimuth = 170.0
        self.main_camera.distance = 2.0
        self.main_camera.elevation = -30.0
        self.main_camera.lookat[:] = np.array([0.01, 0.11, 0.5])

        self.fixed_cameras: "OrderedDict[str, mujoco.MjvCamera]" = OrderedDict()
        for camera_name in ("agentview", "d435i_rgb", "sideview"):
            camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if camera_id < 0:
                self.close()
                raise RuntimeError(f"场景缺少固定相机: {camera_name}")
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            camera.fixedcamid = camera_id
            self.fixed_cameras[camera_name] = camera

    def __enter__(self) -> "EmbeddedCameraViewer":
        """返回当前Viewer以支持上下文管理。

        Returns:
            当前Viewer实例。
        """
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """退出上下文时释放MuJoCo和GLFW资源。

        Args:
            exc_type: 上下文异常类型。
            exc_value: 上下文异常对象。
            traceback: 上下文异常堆栈。
        """
        del exc_type, exc_value, traceback
        self.close()

    def _on_key(
        self,
        window: object,
        key: int,
        scancode: int,
        action: int,
        modifiers: int,
    ) -> None:
        """处理ESC键关闭窗口。

        Args:
            window: GLFW窗口句柄。
            key: GLFW键码。
            scancode: 平台扫描码。
            action: 按下、松开或重复动作。
            modifiers: Shift、Ctrl等修饰键状态。
        """
        del scancode, modifiers
        if action == glfw.PRESS:
            self._pressed_keys.add(key)
            self._key_press_events.add(key)
        elif action == glfw.RELEASE:
            self._pressed_keys.discard(key)
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)

    def is_key_down(self, key: int) -> bool:
        """查询按键当前是否持续按下。

        Args:
            key: GLFW键码。

        Returns:
            按键处于按下状态时返回 ``True``。
        """
        return key in self._pressed_keys

    def consume_key_press(self, key: int) -> bool:
        """消费一次只触发一次的按键事件。

        Args:
            key: GLFW键码。

        Returns:
            本次查询消费到按下事件时返回 ``True``。
        """
        if key not in self._key_press_events:
            return False
        self._key_press_events.remove(key)
        return True

    def set_status(self, title: str, text: str) -> None:
        """设置下一帧显示在Viewer左下角的采集状态。

        Args:
            title: 状态标题。
            text: 可包含换行的状态正文。
        """
        self.status_title = title
        self.status_text = text

    def _on_mouse_button(self, window: object, button: int, action: int, modifiers: int) -> None:
        """记录鼠标按键状态，供自由相机拖拽使用。

        Args:
            window: GLFW窗口句柄。
            button: 鼠标按键编号。
            action: 按下或松开动作。
            modifiers: 键盘修饰键状态。
        """
        del modifiers
        pressed = action == glfw.PRESS
        if button == glfw.MOUSE_BUTTON_LEFT:
            self._left_pressed = pressed
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self._right_pressed = pressed
        self._last_cursor = glfw.get_cursor_pos(window)

    def _on_cursor_move(self, window: object, x_position: float, y_position: float) -> None:
        """根据鼠标拖拽旋转或平移主视角。

        Args:
            window: GLFW窗口句柄。
            x_position: 当前光标横坐标。
            y_position: 当前光标纵坐标。
        """
        if not self._left_pressed and not self._right_pressed:
            self._last_cursor = (x_position, y_position)
            return

        previous_x, previous_y = self._last_cursor
        delta_x = x_position - previous_x
        delta_y = y_position - previous_y
        self._last_cursor = (x_position, y_position)
        _, window_height = glfw.get_window_size(window)
        if window_height <= 0:
            return

        shift_pressed = any(
            glfw.get_key(window, key) == glfw.PRESS
            for key in (glfw.KEY_LEFT_SHIFT, glfw.KEY_RIGHT_SHIFT)
        )
        if self._right_pressed:
            mouse_action = (
                mujoco.mjtMouse.mjMOUSE_MOVE_H
                if shift_pressed
                else mujoco.mjtMouse.mjMOUSE_MOVE_V
            )
        else:
            mouse_action = (
                mujoco.mjtMouse.mjMOUSE_ROTATE_H
                if shift_pressed
                else mujoco.mjtMouse.mjMOUSE_ROTATE_V
            )
        mujoco.mjv_moveCamera(
            self.model,
            mouse_action,
            delta_x / window_height,
            delta_y / window_height,
            self.scene,
            self.main_camera,
        )

    def _on_scroll(self, window: object, x_offset: float, y_offset: float) -> None:
        """使用滚轮缩放主视角。

        Args:
            window: GLFW窗口句柄。
            x_offset: 横向滚轮偏移。
            y_offset: 纵向滚轮偏移。
        """
        del window, x_offset
        mujoco.mjv_moveCamera(
            self.model,
            mujoco.mjtMouse.mjMOUSE_ZOOM,
            0.0,
            -0.05 * y_offset,
            self.scene,
            self.main_camera,
        )

    def is_running(self) -> bool:
        """检查窗口是否仍处于运行状态。

        Returns:
            窗口存在且未收到关闭请求时返回 ``True``。
        """
        return not self._closed and not glfw.window_should_close(self.window)

    @staticmethod
    def _camera_viewports(width: int, height: int) -> "OrderedDict[str, mujoco.MjrRect]":
        """计算与ACT叠加位置一致的三个相机子viewport。

        Args:
            width: 当前framebuffer宽度。
            height: 当前framebuffer高度。

        Returns:
            ``sideview``左上、``agentview``右上、``d435i_rgb``右下的viewport。
        """
        overlay_width = max(1, width // 4)
        overlay_height = max(1, height // 4)
        return OrderedDict(
            (
                ("sideview", mujoco.MjrRect(0, height - overlay_height, overlay_width, overlay_height)),
                (
                    "agentview",
                    mujoco.MjrRect(
                        width - overlay_width,
                        height - overlay_height,
                        overlay_width,
                        overlay_height,
                    ),
                ),
                (
                    "d435i_rgb",
                    mujoco.MjrRect(width - overlay_width, 0, overlay_width, overlay_height),
                ),
            )
        )

    def _render_camera(self, camera: mujoco.MjvCamera, viewport: mujoco.MjrRect) -> None:
        """使用当前MuJoCo上下文直接渲染一个相机viewport。

        Args:
            camera: 主自由相机或模型固定相机。
            viewport: 该相机在GLFW framebuffer中的目标区域。
        """
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.option,
            self.perturb,
            camera,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scene,
        )
        # 与ACT的black_sky=True完全对应：只关闭Viewer天空盒，不改源XML。
        self.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False
        mujoco.mjr_render(viewport, self.scene, self.context)

    def capture_training_images(self, image_size: int = 256) -> "OrderedDict[str, np.ndarray]":
        """在当前GLFW上下文读取第三方与腕部两路RGB图像。

        Args:
            image_size: 方形输出图像边长。

        Returns:
            按 ``agent``、``wrist`` 排列的RGB ``uint8`` 图像。

        Raises:
            ValueError: 图像边长不是正整数时抛出。
        """
        if image_size <= 0:
            raise ValueError(f"image_size 必须为正整数，实际为 {image_size}")
        glfw.make_context_current(self.window)
        viewport = mujoco.MjrRect(0, 0, image_size, image_size)
        images: "OrderedDict[str, np.ndarray]" = OrderedDict()
        for output_name, camera_name in (("agent", "agentview"), ("wrist", "d435i_rgb")):
            self._render_camera(self.fixed_cameras[camera_name], viewport)
            rgb = np.empty((image_size, image_size, 3), dtype=np.uint8)
            mujoco.mjr_readPixels(rgb, None, viewport, self.context)
            images[output_name] = np.flipud(rgb).copy()
        return images

    def render(self) -> None:
        """在同一MuJoCo窗口中刷新主场景和三路固定相机。"""
        glfw.make_context_current(self.window)
        width, height = glfw.get_framebuffer_size(self.window)
        main_viewport = mujoco.MjrRect(0, 0, width, height)
        self._render_camera(self.main_camera, main_viewport)

        if self.show_fixed_cameras:
            for camera_name, viewport in self._camera_viewports(width, height).items():
                self._render_camera(self.fixed_cameras[camera_name], viewport)

            labels = (
                (mujoco.mjtGridPos.mjGRID_TOPLEFT, "Side View"),
                (mujoco.mjtGridPos.mjGRID_TOPRIGHT, "Agent View"),
                (mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT, "Egocentric View"),
            )
            for grid_position, label in labels:
                mujoco.mjr_overlay(
                    mujoco.mjtFontScale.mjFONTSCALE_100,
                    grid_position,
                    main_viewport,
                    label,
                    "",
                    self.context,
                )

        if self.status_title or self.status_text:
            mujoco.mjr_overlay(
                mujoco.mjtFontScale.mjFONTSCALE_100,
                mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                main_viewport,
                self.status_title,
                self.status_text,
                self.context,
            )

        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def close(self) -> None:
        """释放MuJoCo渲染上下文并关闭GLFW窗口。"""
        if self._closed:
            return
        self._closed = True
        self.context.free()
        glfw.destroy_window(self.window)
        glfw.terminate()
