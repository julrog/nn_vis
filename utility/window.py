from typing import Any, Dict, Tuple

import glfw
from OpenGL.GL import glViewport
from pyrr import Vector3

from definitions import CameraPose
from utility.camera import Camera
from utility.singleton import Singleton
from utility.window_config import WindowConfig


class Window:
    def __init__(self, config: WindowConfig) -> None:
        self.config: WindowConfig = config

        # glfw.window_hint(glfw.DECORATED, glfw.FALSE)
        self.window_handle = glfw.create_window(
            config['width'], config['height'], config['title'], None, None)

        if not self.window_handle:
            raise Exception('glfw window can not be created!')
        self.active: bool = False
        self.cam: Camera = Camera(self.config['width'],
                                  self.config['height'],
                                  Vector3([0.0, 0.0, 0.0]),
                                  rotation=self.config['camera_rotation'],
                                  rotation_speed=self.config['camera_rotation_speed'])

        self.last_mouse_pos: Tuple[int, int] = (
            int(self.config['width'] / 2), int(self.config['height'] / 2))
        self.mouse_set: bool = False
        self.freeze: bool = True
        self.gradient: bool = True
        self.mouse_captured: bool = False
        self.focused: bool = True
        self.screenshot: bool = False
        self.record: bool = False
        self.frame_id: int = 0

    def set_size(self, width: float, height: float) -> None:
        self.config['width'] = width
        self.config['height'] = height
        if self.active:
            glViewport(0, 0, width, height)
        self.cam.set_size(width, height)

    def set_callbacks(self) -> None:
        def resize_clb(_: Any, width: float, height: float) -> None:
            self.config['screen_width'] = width
            self.config['screen_height'] = height
            self.config.store()

        def frame_resize_clb(_: Any, width: float, height: float) -> None:
            self.set_size(width, height)

        def mouse_look_clb(_: Any, x_pos: int, y_pos: int) -> None:
            if not self.focused or not self.mouse_captured:
                return

            if not self.mouse_set:
                self.last_mouse_pos = (x_pos, y_pos)
                self.mouse_set = True

            x_offset: float = x_pos - self.last_mouse_pos[0]
            y_offset: float = self.last_mouse_pos[1] - y_pos

            self.last_mouse_pos = (x_pos, y_pos)
            self.cam.process_mouse_movement(x_offset, y_offset)

        def mouse_button_clb(_: Any, button: int, action: int, mods: int) -> None:
            if not self.focused:
                return
            if button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS:
                self.toggle_mouse_capture()

        def focus_clb(_, focused: int) -> None:
            if focused:
                self.focused = True
            else:
                self.focused = False

        def window_pos_clb(_: Any, x_pos: int, y_pos: int) -> None:
            if len(glfw.get_monitors()) >= 1:
                for monitor_id, monitor in enumerate(glfw.get_monitors()):
                    m_x, m_y, width, height = glfw.get_monitor_workarea(
                        monitor)
                    if m_x <= x_pos < m_x + width and m_y <= y_pos < m_y + height:
                        self.config['monitor_id'] = monitor_id
            self.config['screen_x'] = x_pos
            self.config['screen_y'] = y_pos
            self.config.store()

        def key_input_clb(glfw_window, key, scancode, action, mode) -> None:
            if not self.focused:
                return
            if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
                glfw.set_window_should_close(glfw_window, True)
            if key == glfw.KEY_W and action == glfw.PRESS:
                self.cam.move(Vector3([0, 0, 1]))
            elif key == glfw.KEY_W and action == glfw.RELEASE:
                self.cam.stop(Vector3([0, 0, 1]))
            if key == glfw.KEY_S and action == glfw.PRESS:
                self.cam.move(Vector3([0, 0, -1]))
            elif key == glfw.KEY_S and action == glfw.RELEASE:
                self.cam.stop(Vector3([0, 0, -1]))
            if key == glfw.KEY_A and action == glfw.PRESS:
                self.cam.move(Vector3([-1, 0, 0]))
            elif key == glfw.KEY_A and action == glfw.RELEASE:
                self.cam.stop(Vector3([-1, 0, 0]))
            if key == glfw.KEY_D and action == glfw.PRESS:
                self.cam.move(Vector3([1, 0, 0]))
            elif key == glfw.KEY_D and action == glfw.RELEASE:
                self.cam.stop(Vector3([1, 0, 0]))

            if key == glfw.KEY_F and action == glfw.RELEASE:
                self.freeze = not self.freeze
            if key == glfw.KEY_G and action == glfw.RELEASE:
                self.gradient = not self.gradient
            if key == glfw.KEY_H and action == glfw.RELEASE:
                self.cam.rotate_around_base = not self.cam.rotate_around_base
                self.config['camera_rotation'] = self.cam.rotate_around_base
                self.config.store()

            if key == glfw.KEY_K and action == glfw.RELEASE:
                self.screenshot = True
            if key == glfw.KEY_R and action == glfw.RELEASE:
                self.record = not self.record

            if key == glfw.KEY_0 and action == glfw.RELEASE:
                self.cam.set_position(CameraPose.LEFT)
            if key == glfw.KEY_1 and action == glfw.RELEASE:
                self.cam.set_position(CameraPose.FRONT)
            if key == glfw.KEY_2 and action == glfw.RELEASE:
                self.cam.set_position(CameraPose.RIGHT)
            if key == glfw.KEY_3 and action == glfw.RELEASE:
                self.cam.set_position(CameraPose.BACK_RIGHT)
            if key == glfw.KEY_4 and action == glfw.RELEASE:
                self.cam.set_position(CameraPose.BACK_RIGHT)
            if key == glfw.KEY_5 and action == glfw.RELEASE:
                self.cam.set_position(CameraPose.UPPER_BACK_LEFT)
            if key == glfw.KEY_6 and action == glfw.RELEASE:
                self.cam.set_position(CameraPose.UPPER_BACK_RIGHT)
            if key == glfw.KEY_7 and action == glfw.RELEASE:
                self.cam.set_position(CameraPose.LOWER_BACK_RIGHT)
            if key == glfw.KEY_8 and action == glfw.RELEASE:
                self.cam.set_position(CameraPose.BACK)
            if key == glfw.KEY_9 and action == glfw.RELEASE:
                self.cam.set_position(CameraPose.LEFT)

        glfw.set_window_size_callback(self.window_handle, resize_clb)
        glfw.set_framebuffer_size_callback(
            self.window_handle, frame_resize_clb)
        glfw.set_cursor_pos_callback(self.window_handle, mouse_look_clb)
        glfw.set_key_callback(self.window_handle, key_input_clb)
        glfw.set_mouse_button_callback(self.window_handle, mouse_button_clb)
        glfw.set_window_focus_callback(self.window_handle, focus_clb)
        glfw.set_window_pos_callback(self.window_handle, window_pos_clb)

    def activate(self) -> None:
        if self.config['monitor_id'] is not None and 0 <= self.config['monitor_id'] < len(glfw.get_monitors()):
            glfw.set_window_pos(
                self.window_handle, self.config['screen_x'], self.config['screen_y'])
        elif self.config['monitor_id'] is not None:
            self.config['screen_x'] = 0
            self.config['screen_y'] = 0
            glfw.set_window_pos(self.window_handle, 0, 0)

        glfw.make_context_current(self.window_handle)
        glfw.set_input_mode(self.window_handle,
                            glfw.CURSOR, glfw.CURSOR_NORMAL)
        glViewport(0, 0, self.config['width'], self.config['height'])
        self.active = True

    def is_active(self) -> bool:
        return not glfw.window_should_close(self.window_handle)

    def swap(self) -> None:
        glfw.swap_buffers(self.window_handle)

    def destroy(self) -> None:
        glfw.destroy_window(self.window_handle)

    def update(self) -> None:
        self.cam.update()

    def toggle_mouse_capture(self) -> None:
        if self.mouse_captured:
            self.mouse_set = False
            glfw.set_input_mode(self.window_handle,
                                glfw.CURSOR, glfw.CURSOR_NORMAL)
        else:
            self.mouse_set = False
            glfw.set_input_mode(self.window_handle,
                                glfw.CURSOR, glfw.CURSOR_DISABLED)
        self.mouse_captured = not self.mouse_captured


class WindowHandler(metaclass=Singleton):
    def __init__(self) -> None:
        self.windows: Dict[str, Window] = dict()

        if not glfw.init():
            raise Exception('glfw can not be initialized!')

    def create_window(self, hidden: bool = False) -> Window:
        if hidden:
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        window_config: WindowConfig = WindowConfig()
        window = Window(window_config)

        if self.windows.get(window_config['title']):
            self.windows[window_config['title']].destroy()

        self.windows[window_config['title']] = window
        return window

    def get_window(self, title: str) -> Window:
        window = self.windows[title]
        if not window:
            raise Exception('Requested window does not exist!')
        return window

    def destroy(self) -> None:
        for _, window in self.windows.items():
            if window.is_active():
                window.destroy()
        glfw.terminate()

    def update(self) -> None:
        glfw.poll_events()
        for _, window in self.windows.items():
            window.update()
