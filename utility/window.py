from typing import Dict, Tuple
import glfw
from OpenGL.GL import *
from pyrr import Vector3

from utility.camera import Camera
from utility.singleton import Singleton


class Window:
    def __init__(self, settings: Dict[str, any]):
        self.settings: Dict[str, any] = settings

        # glfw.window_hint(glfw.DECORATED, glfw.FALSE)
        self.window_handle = glfw.create_window(settings["width"], settings["height"], settings["title"],
                                                None,
                                                settings.get("share"))

        if not self.window_handle:
            raise Exception("glfw window can not be created!")
        self.width: float = settings["width"]
        self.height: float = settings["height"]
        self.active: bool = False
        self.cam: Camera = Camera(self.width, self.height)

        self.last_mouse_pos: Tuple[int, int] = (int(self.width / 2), int(self.height / 2))
        self.mouse_set: bool = False
        self.freeze: bool = True
        self.gradient: bool = True
        self.render_method: int = 0
        self.mouse_captured: bool = False
        self.focused: bool = True

    def set_position(self, x: float, y: float):
        glfw.set_window_pos(self.window_handle, x, y)

    def set_size(self, width: float, height: float):
        self.width = width
        self.height = height
        if self.active:
            glViewport(0, 0, width, height)
        self.cam.set_size(width, height)

    def set_callbacks(self):
        def resize_clb(glfw_window, width, height):
            self.set_size(width, height)

        def mouse_look_clb(glfw_window, x_pos, y_pos):
            if not self.focused or not self.mouse_captured:
                return

            if not self.mouse_set:
                self.last_mouse_pos = (x_pos, y_pos)
                self.mouse_set = True

            x_offset: float = x_pos - self.last_mouse_pos[0]
            y_offset: float = self.last_mouse_pos[1] - y_pos

            self.last_mouse_pos = (x_pos, y_pos)
            self.cam.process_mouse_movement(x_offset, y_offset)

        def mouse_button_callback(glfw_window, button: int, action: int, mods: int):
            if not self.focused:
                return
            if button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS:
                self.toggle_mouse_capture()

        def focus_callback(glfw_window, focused: int):
            if focused:
                self.focused = True
            else:
                self.focused = False

        def key_input_clb(glfw_window, key, scancode, action, mode):
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

            if key == glfw.KEY_1 and action == glfw.RELEASE:
                self.render_method = 1
            if key == glfw.KEY_2 and action == glfw.RELEASE:
                self.render_method = 2
            if key == glfw.KEY_3 and action == glfw.RELEASE:
                self.render_method = 3
            if key == glfw.KEY_4 and action == glfw.RELEASE:
                self.render_method = 4
            if key == glfw.KEY_5 and action == glfw.RELEASE:
                self.render_method = 5
            if key == glfw.KEY_6 and action == glfw.RELEASE:
                self.render_method = 6
            if key == glfw.KEY_7 and action == glfw.RELEASE:
                self.render_method = 7
            if key == glfw.KEY_8 and action == glfw.RELEASE:
                self.render_method = 8
            if key == glfw.KEY_9 and action == glfw.RELEASE:
                self.render_method = 9

        glfw.set_window_size_callback(self.window_handle, resize_clb)
        glfw.set_cursor_pos_callback(self.window_handle, mouse_look_clb)
        glfw.set_key_callback(self.window_handle, key_input_clb)
        glfw.set_mouse_button_callback(self.window_handle, mouse_button_callback)
        glfw.set_window_focus_callback(self.window_handle, focus_callback)

    def activate(self):
        if self.settings.get("monitor") is not None and 0 < self.settings.get("monitor") < len(glfw.get_monitors()):
            monitor = glfw.get_monitors()[self.settings.get("monitor")]
            m_x, m_y = glfw.get_monitor_pos(monitor)
            glfw.set_window_pos(self.window_handle, m_x, m_y)

        glfw.make_context_current(self.window_handle)
        glfw.set_input_mode(self.window_handle, glfw.CURSOR, glfw.CURSOR_NORMAL)
        glViewport(0, 0, self.width, self.height)
        self.active = True

    def is_active(self) -> bool:
        return not glfw.window_should_close(self.window_handle)

    def swap(self):
        glfw.swap_buffers(self.window_handle)

    def destroy(self):
        glfw.destroy_window(self.window_handle)

    def update(self):
        self.cam.update()

    def toggle_mouse_capture(self):
        if self.mouse_captured:
            self.mouse_set = False
            glfw.set_input_mode(self.window_handle, glfw.CURSOR, glfw.CURSOR_NORMAL)
        else:
            self.mouse_set = False
            glfw.set_input_mode(self.window_handle, glfw.CURSOR, glfw.CURSOR_DISABLED)
        self.mouse_captured = not self.mouse_captured


class WindowHandler(metaclass=Singleton):
    def __init__(self):
        self.windows: Dict[str, Window] = dict()

        if not glfw.init():
            raise Exception("glfw can not be initialized!")

    def create_window(self, title: str, width: float, height: float, monitor: int = None):
        settings: Dict[str, any] = dict()
        settings["title"] = title
        settings["width"] = width
        settings["height"] = height
        if monitor is not None:
            settings["monitor"] = monitor
        window = Window(settings)

        if self.windows.get(title):
            self.windows[title].destroy()

        self.windows[title] = window
        return window

    def get_window(self, title: str):
        window = self.windows[title]
        if not window:
            raise Exception("Requested window does not exist!")
        return window

    def destroy(self):
        for _, window in self.windows.items():
            if window.is_active():
                window.destroy()
        glfw.terminate()

    def update(self):
        glfw.poll_events()
        for _, window in self.windows.items():
            window.update()
