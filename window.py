from typing import Dict
import glfw


class Window:
    def __init__(self, settings: Dict[str, int]):
        self.window_handle = glfw.create_window(settings["width"], settings["height"], settings["title"],
                                                settings.get("monitor"),
                                                settings.get("share"))

        if not self.window_handle:
            raise Exception("glfw window can not be created!")

    def set_position(self, x, y):
        glfw.set_window_pos(self.window_handle, x, y)

    def set_callbacks(self, resize_clb=None, mouse_look_clb=None, key_input_clb=None):
        glfw.set_window_size_callback(self.window_handle, resize_clb)
        glfw.set_cursor_pos_callback(self.window_handle, mouse_look_clb)
        glfw.set_key_callback(self.window_handle, key_input_clb)

    def activate(self):
        glfw.make_context_current(self.window_handle)
        glfw.set_input_mode(self.window_handle, glfw.CURSOR, glfw.CURSOR_DISABLED)

    def active(self):
        return not glfw.window_should_close(self.window_handle)

    def swap(self):
        glfw.swap_buffers(self.window_handle)

    def destroy(self):
        glfw.destroy_window(self.window_handle)


class WindowHandler:
    def __init__(self):
        self.windows: Dict[str, Window] = dict()

        if not glfw.init():
            raise Exception("glfw can not be initialized!")

    def create_window(self, title, width, height):
        settings = dict()
        settings["title"] = title
        settings["width"] = width
        settings["height"] = height
        window = Window(settings)

        if self.windows.get(title):
            self.windows[title].destroy()

        self.windows[title] = window
        return window

    def get_window(self, title):
        window = self.windows[title]
        if not window:
            raise Exception("Requested window does not exist!")
        return window

    def loop(self):
        glfw.poll_events()

    def destroy(self):
        for _, window in self.windows.items():
            if window.active():
                window.destroy()
        glfw.terminate()
