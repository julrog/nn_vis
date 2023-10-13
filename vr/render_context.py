from typing import List

import glfw
import numpy as np
from OpenGL.GL import glViewport

from vr.camera import VRCamera


class VROpenGLContext:
    def __init__(self, width: int, height: int) -> None:
        if not glfw.init():
            raise Exception('glfw can not be initialized!')
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

        self.width: int = width
        self.height: int = height

        self.handle = glfw.create_window(
            self.width, self.height, 'NNVis Render', None, None
        )
        if not self.handle:
            raise Exception('glfw window can not be created!')
        self.active: bool = False
        self.cam: List[VRCamera] = [
            VRCamera(self.width, self.height),
            VRCamera(self.width, self.height),
        ]

    def update_camera_matrices(
        self, id: int, projection: np.ndarray, eye_to_head: np.ndarray
    ) -> None:
        self.cam[id].update_projection(projection)
        self.cam[id].update_eye_to_head(eye_to_head)

    def activate(self) -> None:
        glfw.make_context_current(self.handle)
        glViewport(0, 0, self.width, self.height)
        self.active = True

    def is_active(self) -> bool:
        return not glfw.window_should_close(self.handle)

    def swap(self) -> None:
        glfw.swap_buffers(self.handle)

    def destroy(self) -> None:
        glfw.destroy_window(self.handle)
        glfw.terminate()

    def update(self) -> None:
        glfw.poll_events()
