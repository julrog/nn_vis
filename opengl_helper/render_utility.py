from typing import List, Tuple
from OpenGL.GL import *

from opengl_helper.buffer import BufferObject
from opengl_helper.shader import BaseShader


class VertexDataHandler:
    def __init__(self, targeted_buffer_objects: List[Tuple[BufferObject, int]]):
        self.VAO: int = glGenVertexArrays(1)
        self.targeted_buffer_objects: List[Tuple[BufferObject, int]] = targeted_buffer_objects

    def set(self, rendering: bool = False):
        glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT)
        glBindVertexArray(self.VAO)
        for buffer, location in self.targeted_buffer_objects:
            buffer.bind(location, rendering)


class RenderSet:
    def __init__(self, shader: BaseShader, data_handler: VertexDataHandler):
        self.shader: BaseShader = shader
        self.data_handler: VertexDataHandler = data_handler

    def set_uniform_data(self, data: List[Tuple[str, any, any]]):
        self.shader.set_uniform_data(data)

    def set(self):
        self.shader.use()
        self.data_handler.set(True)


def render_setting_0(clear: bool = True):
    if clear:
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_DEPTH_TEST)
    glDisable(GL_BLEND)


def render_setting_1(clear: bool = True):
    if clear:
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glBlendEquationSeparate(GL_MIN, GL_MAX)
