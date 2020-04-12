from typing import List, Tuple
from OpenGL.GL import *

from shader import BaseShader


class VertexDataHandler:
    def __init__(self):
        self.VAO = glGenVertexArrays(1)
        self.VBO = glGenBuffers(1)

    def load_data(self, data):
        self.set()

        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)

        # cube vertices
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, data.itemsize * 3, ctypes.c_void_p(0))

    def set(self):
        glBindVertexArray(self.VAO)


class RenderSet:
    def __init__(self, shader: BaseShader, data_handler: VertexDataHandler):
        self.shader: BaseShader = shader
        self.data_handler: VertexDataHandler = data_handler

    def set_uniform_data(self, data: List[Tuple[str, any, any]]):
        self.shader.set_uniform_data(data)

    def load_data(self, data: any):
        self.data_handler.load_data(data)

    def set(self):
        self.shader.use()
        self.data_handler.set()


def render_setting_0():
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_DEPTH_TEST)
    glDisable(GL_BLEND)


def render_setting_1():
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glBlendEquationSeparate(GL_MIN, GL_MAX)
