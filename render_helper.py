from typing import List, Tuple
from OpenGL.GL import *

from shader import BaseShader


class VertexDataHandler:
    def __init__(self, vbos: int = 0, ssbos: int = 0):
        self.VAO: int = glGenVertexArrays(1)
        self.VBOs: List[int] = []
        if vbos == 1:
            self.VBOs.append(glGenBuffers(vbos))
        else:
            if vbos > 0:
                self.VBOs.extend(glGenBuffers(vbos))

        self.SSBOs: List[int] = []
        if ssbos == 1:
            self.SSBOs.append(glGenBuffers(ssbos))
        else:
            if ssbos > 0:
                self.SSBOs.extend(glGenBuffers(ssbos))

    def load_ssbo_data(self, data, location: int = 0, buffer_id: int = 0):
        self.set()

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer_id)
        glBufferData(GL_SHADER_STORAGE_BUFFER, data.nbytes, data, GL_STATIC_DRAW)

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, location, buffer_id)

    def load_vbo_data(self, data, buffer_id: int = 0):
        self.set()

        glBindBuffer(GL_ARRAY_BUFFER, self.VBOs[buffer_id])
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)

        glEnableVertexAttribArray(buffer_id)
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, data.itemsize * 4, ctypes.c_void_p(0))

    def set(self):
        glBindVertexArray(self.VAO)


class RenderSet:
    def __init__(self, shader: BaseShader, data_handler: VertexDataHandler):
        self.shader: BaseShader = shader
        self.data_handler: VertexDataHandler = data_handler

    def set_uniform_data(self, data: List[Tuple[str, any, any]]):
        self.shader.set_uniform_data(data)

    def load_vbo_data(self, data: any, buffer_id: int = 0):
        self.data_handler.load_vbo_data(data, buffer_id)

    def load_ssbo_data(self, data: any, location: int = 0, buffer_id: int = 0):
        self.data_handler.load_ssbo_data(data, location, buffer_id)

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
