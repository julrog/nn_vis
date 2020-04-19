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
        self.SSBO_size: List[int] = []
        if ssbos == 1:
            self.SSBOs.append(glGenBuffers(ssbos))
            self.SSBO_size.append(0)
        else:
            if ssbos > 0:
                self.SSBOs.extend(glGenBuffers(ssbos))
                self.SSBO_size.extend([0] * len(self.SSBOs))
        self.max_ssbo_size = glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE)

    def load_ssbo_data(self, data: any, location: int = 0, buffer_id: int = 0):
        if data.nbytes > self.max_ssbo_size:
            raise Exception("Data to big for SSBO (%d bytes, max %d bytes)." % (data.nbytes, self.max_ssbo_size))

        self.set()

        self.SSBO_size[buffer_id] = data.nbytes
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.SSBOs[buffer_id])
        glBufferData(GL_SHADER_STORAGE_BUFFER, data.nbytes, data, GL_STATIC_DRAW)

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, location, self.SSBOs[buffer_id])

    def read_ssbo_data(self, buffer_id: int = 0) -> any:
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.SSBOs[buffer_id])
        return glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, self.SSBO_size[buffer_id])

    def bind_ssbo_data(self, location: int = 0, buffer_id: int = 0):
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, location, self.SSBOs[buffer_id])

    def load_vbo_data(self, data: any, buffer_id: int = 0):
        self.set()

        glBindBuffer(GL_ARRAY_BUFFER, self.VBOs[buffer_id])
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
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
