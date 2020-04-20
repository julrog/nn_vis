from typing import List, Tuple
from OpenGL.GL import *

from shader import BaseShader


class BufferObject:
    def __init__(self, ssbo: bool = False):
        self.handle: int = glGenBuffers(1)
        self.location: int = 0
        self.ssbo: bool = ssbo
        if self.ssbo:
            self.size: int = 0
            self.max_ssbo_size: int = glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE)

    def load(self, data):
        glBindVertexArray(0)

        self.size = data.nbytes
        if self.ssbo:
            if data.nbytes > self.max_ssbo_size:
                raise Exception("Data to big for SSBO (%d bytes, max %d bytes)." % (data.nbytes, self.max_ssbo_size))

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.handle)
            glBufferData(GL_SHADER_STORAGE_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        else:
            glBindBuffer(GL_ARRAY_BUFFER, self.handle)
            glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)

    def read(self) -> any:
        if self.ssbo:
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.handle)
            return glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, self.size)

    def bind(self, location: int, rendering: bool = False):
        if self.ssbo:
            if rendering:
                glBindBuffer(GL_ARRAY_BUFFER, self.handle)
                glEnableVertexAttribArray(location)
                glVertexAttribPointer(location, 4, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
            else:
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, location, self.handle)
        else:
            glBindBuffer(GL_ARRAY_BUFFER, self.handle)
            glEnableVertexAttribArray(location)
            glVertexAttribPointer(location, 4, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))


class SwappingBufferObject(BufferObject):
    def __init__(self, ssbo: bool = False):
        super().__init__(ssbo)
        self.swap_handle: int = glGenBuffers(1)

    def swap(self):
        self.handle, self.swap_handle = self.swap_handle, self.handle

    def bind(self, location: int, rendering: bool = False):
        if self.ssbo:
            if rendering:
                glBindBuffer(GL_ARRAY_BUFFER, self.handle)
                glEnableVertexAttribArray(location)
                glVertexAttribPointer(location, 4, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
            else:
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, location, self.handle)
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, location + 1, self.swap_handle)
        else:
            glBindBuffer(GL_ARRAY_BUFFER, self.handle)
            glEnableVertexAttribArray(location)
            glVertexAttribPointer(location, 4, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))


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
