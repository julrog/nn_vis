from typing import List, Tuple
from OpenGL.GL import *

from opengl_helper.buffer import BufferObject, OverflowingBufferObject
from opengl_helper.shader import BaseShader


class VertexDataHandler:
    def __init__(self, targeted_buffer_objects: List[Tuple[BufferObject, int]]):
        self.handle: int = glGenVertexArrays(1)
        self.targeted_buffer_objects: List[Tuple[BufferObject, int]] = targeted_buffer_objects

    def set(self, rendering: bool = False):
        glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT)
        glBindVertexArray(self.handle)
        for buffer, location in self.targeted_buffer_objects:
            buffer.bind(location, rendering)

    def delete(self):
        glDeleteVertexArrays(1, [self.handle])


class OverflowingVertexDataHandler:
    def __init__(self, targeted_buffer_objects: List[Tuple[BufferObject, int]],
                 targeted_overflowing_buffer_objects: List[Tuple[OverflowingBufferObject, int]]):
        self.handle: int = glGenVertexArrays(1)
        self.targeted_buffer_objects: List[Tuple[BufferObject, int]] = targeted_buffer_objects
        self.targeted_overflowing_buffer_objects: List[
            Tuple[OverflowingBufferObject, int]] = targeted_overflowing_buffer_objects

    def set(self, buffer_id: int, rendering: bool = False):
        glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT)
        glBindVertexArray(self.handle)
        for buffer, location in self.targeted_buffer_objects:
            buffer.bind(location, rendering)
        for buffer, location in self.targeted_overflowing_buffer_objects:
            buffer.bind_single(buffer_id, location, rendering)

    def set_range(self, buffer_id: int, count: int):
        glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT)
        glBindVertexArray(self.handle)
        for buffer, location in self.targeted_buffer_objects:
            buffer.bind(location)
        for buffer, location in self.targeted_overflowing_buffer_objects:
            for i in range(count):
                if buffer_id + i >= 0 and (buffer_id + i) < len(buffer.handle):
                    buffer.bind_single((buffer_id + i) % len(buffer.handle), location + i)

    def set_consecutive(self):
        glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT)
        glBindVertexArray(self.handle)
        for buffer, location in self.targeted_buffer_objects:
            buffer.bind(location)
        for buffer, location in self.targeted_overflowing_buffer_objects:
            buffer.bind_consecutive(location)

    def delete(self):
        glDeleteVertexArrays(1, [self.handle])


class RenderSet:
    def __init__(self, shader: BaseShader, data_handler: VertexDataHandler):
        self.shader: BaseShader = shader
        self.data_handler: VertexDataHandler = data_handler

    def set_uniform_data(self, data: List[Tuple[str, any, any]]):
        if self.shader is not None:
            self.shader.set_uniform_data(data)

    def set(self):
        if self.shader is not None:
            self.shader.use()
            self.data_handler.set(True)


class OverflowingRenderSet:
    def __init__(self, shader: BaseShader, data_handler: OverflowingVertexDataHandler):
        self.shader: BaseShader = shader
        self.data_handler: OverflowingVertexDataHandler = data_handler

    def set_uniform_data(self, data: List[Tuple[str, any, any]]):
        self.shader.set_uniform_data(data)

    def set(self, buffer_index: int = 0):
        self.shader.use()
        self.data_handler.set(buffer_index, True)


def clear_screen(clear_color: List[float]):
    glClearColor(clear_color[0], clear_color[1], clear_color[2], clear_color[3])
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)


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
