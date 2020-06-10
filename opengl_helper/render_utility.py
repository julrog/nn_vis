from typing import List, Tuple, Dict
from OpenGL.GL import *

from opengl_helper.buffer import BufferObject, OverflowingBufferObject
from opengl_helper.shader import BaseShader

LOG_SOURCE = "RENDER_UTILITY"


class VertexDataHandler:
    def __init__(self, targeted_buffer_objects: List[Tuple[BufferObject, int]],
                 buffer_divisor: List[Tuple[int, int]] = None):
        self.handle: int = glGenVertexArrays(1)
        self.targeted_buffer_objects: List[Tuple[BufferObject, int]] = targeted_buffer_objects
        if buffer_divisor is None:
            self.buffer_divisor: List[Tuple[int, int]] = []
        else:
            self.buffer_divisor: List[Tuple[int, int]] = buffer_divisor

    def set(self, rendering: bool = False):
        glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT)
        glBindVertexArray(self.handle)
        for i, (buffer, location) in enumerate(self.targeted_buffer_objects):
            found_divisor: bool = False
            for buffer_id, divisor in self.buffer_divisor:
                if buffer_id == i:
                    found_divisor = True
                    buffer.bind(location, rendering, divisor=divisor)
            if not found_divisor:
                if len(self.buffer_divisor) == 0:
                    buffer.bind(location, rendering)
                else:
                    buffer.bind(location, rendering, divisor=1)

    def delete(self):
        glDeleteVertexArrays(1, [self.handle])


class OverflowingVertexDataHandler:
    def __init__(self, targeted_buffer_objects: List[Tuple[BufferObject, int]],
                 targeted_overflowing_buffer_objects: List[Tuple[OverflowingBufferObject, int]],
                 buffer_divisor: List[Tuple[int, int]] = None):
        self.handle: int = glGenVertexArrays(1)
        self.targeted_buffer_objects: List[Tuple[BufferObject, int]] = targeted_buffer_objects
        self.targeted_overflowing_buffer_objects: List[
            Tuple[OverflowingBufferObject, int]] = targeted_overflowing_buffer_objects
        if buffer_divisor is None:
            self.buffer_divisor: List[Tuple[int, int]] = []
        else:
            self.buffer_divisor: List[Tuple[int, int]] = buffer_divisor

    def set(self, buffer_id: int, rendering: bool = False):
        glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT)
        glBindVertexArray(self.handle)
        for i, (buffer, location) in enumerate(self.targeted_buffer_objects):
            found_divisor: bool = False
            for buffer_id, divisor in self.buffer_divisor:
                if buffer_id == i:
                    found_divisor = True
                    buffer.bind(location, rendering, divisor=divisor)
            if not found_divisor:
                if len(self.buffer_divisor) == 0:
                    buffer.bind(location, rendering)
                else:
                    buffer.bind(location, rendering, divisor=1)
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
        self.uniform_label: Dict[str, str] = dict()

    def set_uniform_label(self, data: List[Tuple[str, str]]):
        for label, uniform_name in data:
            self.uniform_label[label] = uniform_name

    def set_uniform_data(self, data: List[Tuple[str, any, any]]):
        if self.shader is not None:
            self.shader.set_uniform_data(data)

    def set_uniform_labeled_data(self, data: Dict[str, float]):
        if self.shader is not None and data is not None:
            uniform_data = []
            for label, value in data.items():
                if label in self.uniform_label.keys():
                    uniform_data.append((self.uniform_label[label], value, "float"))
            self.shader.set_uniform_data(uniform_data)

    def set(self):
        if self.shader is not None:
            self.shader.use()
            self.data_handler.set(True)


class RenderSetLayered:
    def __init__(self, shader: BaseShader, data_handler: List[List[VertexDataHandler]]):
        self.shader: BaseShader = shader
        self.data_handler: List[List[VertexDataHandler]] = data_handler
        self.uniform_label: Dict[str, str] = dict()
        self.buffer_divisor: List[Tuple[int, int]] = []

    def set_uniform_label(self, data: List[Tuple[str, str]]):
        for label, uniform_name in data:
            self.uniform_label[label] = uniform_name

    def set_uniform_data(self, data: List[Tuple[str, any, any]]):
        if self.shader is not None:
            self.shader.set_uniform_data(data)

    def set_uniform_labeled_data(self, data: Dict[str, float]):
        if self.shader is not None and data is not None:
            uniform_data = []
            for label, value in data.items():
                if label in self.uniform_label.keys():
                    uniform_data.append((self.uniform_label[label], value, "float"))
            self.shader.set_uniform_data(uniform_data)

    def render(self, render_function, point_count_function):
        if self.shader is not None:
            self.shader.use()
            for i in range(len(self.data_handler)):
                for j in range(len(self.data_handler[i])):
                    self.data_handler[i][j].buffer_divisor = self.buffer_divisor
                    self.data_handler[i][j].set(True)
                    render_function(point_count_function(i, j))


class OverflowingRenderSet:
    def __init__(self, shader: BaseShader, data_handler: OverflowingVertexDataHandler):
        self.shader: BaseShader = shader
        self.data_handler: OverflowingVertexDataHandler = data_handler
        self.uniform_label: Dict[str, str] = dict()

    def set_uniform_label(self, data: List[Tuple[str, str]]):
        for label, uniform_name in data:
            self.uniform_label[label] = uniform_name

    def set_uniform_data(self, data: List[Tuple[str, any, any]]):
        self.shader.set_uniform_data(data)

    def set_uniform_labeled_data(self, data: Dict[str, float]):
        if self.shader is not None and data is not None:
            uniform_data = []
            for label, value in data.items():
                if label in self.uniform_label.keys():
                    uniform_data.append((self.uniform_label[label], value, "float"))
            self.shader.set_uniform_data(uniform_data)

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
