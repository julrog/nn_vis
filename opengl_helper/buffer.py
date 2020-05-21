import math
from typing import List

import numpy as np
from OpenGL.GL import *

LOG_SOURCE: str = "BUFFER"


class BufferObject:
    def __init__(self, ssbo: bool = False, object_size: int = 4, render_data_offset: List[int] = None,
                 render_data_size: List[int] = None):
        self.handle: int = glGenBuffers(1)
        self.location: int = 0
        self.ssbo: bool = ssbo
        if self.ssbo:
            self.size: int = 0
            self.max_ssbo_size: int = glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE)
        self.object_size = object_size
        self.render_data_offset = render_data_offset
        if render_data_offset is None:
            self.render_data_offset = [0]
        self.render_data_size = render_data_size
        if render_data_size is None:
            self.render_data_size = [4]

    def load(self, data: any):
        glBindVertexArray(0)

        self.size = data.nbytes
        if self.ssbo:
            if data.nbytes > self.max_ssbo_size:
                raise Exception("[%s] Data to big for SSBO (%d bytes, max %d bytes)." % (
                    LOG_SOURCE, data.nbytes, self.max_ssbo_size))

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
                for i in range(len(self.render_data_offset)):
                    glEnableVertexAttribArray(location + i)
                    glVertexAttribPointer(location + i, self.render_data_size[i], GL_FLOAT, GL_FALSE,
                                          self.object_size * 4, ctypes.c_void_p(4 * self.render_data_offset[i]))
            else:
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, location, self.handle)
        else:
            glBindBuffer(GL_ARRAY_BUFFER, self.handle)
            for i in range(len(self.render_data_offset)):
                glEnableVertexAttribArray(location + i)
                glVertexAttribPointer(location + i, self.render_data_size[i], GL_FLOAT, GL_FALSE,
                                      self.object_size * 4, ctypes.c_void_p(4 * self.render_data_offset[i]))

    def clear(self):
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.handle)
        glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_RGBA32F, GL_RGBA, GL_FLOAT, None)

    def delete(self):
        glDeleteBuffers(1, [self.handle])


class SwappingBufferObject(BufferObject):
    def __init__(self, ssbo: bool = False, object_size: int = 4, render_data_offset: List[int] = None,
                 render_data_size: List[int] = None):
        super().__init__(ssbo, object_size, render_data_offset, render_data_size)
        self.swap_handle: int = glGenBuffers(1)

    def swap(self):
        self.handle, self.swap_handle = self.swap_handle, self.handle

    def bind(self, location: int, rendering: bool = False):
        if self.ssbo:
            if rendering:
                glBindBuffer(GL_ARRAY_BUFFER, self.handle)
                for i in range(len(self.render_data_offset)):
                    glEnableVertexAttribArray(location + i)
                    glVertexAttribPointer(location + i, self.render_data_size[i], GL_FLOAT, GL_FALSE,
                                          self.object_size * 4, ctypes.c_void_p(4 * self.render_data_offset[i]))
            else:
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, location, self.handle)
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, location + 1, self.swap_handle)
        else:
            glBindBuffer(GL_ARRAY_BUFFER, self.handle)
            for i in range(len(self.render_data_offset)):
                glEnableVertexAttribArray(location + i)
                glVertexAttribPointer(location + i, self.render_data_size[i], GL_FLOAT, GL_FALSE,
                                      self.object_size * 4, ctypes.c_void_p(4 * self.render_data_offset[i]))

    def delete(self):
        glDeleteBuffers(1, [self.handle])
        glDeleteBuffers(1, [self.swap_handle])


class OverflowingBufferObject:
    def __init__(self, data_splitting_function):
        self.handle: List[int] = [glGenBuffers(1)]
        self.location: int = 0
        self.overall_size: int = 0
        self.size: List[int] = []
        self.max_ssbo_size: int = glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE)
        self.max_buffer_objects: int = glGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS)
        self.data_splitting_function = data_splitting_function

    def load(self, data: any):
        glBindVertexArray(0)

        self.overall_size = data.nbytes
        if data.nbytes > self.max_ssbo_size:
            buffer_count = math.ceil(data.nbytes / self.max_ssbo_size)
            for i in range(buffer_count):
                if i >= len(self.handle):
                    self.handle.append(glGenBuffers(1))
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.handle[i])
                split_data = self.data_splitting_function(data, i, self.max_ssbo_size)
                self.size.append(split_data.nbytes)
                glBufferData(GL_SHADER_STORAGE_BUFFER, split_data.nbytes, split_data, GL_STATIC_DRAW)
        else:
            self.size[0] = data.nbytes
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.handle[0])
            glBufferData(GL_SHADER_STORAGE_BUFFER, data.nbytes, data, GL_STATIC_DRAW)

    def load_empty(self, dtype, size: int, component_size: int):
        glBindVertexArray(0)

        self.overall_size = size * 16
        if size * 16 > self.max_ssbo_size:
            empty = np.zeros(int(self.max_ssbo_size / 4), dtype=dtype)
            buffer_count = math.ceil(
                int(size * 16 / (component_size * 16)) / int(self.max_ssbo_size / (component_size * 16)))
            print("component size %i, max components: %i, in buffer: %i, buffer: %i" % (
                component_size, int(size * 16 / (component_size * 16)), int(self.max_ssbo_size / (component_size * 16)),
                buffer_count))
            for i in range(buffer_count):
                if i >= len(self.handle):
                    self.handle.append(glGenBuffers(1))
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.handle[i])
                self.size.append(empty.nbytes)
                glBufferData(GL_SHADER_STORAGE_BUFFER, empty.nbytes, empty, GL_STATIC_DRAW)
        else:
            empty = np.zeros(size * 4, dtype=dtype)
            self.size.append(empty.nbytes)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.handle[0])
            glBufferData(GL_SHADER_STORAGE_BUFFER, empty.nbytes, empty, GL_STATIC_DRAW)

    def read(self) -> any:
        data = None
        for i, buffer in enumerate(self.handle):
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer)
            if data is None:
                data = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, self.size[i])
            else:
                data.extend(glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, self.size[i]))
        return data

    def bind_single(self, buffer_id: int, location: int, rendering: bool = False):
        if rendering:
            glBindBuffer(GL_ARRAY_BUFFER, self.handle[buffer_id])
            glEnableVertexAttribArray(location)
            glVertexAttribPointer(location, 4, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
        else:
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, location, self.handle[buffer_id])

    def bind_consecutive(self, location: int):
        for i, buffer in enumerate(self.handle):
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, location + i, len(self.handle), buffer)

    def clear(self):
        for buffer in self.handle:
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer)
            glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_RGBA32F, GL_RGBA, GL_FLOAT, None)

    def delete(self):
        for buffer in self.handle:
            glDeleteBuffers(1, [buffer])
        self.handle = []
