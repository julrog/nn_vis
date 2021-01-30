import abc
from typing import List, Tuple
from OpenGL.GL import *

from opengl_helper.buffer import BufferObject, OverflowingBufferObject


class BaseDataHandler:
    def __init__(self):
        __metaclass__ = abc.ABCMeta
        pass

    @abc.abstractmethod
    def set(self, rendering: bool = False):
        pass

    @abc.abstractmethod
    def delete(self):
        pass


class VertexDataHandler(BaseDataHandler):
    def __init__(self, targeted_buffer_objects: List[Tuple[BufferObject, int]],
                 buffer_divisor: List[Tuple[int, int]] = None):
        super().__init__()
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


class OverflowingVertexDataHandler(VertexDataHandler):
    def __init__(self, targeted_buffer_objects: List[Tuple[BufferObject, int]],
                 targeted_overflowing_buffer_objects: List[Tuple[OverflowingBufferObject, int]],
                 buffer_divisor: List[Tuple[int, int]] = None):
        super().__init__(targeted_buffer_objects, buffer_divisor)
        self.targeted_overflowing_buffer_objects: List[
            Tuple[OverflowingBufferObject, int]] = targeted_overflowing_buffer_objects
        self.current_buffer_id: int = 0

    def set_buffer(self, buffer_id: int):
        self.current_buffer_id = buffer_id

    def set(self, rendering: bool = False):
        VertexDataHandler.set(self, rendering)
        for buffer, location in self.targeted_overflowing_buffer_objects:
            buffer.bind_single(self.current_buffer_id, location, rendering)

    def set_range(self, count: int):
        glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT)
        glBindVertexArray(self.handle)
        for buffer, location in self.targeted_buffer_objects:
            buffer.bind(location)
        for buffer, location in self.targeted_overflowing_buffer_objects:
            for i in range(count):
                if self.current_buffer_id + i >= 0 and (self.current_buffer_id + i) < len(buffer.handle):
                    buffer.bind_single((self.current_buffer_id + i) % len(buffer.handle), location + i)

    def set_consecutive(self):
        glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT)
        glBindVertexArray(self.handle)
        for buffer, location in self.targeted_buffer_objects:
            buffer.bind(location)
        for buffer, location in self.targeted_overflowing_buffer_objects:
            buffer.bind_consecutive(location)


class LayeredVertexDataHandler(BaseDataHandler):
    def __init__(self, layered_data_handler: List[List[VertexDataHandler]]):
        super().__init__()
        if len(layered_data_handler) <= 0 or len(layered_data_handler[0]) <= 0:
            raise Exception("No data handler defined!")
        self.layered_data_handler: List[List[VertexDataHandler]] = layered_data_handler
        self.current_layer_id: int = 0
        self.current_sub_buffer_id: int = 0

    def set(self, rendering: bool = False):
        self.layered_data_handler[self.current_layer_id][self.current_sub_buffer_id].set(rendering)

    def delete(self):
        for layer in self.layered_data_handler:
            for buffer in layer:
                buffer.delete()

    def __iter__(self) -> BaseDataHandler:
        self.current_layer_id = 0
        self.current_sub_buffer_id = -1
        return self

    def __next__(self) -> VertexDataHandler:
        self.current_sub_buffer_id += 1
        if self.current_sub_buffer_id >= len(self.layered_data_handler[self.current_layer_id]):
            if self.current_layer_id + 1 < len(self.layered_data_handler):
                self.current_sub_buffer_id = 0
                self.current_layer_id = self.current_layer_id + 1
            else:
                self.current_layer_id = 0
                self.current_sub_buffer_id = -1
                raise StopIteration

        return self.layered_data_handler[self.current_layer_id][self.current_sub_buffer_id]
