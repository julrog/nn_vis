import abc
from typing import List, Tuple, Dict, Callable
from OpenGL.GL import *
from opengl_helper.buffer import BufferObject, OverflowingBufferObject
from opengl_helper.shader import BaseShader
from rendering.rendering_config import RenderingConfig

LOG_SOURCE = "RENDER_UTILITY"


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
            raise Exception("[%s] No data handler defined" % LOG_SOURCE)
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


class BaseRenderSet:
    def __init__(self, shader: BaseShader, render_func: Callable, point_count_func: Callable):
        __metaclass__ = abc.ABCMeta
        self.shader: BaseShader = shader
        self.uniform_settings: List[str] = []
        self.render_func: Callable = render_func
        self.point_count_func: Callable = point_count_func

    def set_uniform_label(self, data: List[str]):
        for setting in data:
            self.uniform_settings.append(setting)

    def set_uniform_data(self, data: List[Tuple[str, any, any]]):
        if self.shader is not None:
            self.shader.set_uniform_data(data)

    def set_uniform_labeled_data(self, config: RenderingConfig):
        if self.shader is not None and config is not None:
            uniform_data = []
            for setting, shader_name in config.shader_name.items():
                if setting in self.uniform_settings:
                    uniform_data.append((shader_name, config[setting], "float"))
            self.shader.set_uniform_data(uniform_data)

    @abc.abstractmethod
    def render(self):
        pass


class RenderSet(BaseRenderSet):
    def __init__(self, shader: BaseShader, data_handler: VertexDataHandler, render_func: Callable,
                 point_count_func: Callable):
        super().__init__(shader, render_func, point_count_func)
        self.data_handler: VertexDataHandler = data_handler

    def render(self):
        if self.shader is not None:
            self.shader.use()
            self.data_handler.set(True)
            self.render_func(self.point_count_func())


class LayeredRenderSet(BaseRenderSet):
    def __init__(self, shader: BaseShader, data_handler: LayeredVertexDataHandler, render_func: Callable,
                 point_count_func: Callable):
        super().__init__(shader, render_func, point_count_func)
        self.data_handler: LayeredVertexDataHandler = data_handler
        self.buffer_divisor: List[Tuple[int, int]] = []

    def render(self):
        if self.shader is not None:
            self.shader.use()
            for buffer in iter(self.data_handler):
                buffer.set(True)
                self.render_func(
                    self.point_count_func(self.data_handler.current_layer_id, self.data_handler.current_sub_buffer_id))


class OverflowingRenderSet(BaseRenderSet):
    def __init__(self, shader: BaseShader, data_handler: OverflowingVertexDataHandler, render_func: Callable,
                 point_count_func: Callable):
        super().__init__(shader, render_func, point_count_func)
        self.data_handler: OverflowingVertexDataHandler = data_handler

    def render_sub(self, buffer_index: int = 0):
        if self.shader is not None:
            self.shader.use()
            self.data_handler.set_buffer(buffer_index)
            self.data_handler.set(True)

    def render(self):
        if self.shader is not None:
            self.shader.use()
            for i in range(len(self.data_handler.targeted_overflowing_buffer_objects[0][0].handle)):
                self.data_handler.set_buffer(i)
                self.data_handler.set(True)
                self.render_func(self.point_count_func(i))


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
