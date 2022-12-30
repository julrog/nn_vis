import abc
from typing import Any, Callable, List, Tuple

from opengl_helper.shader import BaseShader
from opengl_helper.vertex_data_handler import (LayeredVertexDataHandler,
                                               OverflowingVertexDataHandler,
                                               VertexDataHandler)
from rendering.rendering_config import RenderingConfig


class BaseRenderSet:
    def __init__(self, shader: BaseShader, render_func: Callable, element_count_func: Callable):
        __metaclass__ = abc.ABCMeta  # noqa F841
        self.shader: BaseShader = shader
        self.uniform_settings: List[str] = []
        self.render_func: Callable = render_func
        self.element_count_func: Callable = element_count_func

    def set_uniform_label(self, data: List[str]):
        if self.shader is not None:
            self.shader.set_uniform_label(data)

    def set_uniform_data(self, data: List[Tuple[str, Any, Any]]):
        if self.shader is not None:
            self.shader.set_uniform_data(data)

    def set_uniform_labeled_data(self, config: RenderingConfig):
        if self.shader is not None:
            self.shader.set_uniform_labeled_data(config)

    @abc.abstractmethod
    def render(self):
        pass


class RenderSet(BaseRenderSet):
    def __init__(self, shader: BaseShader, data_handler: VertexDataHandler, render_func: Callable,
                 element_count_func: Callable):
        super().__init__(shader, render_func, element_count_func)
        self.data_handler: VertexDataHandler = data_handler

    def render(self):
        if self.shader is not None:
            self.shader.use()
            self.data_handler.set(True)
            self.render_func(self.element_count_func())


class LayeredRenderSet(BaseRenderSet):
    def __init__(self, shader: BaseShader, data_handler: LayeredVertexDataHandler, render_func: Callable,
                 element_count_func: Callable):
        super().__init__(shader, render_func, element_count_func)
        self.data_handler: LayeredVertexDataHandler = data_handler
        self.buffer_divisor: List[Tuple[int, int]] = []

    def set_buffer_divisor(self, buffer_divisor: List[Tuple[int, int]]):
        self.buffer_divisor: List[Tuple[int, int]] = buffer_divisor

    def render(self):
        if self.shader is not None:
            self.shader.use()
            for buffer in iter(self.data_handler):
                buffer.set(True)
                buffer.buffer_divisor = self.buffer_divisor
                self.render_func(
                    self.element_count_func(self.data_handler.current_layer_id,
                                            self.data_handler.current_sub_buffer_id))


class OverflowingRenderSet(BaseRenderSet):
    def __init__(self, shader: BaseShader, data_handler: OverflowingVertexDataHandler, render_func: Callable,
                 element_count_func: Callable):
        super().__init__(shader, render_func, element_count_func)
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
                self.render_func(self.element_count_func(i))
