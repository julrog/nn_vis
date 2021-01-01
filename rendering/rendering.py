import abc
from typing import List, Dict, Callable

from opengl_helper.render_utility import BaseRenderSet, VertexDataHandler, OverflowingVertexDataHandler, \
    OverflowingRenderSet, LayeredVertexDataHandler, LayeredRenderSet, RenderSet, BaseDataHandler
from opengl_helper.shader import RenderShaderHandler, RenderShader, ShaderSetting
from rendering.rendering_config import RenderingConfig
from utility.camera import Camera

LOG_SOURCE = "RENDERING"


class Renderer:
    def __init__(self):
        __metaclass__ = abc.ABCMeta
        self.shaders: Dict[str, RenderShader] = dict()
        self.sets: Dict[str, BaseRenderSet] = dict()
        self.render_funcs: Dict[str, Callable] = dict()
        self.element_count_funcs: Dict[str, Callable] = dict()
        self.render_elements: int = 0

    def set_shader(self, shader_settings: List[ShaderSetting]):
        shader_handler = RenderShaderHandler()
        for shader_setting in shader_settings:
            self.shaders[shader_setting.id_name] = shader_handler.create(shader_setting)

    def create_sets(self, data_handler: BaseDataHandler):
        if isinstance(data_handler, OverflowingVertexDataHandler):
            for name, shader in self.shaders:
                self.sets[name] = OverflowingRenderSet(shader, data_handler, self.render_funcs[name],
                                                       self.element_count_funcs[name])
        elif isinstance(data_handler, LayeredVertexDataHandler):
            for name, shader in self.shaders:
                self.sets[name] = LayeredRenderSet(shader, data_handler, self.render_funcs[name],
                                                   self.element_count_funcs[name])
        elif isinstance(data_handler, VertexDataHandler):
            for name, shader in self.shaders:
                self.sets[name] = RenderSet(shader, data_handler, self.render_funcs[name],
                                            self.element_count_funcs[name])

    @abc.abstractmethod
    def render(self, set_name: str, cam: Camera, config: RenderingConfig = None, show_class: int = 0):
        return

    @abc.abstractmethod
    def delete(self):
        return
