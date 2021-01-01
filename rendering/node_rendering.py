from typing import List, Tuple, Callable

from OpenGL.GL import *
from models.grid import Grid
from opengl_helper.render_utility import VertexDataHandler, RenderSet, render_setting_0, render_setting_1, BaseRenderSet
from opengl_helper.shader import RenderShaderHandler, RenderShader, ShaderSetting
from processing.node_processing import NodeProcessor
from rendering.rendering import Renderer
from rendering.rendering_config import RenderingConfig
from utility.camera import Camera
from utility.performance import track_time


class NodeRenderer(Renderer):
    def __init__(self, node_processor: NodeProcessor, grid: Grid):
        Renderer.__init__(self)
        self.node_processor = node_processor
        self.grid = grid

        shader_settings: List[ShaderSetting] = []
        shader_settings.extend([ShaderSetting("node_point", ["node/sample.vert", "basic/discard_screen_color.frag"]),
                                ShaderSetting("node_sphere",
                                              ["node/sample_impostor.vert", "node/point_to_sphere_impostor_phong.fragg",
                                               "node/point_to_sphere_impostor.geom"],
                                              ["node_object_radius", "node_importance_threshold"]),
                                ShaderSetting("node_transparent_sphere", ["node/sample_impostor.vert",
                                                                          "node/point_to_sphere_impostor_transparent.frag",
                                                                          "node/point_to_sphere_impostor.geom"],
                                              ["node_object_radius", "node_base_opacity", "node_importance_opacity",
                                               "node_depth_opacity",
                                               "node_opacity_exponent", "node_importance_threshold"])
                                ])
        self.set_shader(shader_settings)

        self.data_handler: VertexDataHandler = VertexDataHandler([(self.node_processor.node_buffer, 0)])

        def point_render_func(elements: int):
            render_setting_0(False)
            glPointSize(10.0)
            glDrawArrays(GL_POINTS, 0, elements)
            glMemoryBarrier(GL_ALL_BARRIER_BITS)

        def point_render_trans_func(elements: int):
            render_setting_1(False)
            glPointSize(10.0)
            glDrawArrays(GL_POINTS, 0, elements)
            glMemoryBarrier(GL_ALL_BARRIER_BITS)

        def generate_element_count_func(np: NodeProcessor) -> Callable:
            buffered_np: NodeProcessor = np
            
            def element_count_func():
                return buffered_np.get_buffer_points()

            return element_count_func

        self.render_funcs["node_point"] = point_render_func
        self.render_funcs["node_sphere"] = point_render_func
        self.render_funcs["node_transparent_sphere"] = point_render_trans_func
        self.element_count_funcs["node_point"] = generate_element_count_func(node_processor)
        self.element_count_funcs["node_sphere"] = generate_element_count_func(node_processor)
        self.element_count_funcs["node_transparent_sphere"] = generate_element_count_func(node_processor)

        self.create_sets(self.data_handler)

    @track_time
    def render(self, set_name: str, cam: Camera, config: RenderingConfig = None, show_class: int = 0):
        current_set: BaseRenderSet = self.sets[set_name]
        current_set.set_uniform_data([("projection", cam.projection, "mat4"),
                                      ("view", cam.view, "mat4")])
        current_set.set_uniform_labeled_data(config)
        current_set.render()

    def delete(self):
        self.data_handler.delete()
