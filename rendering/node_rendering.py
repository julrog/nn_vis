from typing import List, Callable

from OpenGL.GL import *
from models.grid import Grid
from opengl_helper.data_set import BaseRenderSet

from opengl_helper.render_utility import generate_render_function, OGLRenderFunction
from opengl_helper.shader import ShaderSetting
from opengl_helper.vertex_data_handler import VertexDataHandler
from processing.node_processing import NodeProcessor
from rendering.renderer import Renderer
from rendering.rendering_config import RenderingConfig
from utility.camera import Camera
from utility.performance import track_time


class NodeRenderer(Renderer):
    def __init__(self, node_processor: NodeProcessor, grid: Grid):
        Renderer.__init__(self)
        self.node_processor = node_processor
        self.grid = grid

        shader_settings: List[ShaderSetting] = []
        shader_settings.extend([ShaderSetting("node_point", ["node/sample.vert", "basic/discard_screen_color.frag"],
                                              ["screen_width", "screen_height"]),
                                ShaderSetting("node_sphere",
                                              ["node/sample_impostor.vert", "node/point_to_sphere_impostor_phong.frag",
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

        def generate_element_count_func(np: NodeProcessor) -> Callable:
            buffered_np: NodeProcessor = np

            def element_count_func():
                return buffered_np.get_buffer_points()

            return element_count_func

        self.render_funcs["node_point"] = generate_render_function(OGLRenderFunction.ARRAYS, GL_POINTS, 10.0,
                                                                   depth_test=True)
        self.render_funcs["node_sphere"] = generate_render_function(OGLRenderFunction.ARRAYS, GL_POINTS,
                                                                    depth_test=True)
        self.render_funcs["node_transparent_sphere"] = generate_render_function(OGLRenderFunction.ARRAYS, GL_POINTS,
                                                                                add_blending=True)
        self.element_count_funcs["node_point"] = generate_element_count_func(node_processor)
        self.element_count_funcs["node_sphere"] = generate_element_count_func(node_processor)
        self.element_count_funcs["node_transparent_sphere"] = generate_element_count_func(node_processor)

        self.create_sets(self.data_handler)

    @track_time
    def render(self, set_name: str, cam: Camera, config: RenderingConfig = None, show_class: int = 0):
        current_set: BaseRenderSet = self.sets[set_name]
        near: float = 0.0
        far: float = 0.0
        if set_name is "node_transparent_sphere":
            near, far = self.grid.get_near_far_from_view(cam.view)
        current_set.set_uniform_data([("projection", cam.projection, "mat4"),
                                      ("view", cam.view, "mat4"),
                                      ("farthest_point_view_z", far, "float"),
                                      ("nearest_point_view_z", near, "float"),
                                      ("show_class", show_class, "int")])
        current_set.set_uniform_labeled_data(config)
        current_set.render()

    def delete(self):
        self.data_handler.delete()
