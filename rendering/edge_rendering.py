from typing import List, Callable

from OpenGL.GL import *

from models.grid import Grid
from opengl_helper.data_set import LayeredRenderSet, BaseRenderSet
from opengl_helper.render_utility import generate_render_function, OGLRenderFunction
from opengl_helper.shader import ShaderSetting
from opengl_helper.vertex_data_handler import LayeredVertexDataHandler, VertexDataHandler
from processing.edge_processing import EdgeProcessor
from rendering.renderer import Renderer
from rendering.rendering_config import RenderingConfig
from utility.camera import Camera
from utility.performance import track_time


class EdgeRenderer(Renderer):
    def __init__(self, edge_processor: EdgeProcessor, grid: Grid):
        Renderer.__init__(self)
        self.edge_processor = edge_processor
        self.grid = grid

        shader_settings: List[ShaderSetting] = []
        shader_settings.extend(
            [ShaderSetting("sample_point", ["sample/sample.vert", "basic/discard_screen_color.frag"],
                           ["edge_importance_threshold", "screen_width", "screen_height"]),
             ShaderSetting("sample_line",
                           ["sample/sample_impostor.vert", "basic/color.frag",
                            "sample/points_to_line.geom"],
                           ["edge_importance_threshold"]),
             ShaderSetting("sample_sphere", ["sample/sample_impostor.vert",
                                             "sample/point_to_sphere_impostor_phong.frag",
                                             "sample/point_to_sphere_impostor.geom"],
                           ["edge_object_radius", "edge_importance_threshold"]),
             ShaderSetting("sample_transparent_sphere", ["sample/sample_impostor.vert",
                                                         "sample/point_to_sphere_impostor_transparent.frag",
                                                         "sample/point_to_sphere_impostor.geom"],
                           ["edge_object_radius", "edge_base_opacity", "edge_importance_opacity", "edge_depth_opacity",
                            "edge_opacity_exponent", "edge_importance_threshold"]),
             ShaderSetting("sample_ellipsoid_transparent", ["sample/sample_impostor.vert",
                                                            "sample/points_to_ellipsoid_impostor_transparent.frag",
                                                            "sample/points_to_ellipsoid_impostor.geom"],
                           ["edge_object_radius", "edge_base_opacity", "edge_importance_opacity", "edge_depth_opacity",
                            "edge_opacity_exponent", "edge_importance_threshold"])
             ])
        self.set_shader(shader_settings)

        self.data_handler: LayeredVertexDataHandler = LayeredVertexDataHandler([[VertexDataHandler(
            [(self.edge_processor.sample_buffer[i][j], 0), (self.edge_processor.edge_buffer[i][j], 2)], []) for j in
            range(len(self.edge_processor.sample_buffer[i]))] for i in range(len(self.edge_processor.sample_buffer))])

        def generate_element_count_func(ep: EdgeProcessor) -> Callable:
            buffered_ep: EdgeProcessor = ep

            def element_count_func(layer: int, buffer: int):
                return buffered_ep.get_buffer_points(layer, buffer)

            return element_count_func

        self.render_funcs["sample_point"] = generate_render_function(OGLRenderFunction.ARRAYS_INSTANCED, GL_POINTS,
                                                                     10.0, depth_test=True)
        self.render_funcs["sample_line"] = generate_render_function(OGLRenderFunction.ARRAYS_INSTANCED, GL_POINTS,
                                                                    line_width=2.0, depth_test=True)
        self.render_funcs["sample_sphere"] = generate_render_function(OGLRenderFunction.ARRAYS_INSTANCED, GL_POINTS,
                                                                      depth_test=True)
        self.render_funcs["sample_transparent_sphere"] = generate_render_function(OGLRenderFunction.ARRAYS_INSTANCED,
                                                                                  GL_POINTS, add_blending=True)
        self.render_funcs["sample_ellipsoid_transparent"] = generate_render_function(OGLRenderFunction.ARRAYS_INSTANCED,
                                                                                     GL_POINTS, add_blending=True)
        self.element_count_funcs["sample_point"] = generate_element_count_func(edge_processor)
        self.element_count_funcs["sample_line"] = generate_element_count_func(edge_processor)
        self.element_count_funcs["sample_sphere"] = generate_element_count_func(edge_processor)
        self.element_count_funcs["sample_transparent_sphere"] = generate_element_count_func(edge_processor)
        self.element_count_funcs["sample_ellipsoid_transparent"] = generate_element_count_func(edge_processor)

        self.create_sets(self.data_handler)

        self.importance_threshold: float = 0.0

    @track_time
    def render(self, set_name: str, cam: Camera, config: RenderingConfig = None, show_class: int = 0):
        current_set: BaseRenderSet = self.sets[set_name]
        if isinstance(current_set, LayeredRenderSet):
            current_set.set_buffer_divisor([(0, 1), (1, self.edge_processor.max_sample_points)])
        near: float = 0.0
        far: float = 0.0
        if set_name is "sample_ellipsoid_transparent" or set_name is "sample_transparent_sphere":
            near, far = self.grid.get_near_far_from_view(cam.view)
        current_set.set_uniform_data([("projection", cam.projection, "mat4"),
                                      ("view", cam.view, "mat4"),
                                      ("farthest_point_view_z", far, "float"),
                                      ("nearest_point_view_z", near, "float"),
                                      ("object_radius", self.edge_processor.sample_length * 0.5, "float"),
                                      ("importance_threshold", self.importance_threshold, "float"),
                                      ("importance_max", self.edge_processor.edge_max_importance, "float"),
                                      ('max_sample_points', self.edge_processor.max_sample_points, "int"),
                                      ('show_class', show_class, "int"),
                                      ('edge_importance_type', 0, "int")])
        current_set.set_uniform_labeled_data(config)
        current_set.render()

    def delete(self):
        self.data_handler.delete()
