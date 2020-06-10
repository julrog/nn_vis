from typing import Dict, List

from OpenGL.GL import *

from models.grid import Grid
from opengl_helper.render_utility import VertexDataHandler, RenderSet, render_setting_0, render_setting_1, \
    RenderSetLayered
from opengl_helper.shader import RenderShaderHandler, RenderShader
from processing.edge_processing import EdgeProcessor
from utility.performance import track_time
from utility.window import Window


class EdgeRenderer:
    def __init__(self, edge_processor: EdgeProcessor, grid: Grid):
        self.edge_processor = edge_processor
        self.grid = grid

        shader_handler: RenderShaderHandler = RenderShaderHandler()
        sample_point_shader: RenderShader = shader_handler.create("sample_point", "sample/sample.vert",
                                                                  "basic/discard_screen_color.frag")
        sample_line_shader: RenderShader = shader_handler.create("sample_line", "sample/sample_impostor.vert",
                                                                 "basic/color.frag",
                                                                 "sample/points_to_line.geom")
        sample_sphere_shader: RenderShader = shader_handler.create("sample_sphere", "sample/sample_impostor.vert",
                                                                   "sample/point_to_sphere_impostor_phong.frag",
                                                                   "sample/point_to_sphere_impostor.geom")
        sample_transparent_shader: RenderShader = shader_handler.create("sample_transparent_sphere",
                                                                        "sample/sample_impostor.vert",
                                                                        "sample/point_to_sphere_impostor_transparent.frag",
                                                                        "sample/point_to_sphere_impostor.geom")
        sample_ellipse_shader: RenderShader = shader_handler.create("sample_ellipsoid_transparent",
                                                                    "sample/sample_impostor.vert",
                                                                    "sample/points_to_ellipsoid_impostor_transparent.frag",
                                                                    "sample/points_to_ellipsoid_impostor.geom")

        self.data_handler: List[List[VertexDataHandler]] = [[VertexDataHandler(
            [(self.edge_processor.sample_buffer[i][j], 0), (self.edge_processor.edge_buffer[i][j], 2)], []) for j in
            range(len(self.edge_processor.sample_buffer[i]))] for i in range(len(self.edge_processor.sample_buffer))]

        self.point_render: RenderSetLayered = RenderSetLayered(sample_point_shader, self.data_handler)
        self.point_render.set_uniform_label([("Importance Threshold", "importance_threshold")])
        self.line_render: RenderSetLayered = RenderSetLayered(sample_line_shader, self.data_handler)
        self.line_render.set_uniform_label([("Importance Threshold", "importance_threshold")])
        self.sphere_render: RenderSetLayered = RenderSetLayered(sample_sphere_shader, self.data_handler)
        self.sphere_render.set_uniform_label(
            [("Size", "object_radius"), ("Importance Threshold", "importance_threshold")])
        self.transparent_render: RenderSetLayered = RenderSetLayered(sample_transparent_shader, self.data_handler)
        self.transparent_render.set_uniform_label(
            [("Size", "object_radius"), ("Base Opacity", "base_opacity"),
             ("Base Density Opacity", "base_shpere_opacity"),
             ("Density Exponent", "opacity_exponent"), ("Importance Threshold", "importance_threshold")])
        self.ellipse_render: RenderSetLayered = RenderSetLayered(sample_ellipse_shader, self.data_handler)
        self.ellipse_render.set_uniform_label(
            [("Size", "object_radius"), ("Base Opacity", "base_opacity"),
             ("Base Density Opacity", "base_shpere_opacity"),
             ("Density Exponent", "opacity_exponent"), ("Importance Threshold", "importance_threshold")])
        self.importance_threshold: float = 0.0

    @track_time
    def render_point(self, window: Window, clear: bool = True, swap: bool = False, options: Dict[str, float] = None):
        self.point_render.buffer_divisor = [(0, 1), (1, self.edge_processor.max_sample_points)]

        self.point_render.set_uniform_data([("projection", window.cam.projection, "mat4"),
                                            ("view", window.cam.view, "mat4"),
                                            ("screen_width", 1920.0, "float"),
                                            ("screen_height", 1080.0, "float"),
                                            ('max_sample_points', self.edge_processor.max_sample_points, 'int'),
                                            ("importance_threshold", self.importance_threshold, "float")])
        self.point_render.set_uniform_labeled_data(options)

        def render_function(sample_points: int):
            render_setting_0(clear)
            glPointSize(10.0)
            glDrawArraysInstanced(GL_POINTS, 0, 1, sample_points)

        self.point_render.render(render_function, self.edge_processor.get_buffer_points)

        if swap:
            window.swap()

    @track_time
    def render_line(self, window: Window, clear: bool = True, swap: bool = False, options: Dict[str, float] = None):
        self.line_render.buffer_divisor = [(0, 1), (1, self.edge_processor.max_sample_points)]

        self.line_render.set_uniform_data([("projection", window.cam.projection, "mat4"),
                                           ("view", window.cam.view, "mat4"),
                                           ('max_sample_points', self.edge_processor.max_sample_points, 'int'),
                                           ("importance_threshold", self.importance_threshold, "float")])
        self.line_render.set_uniform_labeled_data(options)

        def render_function(sample_points: int):
            render_setting_0(clear)
            glLineWidth(2.0)
            glDrawArraysInstanced(GL_POINTS, 0, 1, sample_points)

        self.line_render.render(render_function, self.edge_processor.get_buffer_points)

        if swap:
            window.swap()

    @track_time
    def render_sphere(self, window: Window, sphere_radius: float = 0.05, clear: bool = True, swap: bool = False,
                      options: Dict[str, float] = None):
        self.sphere_render.buffer_divisor = [(0, 1), (1, self.edge_processor.max_sample_points)]

        self.sphere_render.set_uniform_data([("projection", window.cam.projection, "mat4"),
                                             ("view", window.cam.view, "mat4"),
                                             ("object_radius", sphere_radius, "float"),
                                             ("importance_threshold", self.importance_threshold, "float"),
                                             ('max_sample_points', self.edge_processor.max_sample_points, 'int')])
        self.sphere_render.set_uniform_labeled_data(options)

        def render_function(sample_points: int):
            render_setting_0(clear)
            glDrawArraysInstanced(GL_POINTS, 0, 1, sample_points)

        self.sphere_render.render(render_function, self.edge_processor.get_buffer_points)

        if swap:
            window.swap()

    @track_time
    def render_transparent_sphere(self, window: Window, sphere_radius: float = 0.05, clear: bool = True,
                                  swap: bool = False, options: Dict[str, float] = None):
        self.transparent_render.buffer_divisor = [(0, 1), (1, self.edge_processor.max_sample_points)]

        near, far = self.grid.get_near_far_from_view(window.cam.view)
        self.transparent_render.set_uniform_data([("projection", window.cam.projection, "mat4"),
                                                  ("view", window.cam.view, "mat4"),
                                                  ("farthest_point_view_z", far, "float"),
                                                  ("nearest_point_view_z", near, "float"),
                                                  ("object_radius", sphere_radius, "float"),
                                                  ("importance_threshold", self.importance_threshold, "float"),
                                                  ('max_sample_points', self.edge_processor.max_sample_points, 'int')])
        self.transparent_render.set_uniform_labeled_data(options)

        def render_function(sample_points: int):
            render_setting_1(clear)
            glDrawArraysInstanced(GL_POINTS, 0, 1, sample_points)

        self.transparent_render.render(render_function, self.edge_processor.get_buffer_points)

        if swap:
            window.swap()

    @track_time
    def render_ellipsoid_transparent(self, window: Window, clear: bool = True, swap: bool = False,
                                     options: Dict[str, float] = None):
        self.ellipse_render.buffer_divisor = [(0, 1), (1, self.edge_processor.max_sample_points)]

        near, far = self.grid.get_near_far_from_view(window.cam.view)
        self.ellipse_render.set_uniform_data([("projection", window.cam.projection, "mat4"),
                                              ("view", window.cam.view, "mat4"),
                                              ("farthest_point_view_z", far, "float"),
                                              ("nearest_point_view_z", near, "float"),
                                              ("object_radius", self.edge_processor.sample_length * 0.5, "float"),
                                              ("importance_threshold", self.importance_threshold, "float"),
                                              ('max_sample_points', self.edge_processor.max_sample_points, 'int')])
        self.ellipse_render.set_uniform_labeled_data(options)

        def render_function(sample_points: int):
            render_setting_1(clear)
            glDrawArraysInstanced(GL_POINTS, 0, 1, sample_points)

        self.ellipse_render.render(render_function, self.edge_processor.get_buffer_points)

        if swap:
            window.swap()

    def delete(self):
        for layer_data_handler in self.data_handler:
            for container_data_handler in layer_data_handler:
                container_data_handler.delete()
        self.data_handler = []
