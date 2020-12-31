from typing import List
from OpenGL.GL import *
from models.grid import Grid
from opengl_helper.render_utility import VertexDataHandler, render_setting_0, render_setting_1, \
    LayeredRenderSet
from opengl_helper.shader import RenderShaderHandler, RenderShader
from processing.edge_processing import EdgeProcessor
from rendering.rendering_config import RenderingConfig
from utility.camera import Camera
from utility.performance import track_time


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

        self.point_render: LayeredRenderSet = LayeredRenderSet(sample_point_shader, self.data_handler)
        self.point_render.set_uniform_label(["edge_importance_threshold"])
        self.line_render: LayeredRenderSet = LayeredRenderSet(sample_line_shader, self.data_handler)
        self.line_render.set_uniform_label(["edge_importance_threshold"])
        self.sphere_render: LayeredRenderSet = LayeredRenderSet(sample_sphere_shader, self.data_handler)
        self.sphere_render.set_uniform_label(["edge_object_radius", "edge_importance_threshold"])
        self.transparent_render: LayeredRenderSet = LayeredRenderSet(sample_transparent_shader, self.data_handler)
        self.transparent_render.set_uniform_label(
            ["edge_object_radius", "edge_base_opacity", "edge_importance_opacity", "edge_depth_opacity",
             "edge_opacity_exponent", "edge_importance_threshold"])
        self.ellipse_render: LayeredRenderSet = LayeredRenderSet(sample_ellipse_shader, self.data_handler)
        self.ellipse_render.set_uniform_label(
            ["edge_object_radius", "edge_base_opacity", "edge_importance_opacity", "edge_depth_opacity",
             "edge_opacity_exponent", "edge_importance_threshold"])
        self.importance_threshold: float = 0.0

    @track_time
    def render_point(self, cam: Camera, config: RenderingConfig = None, show_class: int = 0):
        self.point_render.buffer_divisor = [(0, 1), (1, self.edge_processor.max_sample_points)]

        self.point_render.set_uniform_data([("projection", cam.projection, "mat4"),
                                            ("view", cam.view, "mat4"),
                                            ("screen_width", 1920.0, "float"),
                                            ("screen_height", 1080.0, "float"),
                                            ('max_sample_points', self.edge_processor.max_sample_points, 'int'),
                                            ("importance_threshold", self.importance_threshold, "float"),
                                            ("importance_max", self.edge_processor.edge_max_importance, "float"),
                                            ('show_class', show_class, 'int'),
                                            ('edge_importance_type', 0, 'int')])
        self.point_render.set_uniform_labeled_data(config)

        def render_function(sample_points: int):
            render_setting_0(False)
            glPointSize(10.0)
            glDrawArraysInstanced(GL_POINTS, 0, 1, sample_points)
            glMemoryBarrier(GL_ALL_BARRIER_BITS)

        self.point_render.render(render_function, self.edge_processor.get_buffer_points)

    @track_time
    def render_line(self, cam: Camera, config: RenderingConfig = None,
                    show_class: int = 0):
        self.line_render.buffer_divisor = [(0, 1), (1, self.edge_processor.max_sample_points)]

        self.line_render.set_uniform_data([("projection", cam.projection, "mat4"),
                                           ("view", cam.view, "mat4"),
                                           ('max_sample_points', self.edge_processor.max_sample_points, 'int'),
                                           ("importance_threshold", self.importance_threshold, "float"),
                                           ("importance_max", self.edge_processor.edge_max_importance, "float"),
                                           ('show_class', show_class, 'int'),
                                           ('edge_importance_type', 0, 'int')])
        self.line_render.set_uniform_labeled_data(config)

        def render_function(sample_points: int):
            render_setting_0(False)
            glLineWidth(2.0)
            glDrawArraysInstanced(GL_POINTS, 0, 1, sample_points)
            glMemoryBarrier(GL_ALL_BARRIER_BITS)

        self.line_render.render(render_function, self.edge_processor.get_buffer_points)

    @track_time
    def render_sphere(self, cam: Camera, sphere_radius: float = 0.05, config: RenderingConfig = None,
                      show_class: int = 0):
        self.sphere_render.buffer_divisor = [(0, 1), (1, self.edge_processor.max_sample_points)]

        self.sphere_render.set_uniform_data([("projection", cam.projection, "mat4"),
                                             ("view", cam.view, "mat4"),
                                             ("object_radius", sphere_radius, "float"),
                                             ("importance_threshold", self.importance_threshold, "float"),
                                             ("importance_max", self.edge_processor.edge_max_importance, "float"),
                                             ('max_sample_points', self.edge_processor.max_sample_points, 'int'),
                                             ('show_class', show_class, 'int'),
                                             ('edge_importance_type', 0, 'int')])
        self.sphere_render.set_uniform_labeled_data(config)

        def render_function(sample_points: int):
            render_setting_0(False)
            glDrawArraysInstanced(GL_POINTS, 0, 1, sample_points)
            glMemoryBarrier(GL_ALL_BARRIER_BITS)

        self.sphere_render.render(render_function, self.edge_processor.get_buffer_points)

    @track_time
    def render_transparent_sphere(self, cam: Camera, sphere_radius: float = 0.05, config: RenderingConfig = None,
                                  show_class: int = 0):
        self.transparent_render.buffer_divisor = [(0, 1), (1, self.edge_processor.max_sample_points)]

        near, far = self.grid.get_near_far_from_view(cam.view)
        self.transparent_render.set_uniform_data([("projection", cam.projection, "mat4"),
                                                  ("view", cam.view, "mat4"),
                                                  ("farthest_point_view_z", far, "float"),
                                                  ("nearest_point_view_z", near, "float"),
                                                  ("object_radius", sphere_radius, "float"),
                                                  ("importance_threshold", self.importance_threshold, "float"),
                                                  ("importance_max", self.edge_processor.edge_max_importance, "float"),
                                                  ('max_sample_points', self.edge_processor.max_sample_points, 'int'),
                                                  ('show_class', show_class, 'int'),
                                                  ('edge_importance_type', 0, 'int')])
        self.transparent_render.set_uniform_labeled_data(config)

        def render_function(sample_points: int):
            render_setting_1(False)
            glDrawArraysInstanced(GL_POINTS, 0, 1, sample_points)
            glMemoryBarrier(GL_ALL_BARRIER_BITS)

        self.transparent_render.render(render_function, self.edge_processor.get_buffer_points)

    @track_time
    def render_ellipsoid_transparent(self, cam: Camera, config: RenderingConfig = None, show_class: int = 0):
        self.ellipse_render.buffer_divisor = [(0, 1), (1, self.edge_processor.max_sample_points)]

        near, far = self.grid.get_near_far_from_view(cam.view)
        self.ellipse_render.set_uniform_data([("projection", cam.projection, "mat4"),
                                              ("view", cam.view, "mat4"),
                                              ("farthest_point_view_z", far, "float"),
                                              ("nearest_point_view_z", near, "float"),
                                              ("object_radius", self.edge_processor.sample_length * 0.5, "float"),
                                              ("importance_threshold", self.importance_threshold, "float"),
                                              ("importance_max", self.edge_processor.edge_max_importance, "float"),
                                              ('max_sample_points', self.edge_processor.max_sample_points, 'int'),
                                              ('show_class', show_class, 'int'),
                                              ('edge_importance_type', 0, 'int')])
        self.ellipse_render.set_uniform_labeled_data(config)

        def render_function(sample_points: int):
            render_setting_1(False)
            glDrawArraysInstanced(GL_POINTS, 0, 1, sample_points)
            glMemoryBarrier(GL_ALL_BARRIER_BITS)

        self.ellipse_render.render(render_function, self.edge_processor.get_buffer_points)

    def delete(self):
        for layer_data_handler in self.data_handler:
            for container_data_handler in layer_data_handler:
                container_data_handler.delete()
        self.data_handler = []
