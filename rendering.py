from OpenGL.GL import *

from opengl_helper.render_utility import VertexDataHandler, RenderSet, render_setting_0, render_setting_1
from opengl_helper.shader import RenderShaderHandler, RenderShader
from processing import EdgeProcessor
from utility.performance import track_time
from utility.window import Window


class EdgeRenderer:
    def __init__(self, edge_handler: EdgeProcessor):
        self.edge_handler = edge_handler

        shader_handler = RenderShaderHandler()
        sample_point_shader: RenderShader = shader_handler.create("base", "sample/point.vert", "sample/point.frag")
        sample_sphere_shader: RenderShader = shader_handler.create("sample", "sample/ball_from_point.vert",
                                                                   "sample/ball_from_point.frag",
                                                                   "sample/ball_from_point.geom")
        sample_transparent_shader: RenderShader = shader_handler.create("trans", "sample/ball_from_point.vert",
                                                                        "sample/transparent_ball.frag",
                                                                        "sample/ball_from_point.geom")

        self.data_handler: VertexDataHandler = VertexDataHandler([(self.edge_handler.sample_buffer, 0)])

        self.point_render: RenderSet = RenderSet(sample_point_shader, self.data_handler)
        self.sphere_render: RenderSet = RenderSet(sample_sphere_shader, self.data_handler)
        self.transparent_render: RenderSet = RenderSet(sample_transparent_shader, self.data_handler)

    @track_time
    def render_point(self, window: Window, swap: bool = False):
        sampled_points: int = self.edge_handler.get_buffer_points()

        self.point_render.set_uniform_data([("projection", window.cam.projection, "mat4"),
                                            ("view", window.cam.get_view_matrix(), "mat4"),
                                            ("point_color", [1.0, 0.0, 0.0], "vec3")])

        self.point_render.set()

        render_setting_0()
        glPointSize(1.0)
        glDrawArrays(GL_POINTS, 0, sampled_points)
        if swap:
            window.swap()

    @track_time
    def render_sphere(self, window: Window, swap: bool = False):
        sampled_points: int = self.edge_handler.get_buffer_points()

        self.sphere_render.set_uniform_data([("projection", window.cam.projection, "mat4"),
                                             ("view", window.cam.get_view_matrix(), "mat4")])

        self.sphere_render.set()

        render_setting_0()
        glDrawArrays(GL_POINTS, 0, sampled_points)
        if swap:
            window.swap()

    @track_time
    def render_transparent(self, window: Window, swap: bool = False):
        sampled_points: int = self.edge_handler.get_buffer_points()

        near, far = self.edge_handler.get_near_far_from_view()
        self.transparent_render.set_uniform_data([("projection", window.cam.projection, "mat4"),
                                                  ("view", window.cam.get_view_matrix(), "mat4"),
                                                  ("farthest_point_view_z", far, "float"),
                                                  ("nearest_point_view_z", near, "float")])

        self.transparent_render.set()

        render_setting_1()
        glDrawArrays(GL_POINTS, 0, sampled_points)
        if swap:
            window.swap()
