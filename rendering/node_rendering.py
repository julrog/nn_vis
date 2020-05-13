from OpenGL.GL import *

from models.grid import Grid
from opengl_helper.render_utility import VertexDataHandler, RenderSet, render_setting_0, render_setting_1
from opengl_helper.shader import RenderShaderHandler, RenderShader
from processing.node_processing import NodeProcessor
from utility.performance import track_time
from utility.window import Window


class NodeRenderer:
    def __init__(self, node_processor: NodeProcessor, grid: Grid):
        self.node_processor = node_processor
        self.grid = grid

        shader_handler: RenderShaderHandler = RenderShaderHandler()
        sample_point_shader: RenderShader = shader_handler.create("base", "sample/point.vert", "sample/point.frag")
        sample_sphere_shader: RenderShader = shader_handler.create("sample", "sample/ball_from_point.vert",
                                                                   "sample/ball_from_point.frag",
                                                                   "sample/ball_from_point.geom")
        sample_transparent_shader: RenderShader = shader_handler.create("trans", "sample/ball_from_point.vert",
                                                                        "sample/transparent_ball.frag",
                                                                        "sample/ball_from_point.geom")

        self.data_handler: VertexDataHandler = VertexDataHandler([(self.node_processor.node_buffer, 0)])

        self.point_render: RenderSet = RenderSet(sample_point_shader, self.data_handler)
        self.sphere_render: RenderSet = RenderSet(sample_sphere_shader, self.data_handler)
        self.transparent_render: RenderSet = RenderSet(sample_transparent_shader, self.data_handler)

    @track_time
    def render_point(self, window: Window, clear: bool = True, swap: bool = False):
        node_count: int = len(self.node_processor.nodes)

        self.point_render.set_uniform_data([("projection", window.cam.projection, "mat4"),
                                            ("view", window.cam.view, "mat4")])

        self.point_render.set()

        render_setting_0(clear)
        glPointSize(10.0)
        glDrawArrays(GL_POINTS, 0, node_count)
        if swap:
            window.swap()

    @track_time
    def render_sphere(self, window: Window, clear: bool = True, swap: bool = False):
        node_count: int = len(self.node_processor.nodes)

        self.sphere_render.set_uniform_data([("projection", window.cam.projection, "mat4"),
                                             ("view", window.cam.view, "mat4")])

        self.sphere_render.set()

        render_setting_0(clear)
        glDrawArrays(GL_POINTS, 0, node_count)
        if swap:
            window.swap()

    @track_time
    def render_transparent(self, window: Window, clear: bool = True, swap: bool = False):
        node_count: int = len(self.node_processor.nodes)

        near, far = self.grid.get_near_far_from_view(window.cam.view)
        self.transparent_render.set_uniform_data([("projection", window.cam.projection, "mat4"),
                                                  ("view", window.cam.view, "mat4"),
                                                  ("farthest_point_view_z", far, "float"),
                                                  ("nearest_point_view_z", near, "float")])

        self.transparent_render.set()

        render_setting_1(clear)
        glDrawArrays(GL_POINTS, 0, node_count)
        if swap:
            window.swap()

    def delete(self):
        self.data_handler.delete()
