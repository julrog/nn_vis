from OpenGL.GL import *
from opengl_helper.render_utility import RenderSet, VertexDataHandler, render_setting_0
from opengl_helper.shader import RenderShaderHandler, RenderShader
from processing.grid_processing import GridProcessor
from utility.performance import track_time
from utility.window import Window


class GridRenderer:
    def __init__(self, grid_processor: GridProcessor):
        self.grid_processor = grid_processor

        shader_handler = RenderShaderHandler()
        point_shader: RenderShader = shader_handler.create("grid_base", "grid/point.vert", "grid/point.frag")
        cube_shader: RenderShader = shader_handler.create("grid_cube", "grid/cube.vert",
                                                          "grid/cube.frag",
                                                          "grid/cube.geom")

        self.data_handler: VertexDataHandler = VertexDataHandler(
            [(self.grid_processor.grid_position_buffer, 0), (self.grid_processor.grid_density_buffer, 1)])

        self.point_render: RenderSet = RenderSet(point_shader, self.data_handler)
        self.cube_render: RenderSet = RenderSet(cube_shader, self.data_handler)

    @track_time
    def render_point(self, window: Window, clear: bool = True, swap: bool = False):
        grid_count: int = self.grid_processor.grid.grid_cell_count_overall

        self.point_render.set_uniform_data([("projection", window.cam.projection, "mat4"),
                                            ("view", window.cam.view, "mat4")])

        self.point_render.set()

        render_setting_0(clear)
        glPointSize(10.0)
        glDrawArrays(GL_POINTS, 0, grid_count)
        if swap:
            window.swap()

    @track_time
    def render_cube(self, window: Window, clear: bool = True, swap: bool = False):
        grid_count: int = self.grid_processor.grid.grid_cell_count_overall

        self.cube_render.set_uniform_data([("projection", window.cam.projection, "mat4"),
                                           ("view", window.cam.view, "mat4")])

        self.cube_render.set()

        render_setting_0(clear)
        glDrawArrays(GL_POINTS, 0, grid_count)
        if swap:
            window.swap()
