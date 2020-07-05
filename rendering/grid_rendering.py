from typing import Dict

from OpenGL.GL import *
from opengl_helper.render_utility import RenderSet, VertexDataHandler, render_setting_0, OverflowingVertexDataHandler, \
    OverflowingRenderSet
from opengl_helper.shader import RenderShaderHandler, RenderShader
from processing.grid_processing import GridProcessor
from utility.camera import Camera
from utility.performance import track_time
from utility.window import Window


class GridRenderer:
    def __init__(self, grid_processor: GridProcessor):
        self.grid_processor = grid_processor

        shader_handler = RenderShaderHandler()
        point_shader: RenderShader = shader_handler.create("grid_point", "grid/grid.vert",
                                                           "basic/discard_screen_color.frag")
        cube_shader: RenderShader = shader_handler.create("grid_cube", "grid/grid_impostor.vert",
                                                          "basic/screen_color.frag",
                                                          "grid/point_to_cube_impostor.geom")

        self.data_handler: OverflowingVertexDataHandler = OverflowingVertexDataHandler(
            [], [(self.grid_processor.grid_position_buffer, 0), (self.grid_processor.grid_density_buffer, 1)])

        self.point_render: OverflowingRenderSet = OverflowingRenderSet(point_shader, self.data_handler)
        self.cube_render: OverflowingRenderSet = OverflowingRenderSet(cube_shader, self.data_handler)

    @track_time
    def render_point(self, cam: Camera, options: Dict[str, float] = None):
        self.point_render.set_uniform_data([("projection", cam.projection, "mat4"),
                                            ("view", cam.view, "mat4"),
                                            ("screen_width", 1920.0, "float"),
                                            ("screen_height", 1080.0, "float")])
        self.point_render.set_uniform_labeled_data(options)

        for i in range(len(self.grid_processor.grid_density_buffer.handle)):
            grid_count: int = self.grid_processor.grid_density_buffer.get_objects() - self.grid_processor.grid_slice_size

            self.point_render.set(i)

            render_setting_0(False)
            glPointSize(10.0)
            glDrawArrays(GL_POINTS, 0, grid_count)
            glMemoryBarrier(GL_ALL_BARRIER_BITS)

    @track_time
    def render_cube(self, cam: Camera, options: Dict[str, float] = None):
        self.cube_render.set_uniform_data([("projection", cam.projection, "mat4"),
                                           ("view", cam.view, "mat4"),
                                           ("screen_width", 1920.0, "float"),
                                           ("screen_height", 1080.0, "float")])
        self.cube_render.set_uniform_labeled_data(options)

        for i in range(len(self.grid_processor.grid_density_buffer.handle)):
            grid_count: int = self.grid_processor.grid_density_buffer.get_objects() - self.grid_processor.grid_slice_size

            self.cube_render.set(i)

            render_setting_0(False)

            glDrawArrays(GL_POINTS, 0, grid_count)
            glMemoryBarrier(GL_ALL_BARRIER_BITS)

    def delete(self):
        self.data_handler.delete()
