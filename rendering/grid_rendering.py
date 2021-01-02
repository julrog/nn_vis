from typing import List, Callable

from OpenGL.GL import *
from opengl_helper.render_utility import OverflowingVertexDataHandler, generate_render_function, OGLRenderFunction, \
    BaseRenderSet
from opengl_helper.shader import ShaderSetting
from processing.grid_processing import GridProcessor
from rendering.rendering import Renderer
from rendering.rendering_config import RenderingConfig
from utility.camera import Camera
from utility.performance import track_time


class GridRenderer(Renderer):
    def __init__(self, grid_processor: GridProcessor):
        Renderer.__init__(self)
        self.grid_processor: GridProcessor = grid_processor

        shader_settings: List[ShaderSetting] = []
        shader_settings.extend([ShaderSetting("grid_point", ["grid/grid.vert", "basic/discard_screen_color.frag"]),
                                ShaderSetting("grid_cube", ["grid/grid_impostor.vert", "basic/screen_color.frag",
                                                            "grid/point_to_cube_impostor.geom"])
                                ])
        self.set_shader(shader_settings)

        self.data_handler: OverflowingVertexDataHandler = OverflowingVertexDataHandler(
            [], [(self.grid_processor.grid_position_buffer, 0), (self.grid_processor.grid_density_buffer, 1)])

        def generate_element_count_func(gp: GridProcessor) -> Callable:
            buffered_gp: GridProcessor = gp

            def element_count_func(buffer: int):
                return buffered_gp.grid_density_buffer.get_objects() - buffered_gp.grid_slice_size

            return element_count_func

        self.render_funcs["grid_point"] = generate_render_function(OGLRenderFunction.ARRAYS, GL_POINTS, 10.0,
                                                                   depth_test=True)
        self.render_funcs["grid_cube"] = generate_render_function(OGLRenderFunction.ARRAYS, GL_POINTS,
                                                                  depth_test=True)
        self.element_count_funcs["grid_point"] = generate_element_count_func(grid_processor)
        self.element_count_funcs["grid_cube"] = generate_element_count_func(grid_processor)

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
