from typing import List, Tuple

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
                                               "node/point_to_sphere_impostor.geom"]),
                                ShaderSetting("node_transparent_sphere", ["node/sample_impostor.vert",
                                                                          "node/point_to_sphere_impostor_transparent.frag",
                                                                          "node/point_to_sphere_impostor.geom"])
                                ])
        self.set_shader(shader_settings)

        self.data_handler: VertexDataHandler = VertexDataHandler([(self.node_processor.node_buffer, 0)])

        def point_render_func():
            node_count: int = len(self.node_processor.nodes)
            render_setting_0(False)
            glPointSize(10.0)
            glDrawArrays(GL_POINTS, 0, node_count)
            glMemoryBarrier(GL_ALL_BARRIER_BITS)

        self.render_funcs["node_point"] = point_render_func

        def point_render_func():
            node_count: int = len(self.node_processor.nodes)
            render_setting_0(False)
            glPointSize(10.0)
            glDrawArrays(GL_POINTS, 0, node_count)
            glMemoryBarrier(GL_ALL_BARRIER_BITS)

        self.render_funcs["node_point"] = point_render_func

        # self.create_sets(nod)

        self.point_render: RenderSet = RenderSet(node_point_shader, self.data_handler)
        self.sphere_render: RenderSet = RenderSet(node_sphere_shader, self.data_handler)
        self.transparent_render: RenderSet = RenderSet(node_transparent_shader, self.data_handler)

        self.sphere_render.set_uniform_label(["node_object_radius", "node_importance_threshold"])
        self.transparent_render.set_uniform_label(
            ["node_object_radius", "node_base_opacity", "node_importance_opacity", "node_depth_opacity",
             "node_opacity_exponent", "node_importance_threshold"])

    @track_time
    def render(self, set_name: str, cam: Camera, config: RenderingConfig = None, show_class: int = 0):
        current_set: BaseRenderSet = self.sets[set_name]
        current_set.set_uniform_data([("projection", cam.projection, "mat4"),
                                      ("view", cam.view, "mat4")])
        current_set.set_uniform_labeled_data(config)
        current_set.render()

    @track_time
    def render_point(self, cam: Camera, config: RenderingConfig = None, show_class: int = 0):
        self.point_render.set_uniform_data([("projection", cam.projection, "mat4"),
                                            ("view", cam.view, "mat4")])
        self.point_render.set_uniform_labeled_data(config)

        self.point_render.set()

        render_setting_0(False)
        glPointSize(10.0)
        node_count: int = len(self.node_processor.nodes)
        glDrawArrays(GL_POINTS, 0, node_count)
        glMemoryBarrier(GL_ALL_BARRIER_BITS)

    @track_time
    def render_sphere(self, cam: Camera, sphere_radius: float = 0.03, config: RenderingConfig = None,
                      show_class: int = 0):
        node_count: int = len(self.node_processor.nodes)

        self.sphere_render.set_uniform_data([("projection", cam.projection, "mat4"),
                                             ("view", cam.view, "mat4"),
                                             ("object_radius", sphere_radius, "float"),
                                             ("importance_max", self.node_processor.node_max_importance, "float"),
                                             ('show_class', show_class, 'int')])
        self.sphere_render.set_uniform_labeled_data(config)

        self.sphere_render.set()

        render_setting_0(False)
        glDrawArrays(GL_POINTS, 0, node_count)
        glMemoryBarrier(GL_ALL_BARRIER_BITS)

    @track_time
    def render_transparent(self, cam: Camera, sphere_radius: float = 0.03, config: RenderingConfig = None,
                           show_class: int = 0):
        node_count: int = len(self.node_processor.nodes)

        near, far = self.grid.get_near_far_from_view(cam.view)
        self.transparent_render.set_uniform_data([("projection", cam.projection, "mat4"),
                                                  ("view", cam.view, "mat4"),
                                                  ("farthest_point_view_z", far, "float"),
                                                  ("nearest_point_view_z", near, "float"),
                                                  ("object_radius", sphere_radius, "float"),
                                                  ("importance_max", self.node_processor.node_max_importance, "float"),
                                                  ('show_class', show_class, 'int')])
        self.transparent_render.set_uniform_labeled_data(config)

        self.transparent_render.set()

        render_setting_1(False)
        glDrawArrays(GL_POINTS, 0, node_count)
        glMemoryBarrier(GL_ALL_BARRIER_BITS)

    def delete(self):
        self.data_handler.delete()
