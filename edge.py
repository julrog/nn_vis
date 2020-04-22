import math
from random import random
from typing import List, Tuple
from pyrr import Vector3, matrix44, Vector4, vector3, Matrix44
import numpy as np
from OpenGL.GL import *

from compute_shader import ComputeShader, ComputeShaderHandler
from performance import track_time
from render_helper import RenderSet, VertexDataHandler, render_setting_0, render_setting_1, SwappingBufferObject, \
    BufferObject
from shader import RenderShader, RenderShaderHandler
from window import Window

LOG_SOURCE: str = "EDGE"


class Edge:
    def __init__(self, start_position: Vector3, end_position: Vector3):
        self.start: Vector4 = Vector4([start_position.x, start_position.y, start_position.z, 1.0])
        self.end: Vector4 = Vector4([end_position.x, end_position.y, end_position.z, 0.0])
        self.initial_data: List[float] = [start_position.x, start_position.y, start_position.z, 1.0, end_position.x,
                                          end_position.y, end_position.z, 0.0]
        self.sample_points: List[Vector4] = [self.start, self.end]


class EdgeHandler:
    def __init__(self, sample_length: float):
        self.sample_compute_shader: ComputeShader = ComputeShaderHandler().create("edge_sampler",
                                                                                  "edge_sample.comp")
        self.noise_compute_shader: ComputeShader = ComputeShaderHandler().create("edge_noise",
                                                                                 "edge_noise.comp")
        self.limit_compute_shader: ComputeShader = ComputeShaderHandler().create("edge_limits",
                                                                                 "edge_limits.comp")
        self.sample_buffer: SwappingBufferObject = SwappingBufferObject(True)
        self.limits_buffer: BufferObject = BufferObject(True)
        self.ssbo_handler: VertexDataHandler = VertexDataHandler([(self.sample_buffer, 0), (self.limits_buffer, 2)])

        self.edges: List[Edge] = []
        self.sampled: bool = False
        self.sample_length: float = sample_length

        self.point_count: int = 0
        self.nearest_view_z: int = -1000000
        self.farthest_view_z: int = 1000000
        self.max_sample_points: int = 0

    def set_data(self, node_positions_layer_one: List[float], node_positions_layer_two: List[float]):
        def group_wise(it):
            it = iter(it)
            while True:
                yield next(it), next(it), next(it)

        # generate edges
        vec_nodes_layer_one: List[Vector3] = [Vector3([x, y, z]) for x, y, z in group_wise(node_positions_layer_one)]
        vec_nodes_layer_two: List[Vector3] = [Vector3([x, y, z]) for x, y, z in group_wise(node_positions_layer_two)]
        self.edges = []
        for node_one in vec_nodes_layer_one:
            for node_two in vec_nodes_layer_two:
                self.edges.append(Edge(node_one, node_two))

        #  estimate a suitable sample size for buffer objects
        max_distance = 0
        for node_one in vec_nodes_layer_one:
            for node_two in vec_nodes_layer_two:
                test_distance = (node_one - node_two).length
                if test_distance > max_distance:
                    max_distance = test_distance
        self.max_sample_points = int((max_distance * 2.0) / self.sample_length)

        # generate and load initial data for the buffer
        initial_data: List[float] = []
        for edge in self.edges:
            initial_data.extend(edge.initial_data)
            initial_data.extend([0] * (self.max_sample_points * 4 - len(edge.initial_data)))
        transfer_data = np.array(initial_data, dtype=np.float32)
        self.sample_buffer.load(transfer_data)
        self.sample_buffer.swap()
        self.sample_buffer.load(transfer_data)

    @track_time
    def resize_sample_storage(self, new_max_samples: int):
        edge_sample_data = np.array(self.read_samples_from_sample_storage(raw=True, auto_resize_enabled=False),
                                    dtype=np.float32)
        edge_sample_data = edge_sample_data.reshape((len(self.edges), self.max_sample_points * 4))

        self.max_sample_points = new_max_samples

        buffer_data = []
        for i in range(len(self.edges)):
            edge_points: int = int(edge_sample_data[i][3])
            buffer_data.extend(edge_sample_data[i][None:(int(edge_points * 4))])
            buffer_data.extend([0] * (self.max_sample_points * 4 - edge_points * 4))

        transfer_data = np.array(buffer_data, dtype=np.float32)

        self.sample_buffer.load(transfer_data)
        self.sample_buffer.swap()
        self.sample_buffer.load(transfer_data)

    @track_time
    def sample_edges(self, sample_length: float = None):
        if sample_length is not None:
            self.sample_length = sample_length

        self.ssbo_handler.set()
        self.sample_compute_shader.set_uniform_data([('sample_length', self.sample_length, 'float')])
        self.sample_compute_shader.set_uniform_data([('max_sample_points', self.max_sample_points, 'int')])

        for i in range(math.ceil(len(self.edges) / self.sample_compute_shader.max_workgroup_size)):
            self.sample_compute_shader.set_uniform_data(
                [('work_group_offset', i * self.sample_compute_shader.max_workgroup_size, 'int')])

            if i == math.ceil(len(self.edges) / self.sample_compute_shader.max_workgroup_size) - 1:
                self.sample_compute_shader.compute(len(self.edges) % self.sample_compute_shader.max_workgroup_size)
            else:
                self.sample_compute_shader.compute(self.sample_compute_shader.max_workgroup_size)

        self.sample_buffer.swap()
        self.sampled = True

    @track_time
    def sample_noise(self, strength: float = 1.0):
        self.ssbo_handler.set()

        self.noise_compute_shader.set_uniform_data([('sample_length', self.sample_length, 'float')])
        self.noise_compute_shader.set_uniform_data([('noise_strength', strength, 'float')])
        self.noise_compute_shader.set_uniform_data([('max_sample_points', self.max_sample_points, 'int')])

        for i in range(math.ceil(len(self.edges) / self.noise_compute_shader.max_workgroup_size)):
            self.noise_compute_shader.set_uniform_data(
                [('work_group_offset', i * self.noise_compute_shader.max_workgroup_size, 'int')])

            if i == math.ceil(len(self.edges) / self.noise_compute_shader.max_workgroup_size) - 1:
                self.noise_compute_shader.compute(len(self.edges) % self.noise_compute_shader.max_workgroup_size)
            else:
                self.noise_compute_shader.compute(self.noise_compute_shader.max_workgroup_size)

        self.sample_buffer.swap()

    @track_time
    def read_samples_from_sample_storage(self, raw: bool = False, auto_resize_enabled: bool = True) -> List[float]:
        edge_sample_data = np.frombuffer(self.sample_buffer.read(), dtype=np.float32)
        if raw:
            return edge_sample_data

        edge_sample_data = edge_sample_data.reshape((len(self.edges), self.max_sample_points * 4))
        buffer_data = []
        max_edge_samples = 0
        self.point_count = 0
        for i in range(len(self.edges)):
            edge_points: int = int(edge_sample_data[i][3])
            buffer_data.extend(edge_sample_data[i][None:(int(edge_points * 4))])
            self.point_count += edge_points
            if edge_points > max_edge_samples:
                max_edge_samples = edge_points

        if auto_resize_enabled and max_edge_samples >= (self.max_sample_points - 5) * 0.8:
            self.resize_sample_storage(max_edge_samples * 2)

        return buffer_data

    @track_time
    def check_limits(self, view: Matrix44):
        self.limits_buffer.load(np.array([0, -1000000, 1000000, 0], dtype=int32_t))

        self.ssbo_handler.set()

        self.limit_compute_shader.set_uniform_data([('view', view, 'mat4')])
        self.limit_compute_shader.set_uniform_data([('max_sample_points', self.max_sample_points, 'int')])

        for i in range(math.ceil(len(self.edges) / self.limit_compute_shader.max_workgroup_size)):
            self.limit_compute_shader.set_uniform_data(
                [('work_group_offset', i * self.limit_compute_shader.max_workgroup_size, 'int')])

            if i == math.ceil(len(self.edges) / self.limit_compute_shader.max_workgroup_size) - 1:
                self.limit_compute_shader.compute(len(self.edges) % self.limit_compute_shader.max_workgroup_size)
            else:
                self.limit_compute_shader.compute(self.limit_compute_shader.max_workgroup_size)

        limits: List[int] = np.frombuffer(self.limits_buffer.read(), dtype=int32_t)
        self.point_count = limits[0]
        self.nearest_view_z = limits[1]
        self.farthest_view_z = limits[2]
        max_edge_samples = limits[3]
        if max_edge_samples >= (self.max_sample_points - 5) * 0.8:
            self.resize_sample_storage(int(max_edge_samples * 2))

    @track_time
    def get_near_far_from_view(self) -> Tuple[float, float]:
        return -0.49, -2.5  # self.nearest_view_z / 1000.0, self.farthest_view_z / 1000.0

    @track_time
    def get_buffer_points(self) -> int:
        return int(self.sample_buffer.size / 16.0)


class EdgeRenderer:
    def __init__(self, edge_handler: EdgeHandler):
        self.edge_handler = edge_handler

        shader_handler = RenderShaderHandler()
        sample_point_shader: RenderShader = shader_handler.create("base", "base.vert", "base.frag")
        sample_sphere_shader: RenderShader = shader_handler.create("ball", "ball/ball_from_point.vert",
                                                                   "ball/ball_from_point.frag",
                                                                   "ball/ball_from_point.geom")
        sample_transparent_shader: RenderShader = shader_handler.create("trans", "ball/ball_from_point.vert",
                                                                        "ball/transparent_ball.frag",
                                                                        "ball/ball_from_point.geom")

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
