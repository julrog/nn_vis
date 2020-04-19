import math
from random import random
from typing import List, Tuple
from pyrr import Vector3, matrix44, Vector4, vector3
import numpy as np
from OpenGL.GL import *

from compute_shader import ComputeShader, ComputeShaderHandler
from performance import track_time
from render_helper import RenderSet, VertexDataHandler, render_setting_0, render_setting_1
from shader import RenderShader, RenderShaderHandler
from texture import Texture
from window import Window

LOG_SOURCE: str = "EDGE"


class Edge:
    def __init__(self, start_position: Vector3, end_position: Vector3):
        self.start: Vector4 = Vector4([start_position.x, start_position.y, start_position.z, 1.0])
        self.end: Vector4 = Vector4([end_position.x, end_position.y, end_position.z, 0.0])
        self.sample_points: List[Vector4] = [self.start, self.end]
        self.sample_length: float = 0.0
        self.squared_sample_length: float = 0.0

    def set_sample_points(self, sample_length: float, data):
        self.sample_length = sample_length
        self.squared_sample_length = self.sample_length * self.sample_length
        self.sample_points = []

        def group_wise(it):
            it = iter(it)
            while True:
                yield next(it), next(it), next(it), next(it)

        for x, y, z, w in group_wise(data):
            if w > 0.0:
                self.sample_points.append(Vector4([x, y, z, w]))
            else:
                self.sample_points.append(Vector4([x, y, z, w]))
                return

    def sample(self, sample_length: float):
        self.sample_length = sample_length
        self.squared_sample_length = self.sample_length * self.sample_length
        if sample_length:
            self.sample_length = sample_length

        new_sample_points: List[Vector4] = []
        previous_added_point: Vector3 = Vector3(vector3.create_from_vector4(self.sample_points[0])[0])
        previous_looked_point: Vector3 = previous_added_point
        new_sample_points.append(self.sample_points[0])
        for i in range(1, len(self.sample_points)):
            current_point: Vector3 = Vector3(vector3.create_from_vector4(self.sample_points[i])[0])
            while Vector3(current_point - previous_added_point).length >= self.sample_length * 0.99:
                prev_distance = (previous_looked_point - previous_added_point).length
                current_distance = (current_point - previous_added_point).length
                t = (self.sample_length - prev_distance) / (current_distance - prev_distance)
                np.clip(t, 0.0, 1.0)
                interpolated_point = (previous_added_point * (1.0 - t) + current_point * t)
                new_sample_points.append(
                    Vector4([interpolated_point.x, interpolated_point.y, interpolated_point.z, 1.0]))
                previous_added_point = interpolated_point
                previous_looked_point = previous_added_point
            previous_looked_point = current_point
        new_sample_points.append(self.sample_points[-1])
        self.sample_points = new_sample_points

    def sample_noise(self, strength: float = 1.0):
        if not self.sample_points:
            return
        noise_strength = strength * self.sample_length
        for i in range(len(self.sample_points)):
            self.sample_points[i] = self.sample_points[i] + Vector4(
                [random() * noise_strength * 2.0 - noise_strength,
                 random() * noise_strength * 2.0 - noise_strength,
                 random() * noise_strength * 2.0 - noise_strength,
                 0.0])

    def get_edge_length(self) -> float:
        if not self.sample_points:
            raise Exception("[EDGE] Not yet sampled, can't get length of edge!")
        length = 0.0
        previous_point = None
        for point in self.sample_points:
            if previous_point:
                distance_vector: Vector3 = Vector3(vector3.create_from_vector4(point)[0]) - Vector3(
                    vector3.create_from_vector4(previous_point)[0])
                length += distance_vector.length
        return length


class EdgeHandler:
    def __init__(self, sample_length: float, use_compute: bool = False):
        self.use_compute = use_compute
        if self.use_compute:
            self.sample_compute_shader: ComputeShader = ComputeShaderHandler().create("edge_sampler",
                                                                                      "edge_sample.comp")
            self.noise_compute_shader: ComputeShader = ComputeShaderHandler().create("edge_noise",
                                                                                     "edge_noise.comp")
            self.ssbo_handler: VertexDataHandler or None = None
            self.switched_sample_ssbos: bool = False
            self.max_distance: float = 0
            self.max_sample_points: int = 0
            self.point_count: int = 0
        self.edges: List[Edge] = []
        self.sampled: bool = False
        self.sample_length: float = sample_length

    def set_data(self, node_positions_layer_one: List[float], node_positions_layer_two: List[float]):
        def group_wise(it):
            it = iter(it)
            while True:
                yield next(it), next(it), next(it)

        vec_nodes_layer_one: List[Vector3] = [Vector3([x, y, z]) for x, y, z in group_wise(node_positions_layer_one)]
        vec_nodes_layer_two: List[Vector3] = [Vector3([x, y, z]) for x, y, z in group_wise(node_positions_layer_two)]
        self.edges = []
        for node_one in vec_nodes_layer_one:
            for node_two in vec_nodes_layer_two:
                self.edges.append(Edge(node_one, node_two))

        if self.use_compute:
            for node_one in vec_nodes_layer_one:
                for node_two in vec_nodes_layer_two:
                    test_distance = (node_one - node_two).length
                    if test_distance > self.max_distance:
                        self.max_distance = test_distance

            self.max_sample_points = int((self.max_distance * 2.0) / self.sample_length)
            initial_data: List[float] = []
            for edge in self.edges:
                point_data = []
                point_data.extend(edge.start.data)
                point_data.extend(edge.end.data)
                initial_data.extend(point_data)
                initial_data.extend([0] * (self.max_sample_points * 4 - len(point_data)))

            transfer_data = np.array(initial_data, dtype=np.float32)

            self.ssbo_handler = VertexDataHandler(ssbos=2)
            self.ssbo_handler.load_ssbo_data(transfer_data, location=0, buffer_id=0)
            self.ssbo_handler.load_ssbo_data(transfer_data, location=1, buffer_id=1)

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

        self.ssbo_handler.load_ssbo_data(transfer_data, location=0, buffer_id=0)
        self.ssbo_handler.load_ssbo_data(transfer_data, location=1, buffer_id=1)

    @track_time
    def sample_edges(self, sample_length: float = None):
        if sample_length is not None:
            self.sample_length = sample_length

        if self.use_compute:
            self.ssbo_handler.set()
            self.ssbo_handler.bind_ssbo_data(0, 1 if self.switched_sample_ssbos else 0)
            self.ssbo_handler.bind_ssbo_data(1, 0 if self.switched_sample_ssbos else 1)
            self.sample_compute_shader.set_uniform_data([('sample_length', self.sample_length, 'float')])
            self.sample_compute_shader.set_uniform_data([('max_sample_points', self.max_sample_points, 'int')])

            print("[%s] Sample %d edges." % (LOG_SOURCE, len(self.edges)))
            for i in range(math.ceil(len(self.edges) / self.sample_compute_shader.max_workgroup_size)):
                self.sample_compute_shader.set_uniform_data(
                    [('work_group_offset', i * self.sample_compute_shader.max_workgroup_size, 'int')])

                if i == math.ceil(len(self.edges) / self.sample_compute_shader.max_workgroup_size) - 1:
                    self.sample_compute_shader.compute(len(self.edges) % self.sample_compute_shader.max_workgroup_size)
                else:
                    self.sample_compute_shader.compute(self.sample_compute_shader.max_workgroup_size)

            self.switched_sample_ssbos = not self.switched_sample_ssbos
        else:
            for edge in self.edges:
                edge.sample(self.sample_length)
        self.sampled = True

    @track_time
    def generate_buffer_data(self):
        if not self.sampled:
            raise Exception("[%s] Not sampled, can't generate data!" % LOG_SOURCE)

        if self.use_compute:
            return np.array(self.read_samples_from_sample_storage(), dtype=np.float32)
        else:
            buffer_data: List[Vector4] = []
            for edge in self.edges:
                for point in edge.sample_points:
                    buffer_data.extend(point)
            return np.array(buffer_data, dtype=np.float32)

    @track_time
    def get_buffer_points(self) -> int:
        if self.use_compute:
            return self.point_count
        else:
            return sum([len(edge.sample_points) for edge in self.edges])

    @track_time
    def sample_noise(self, strength: float = 1.0):
        if self.use_compute:
            self.ssbo_handler.set()
            self.ssbo_handler.bind_ssbo_data(0, 1 if self.switched_sample_ssbos else 0)
            self.ssbo_handler.bind_ssbo_data(1, 0 if self.switched_sample_ssbos else 1)

            self.noise_compute_shader.set_uniform_data([('sample_length', self.sample_length, 'float')])
            self.noise_compute_shader.set_uniform_data([('noise_strength', strength, 'float')])
            self.noise_compute_shader.set_uniform_data([('max_sample_points', self.max_sample_points, 'int')])

            print("[%s] Add noise to %d edges." % (LOG_SOURCE, len(self.edges)))
            for i in range(math.ceil(len(self.edges) / self.noise_compute_shader.max_workgroup_size)):
                self.noise_compute_shader.set_uniform_data(
                    [('work_group_offset', i * self.noise_compute_shader.max_workgroup_size, 'int')])

                if i == math.ceil(len(self.edges) / self.noise_compute_shader.max_workgroup_size) - 1:
                    self.noise_compute_shader.compute(len(self.edges) % self.noise_compute_shader.max_workgroup_size)
                else:
                    self.noise_compute_shader.compute(self.noise_compute_shader.max_workgroup_size)

            self.switched_sample_ssbos = not self.switched_sample_ssbos
        else:
            for edge in self.edges:
                edge.sample_noise(strength)

    @track_time
    def read_samples_from_sample_storage(self, raw: bool = False, auto_resize_enabled: bool = True) -> List[float]:
        edge_sample_data = np.frombuffer(self.ssbo_handler.read_ssbo_data(1 if self.switched_sample_ssbos else 0),
                                         dtype=np.float32)
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
    def get_extends(self) -> Tuple[List[float], List[float]]:
        min_extend: Vector3 = Vector3([10000.0, 10000.0, 10000.0])
        max_extend: Vector3 = Vector3([-10000.0, -10000.0, -10000.0])
        for edge in self.edges:
            for sample in edge.sample_points:
                min_extend.x = min_extend.x if min_extend.x < sample.x else sample.x
                min_extend.y = min_extend.y if min_extend.y < sample.y else sample.y
                min_extend.z = min_extend.z if min_extend.z < sample.z else sample.z
                max_extend.x = max_extend.x if max_extend.x > sample.x else sample.x
                max_extend.y = max_extend.y if max_extend.y > sample.y else sample.y
                max_extend.z = max_extend.z if max_extend.z > sample.z else sample.z
        return [value for value in min_extend], [value for value in max_extend]

    @track_time
    def get_near_far_from_view(self, view: matrix44) -> Tuple[float, float]:
        far = 10000.0
        near = -10000.0
        for edge in self.edges:
            first = edge.sample_points[0]
            first_z = (view * Vector4([first.x, first.y, first.z, 1.0])).z
            last = edge.sample_points[-1]
            last_z = (view * Vector4([last.x, last.y, last.z, 1.0])).z
            if first_z < far:
                far = first_z
            else:
                if near < first_z:
                    near = first_z
            if last_z < far:
                far = last_z
            else:
                if near < last_z:
                    near = last_z
        if near > 0:
            near = 0
        return far, near


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

        self.point_render: RenderSet = RenderSet(sample_point_shader, VertexDataHandler(vbos=1))
        self.sphere_render: RenderSet = RenderSet(sample_sphere_shader, VertexDataHandler(vbos=1))
        self.transparent_render: RenderSet = RenderSet(sample_transparent_shader, VertexDataHandler(vbos=1))

    @track_time
    def render_point(self, window: Window):
        sampled_point_buffer = self.edge_handler.generate_buffer_data()
        sampled_points: int = self.edge_handler.get_buffer_points()

        self.point_render.set_uniform_data([("projection", window.cam.projection, "mat4"),
                                            ("view", window.cam.get_view_matrix(), "mat4"),
                                            ("point_color", [1.0, 0.0, 0.0], "vec3")])

        self.point_render.set()
        self.point_render.load_vbo_data(sampled_point_buffer)

        render_setting_0()
        glPointSize(10.0)
        glDrawArrays(GL_POINTS, 0, sampled_points)

    @track_time
    def render_sphere(self, window: Window):
        sampled_point_buffer = self.edge_handler.generate_buffer_data()
        sampled_points: int = self.edge_handler.get_buffer_points()

        self.sphere_render.set_uniform_data([("projection", window.cam.projection, "mat4"),
                                             ("view", window.cam.get_view_matrix(), "mat4")])

        self.sphere_render.set()
        self.sphere_render.load_vbo_data(sampled_point_buffer)

        render_setting_0()
        glDrawArrays(GL_POINTS, 0, sampled_points)

    @track_time
    def render_transparent(self, window: Window):
        sampled_point_buffer = self.edge_handler.generate_buffer_data()
        sampled_points: int = self.edge_handler.get_buffer_points()
        print("[%s] Rendering %d points." % (LOG_SOURCE, sampled_points))

        far, near = self.edge_handler.get_near_far_from_view(window.cam.get_view_matrix())
        self.transparent_render.set_uniform_data([("projection", window.cam.projection, "mat4"),
                                                  ("view", window.cam.get_view_matrix(), "mat4"),
                                                  ("farthest_point_view_z", far, "float"),
                                                  ("nearest_point_view_z", near, "float")])

        self.transparent_render.set()
        self.transparent_render.load_vbo_data(sampled_point_buffer)

        render_setting_1()
        glDrawArrays(GL_POINTS, 0, sampled_points)
