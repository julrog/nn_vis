from random import random
from typing import List, Tuple
from pyrr import Vector3, matrix44, Vector4
import numpy as np
from OpenGL.GL import *

from compute_shader import ComputeShader, ComputeShaderHandler
from render_helper import RenderSet, VertexDataHandler, render_setting_0, render_setting_1
from shader import RenderShader, RenderShaderHandler
from texture import Texture
from window import WindowHandler, Window

LOG_SOURCE: str = "EDGE"


class Edge:
    def __init__(self, start_position: Vector3, end_position: Vector3):
        self.start: Vector3 = start_position
        self.end: Vector3 = end_position
        self.sample_points: List[Vector3] = [self.start, self.end]
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
                self.sample_points.append(Vector3([x, y, z]))
            else:
                self.sample_points.append(Vector3([x, y, z]))
                return

    def sample(self, sample_length: float):
        self.sample_length = sample_length
        self.squared_sample_length = self.sample_length * self.sample_length
        self.sample_points = []
        divisions = int(round((self.end - self.start).length / self.sample_length))
        for i in range(divisions):
            self.sample_points.append(
                Vector3(self.start * (1.0 - i / (divisions - 1.0)) + self.end * (i / (divisions - 1.0))))

    def sample_noise(self, strength: float = 1.0):
        if not self.sample_points:
            return
        noise_strength = strength * self.sample_length
        for i in range(len(self.sample_points)):
            # if i is not 0 and i is not len(self.sample_points) - 1:
            self.sample_points[i] = self.sample_points[i] + Vector3(
                [random() * noise_strength * 2.0 - noise_strength,
                 random() * noise_strength * 2.0 - noise_strength,
                 0.0])

    def resample(self, sample_length: float = None):
        if not self.sample_points:
            raise Exception("[%s] No samples set yet, can't resample." % LOG_SOURCE)

        if sample_length:
            self.sample_length = sample_length

        new_sample_points = []
        previous_point = None
        for i in range(len(self.sample_points)):
            if i == 0:
                new_sample_points.append(self.sample_points[i])
                previous_point = self.sample_points[i]
            else:
                new_point_added = False
                while (self.sample_points[i] - new_sample_points[-1]).length >= self.sample_length * 0.99:
                    prev_distance = (previous_point - new_sample_points[-1]).length
                    current_distance = (self.sample_points[i] - new_sample_points[-1]).length
                    t = (self.sample_length - prev_distance) / (current_distance - prev_distance)
                    np.clip(t, 0.0, 1.0)
                    new_sample_points.append(previous_point * (1.0 - t) + self.sample_points[i] * t)
                    previous_point = new_sample_points[-1]
                    new_point_added = True
                if not new_point_added:
                    previous_point = self.sample_points[i]
        new_sample_points.append(self.sample_points[-1])
        self.sample_points = new_sample_points

    def get_edge_length(self) -> float:
        if not self.sample_points:
            raise Exception("[EDGE] Not yet sampled, can't get length of edge!")
        length = 0.0
        previous_point = None
        for point in self.sample_points:
            if previous_point:
                distance_vector: Vector3 = point - previous_point
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
            self.edge_sample_texture_read: Texture or None = None
            self.edge_sample_texture_write: Texture or None = None
            self.max_distance: float = 0
            self.max_sample_points: int = 0
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

        for edge in self.edges:
            point_data = []
            point_data.extend(edge.start.data)
            point_data.append(1.0)
            point_data.extend(edge.end.data)

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
                point_data.append(1.0)
                point_data.extend(edge.end.data)
                point_data.append(0.0)
                initial_data.extend(point_data)
                initial_data.extend([0] * (self.max_sample_points * 4 - len(point_data)))

            self.edge_sample_texture_read = Texture(self.max_sample_points, len(self.edges))
            self.edge_sample_texture_write = Texture(self.max_sample_points, len(self.edges))
            self.edge_sample_texture_read.setup(0, initial_data)
            self.edge_sample_texture_write.setup(0, initial_data)

    def sample_edges(self):
        if self.use_compute:
            self.sample_compute_shader.set_textures(
                [(self.edge_sample_texture_read, "read", 0), (self.edge_sample_texture_write, "write", 1)])
            self.sample_compute_shader.set_uniform_data([('sample_length', self.sample_length, 'float')])
            self.sample_compute_shader.use(len(self.edges))  # use dynamic workgroup sizes
            self.edge_sample_texture_read, self.edge_sample_texture_write = self.edge_sample_texture_write, self.edge_sample_texture_read
        else:
            for edge in self.edges:
                edge.sample(self.sample_length)
        self.sampled = True

    def generate_buffer_data(self):
        buffer_data: List[Vector3] = []
        if not self.sampled:
            raise Exception("[%s] Not sampled, can't generate data!" % LOG_SOURCE)

        if self.use_compute:
            self.read_samples_from_texture()

        for edge in self.edges:
            for point in edge.sample_points:
                buffer_data.extend(point)

        return np.array(buffer_data, dtype=np.float32)

    def get_points(self):
        return sum([len(edge.sample_points) for edge in self.edges])

    def sample_noise(self, strength: float = 1.0):
        if self.use_compute:
            self.noise_compute_shader.set_textures(
                [(self.edge_sample_texture_read, "read", 0), (self.edge_sample_texture_write, "write", 1)])
            self.noise_compute_shader.set_uniform_data([('sample_length', self.sample_length, 'float')])
            self.noise_compute_shader.set_uniform_data([('noise_strength', strength, 'float')])
            self.noise_compute_shader.use(len(self.edges))  # use dynamic workgroup sizes

            # switch textures, because the written textures resembles now the current edge samples
            self.edge_sample_texture_read, self.edge_sample_texture_write = self.edge_sample_texture_write, self.edge_sample_texture_read

            self.read_samples_from_texture()
        else:
            for edge in self.edges:
                edge.sample_noise(strength)

    def resample(self, sample_length: float = None):
        if sample_length is not None:
            self.sample_length = sample_length
        if self.use_compute:
            self.sample_edges()
        else:
            for edge in self.edges:
                edge.resample(self.sample_length)

    def read_samples_from_texture(self):
        edge_sample_data = self.edge_sample_texture_read.read()
        edge_sample_data = edge_sample_data.flatten()
        for i in range(len(self.edges)):
            start_split = (i * self.max_sample_points * 4) if i is not 0 else None
            end_split = ((i + 1) * self.max_sample_points * 4 + 1) if i is not len(self.edges) - 1 else None
            split_data = edge_sample_data[start_split: end_split]
            self.edges[i].set_sample_points(self.sample_length, split_data)

    def get_extends(self) -> Tuple[Vector3, Vector3]:
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
        return min_extend, max_extend

    def get_near_far_from_view(self, view: matrix44) -> Tuple[float, float]:
        far = 10000.0
        near = -10000.0
        for edge in self.edges:
            for sample in edge.sample_points:
                view_pos = view * Vector4([sample.x, sample.y, sample.z, 1.0])
                far = view_pos.z if view_pos.z < far else far
                near = view_pos.z if near < view_pos.z < 0 else near
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

        self.point_render: RenderSet = RenderSet(sample_point_shader, VertexDataHandler())
        self.sphere_render: RenderSet = RenderSet(sample_sphere_shader, VertexDataHandler())
        self.transparent_render: RenderSet = RenderSet(sample_transparent_shader, VertexDataHandler())

    def render_point(self, window: Window):
        sampled_point_buffer = self.edge_handler.generate_buffer_data()
        sampled_points = self.edge_handler.get_points()

        self.point_render.set_uniform_data([("projection", window.cam.projection, "mat4"),
                                            ("view", window.cam.get_view_matrix(), "mat4"),
                                            ("point_color", [1.0, 0.0, 0.0], "vec3")])

        self.point_render.set()
        self.point_render.load_data(sampled_point_buffer)

        render_setting_0()
        glPointSize(10.0)
        glDrawArrays(GL_POINTS, 0, sampled_points)

    def render_sphere(self, window: Window):
        sampled_point_buffer = self.edge_handler.generate_buffer_data()
        sampled_points = self.edge_handler.get_points()

        self.sphere_render.set_uniform_data([("projection", window.cam.projection, "mat4"),
                                             ("view", window.cam.get_view_matrix(), "mat4")])

        self.sphere_render.set()
        self.sphere_render.load_data(sampled_point_buffer)

        render_setting_0()
        glDrawArrays(GL_POINTS, 0, sampled_points)

    def render_transparent(self, window: Window):
        sampled_point_buffer = self.edge_handler.generate_buffer_data()
        sampled_points = self.edge_handler.get_points()

        far, near = self.edge_handler.get_near_far_from_view(window.cam.get_view_matrix())
        self.transparent_render.set_uniform_data([("projection", window.cam.projection, "mat4"),
                                                  ("view", window.cam.get_view_matrix(), "mat4"),
                                                  ("farthest_point_view_z", far, "float"),
                                                  ("nearest_point_view_z", near, "float")])

        self.transparent_render.set()
        self.transparent_render.load_data(sampled_point_buffer)

        render_setting_1()
        glDrawArrays(GL_POINTS, 0, sampled_points)
