from random import random
from typing import List
from pyrr import Vector3
import numpy as np

from compute_shader import ComputeShader, ComputeShaderHandler
from texture import Texture


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
            return

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
            self.edge_sample_texture_read = Texture(self.max_sample_points, len(self.edges))
            self.edge_sample_texture_write = Texture(self.max_sample_points, len(self.edges))
            initial_data: List[float] = []
            for edge in self.edges:
                point_data = []
                point_data.extend(edge.start.data)
                point_data.append(1.0)
                point_data.extend(edge.end.data)
                point_data.append(0.0)
                initial_data.extend(point_data)
                initial_data.extend([0] * (self.max_sample_points * 4 - len(point_data)))
            self.edge_sample_texture_read.setup(initial_data)
            self.edge_sample_texture_write.setup(initial_data)

    def sample_edges(self):
        if self.use_compute:
            self.sample_compute_shader.set_textures(
                [(self.edge_sample_texture_read, "read"), (self.edge_sample_texture_write, "write")])
            self.sample_compute_shader.set_uniform_data([('sample_length', self.sample_length, 'float')])
            self.sample_compute_shader.use(len(self.edges))
            self.edge_sample_texture_write.bind("read")
            edge_sample_data = self.edge_sample_texture_write.read()
            edge_sample_data = edge_sample_data.flatten()

            for i in range(len(self.edges)):
                start_split = i * self.max_sample_points * 4 if i is not 0 else 0
                end_split = ((i + 1) * self.max_sample_points * 4) - 0 if i is not len(self.edges) - 1 else None
                split_data = edge_sample_data[start_split: end_split]
                self.edges[i].set_sample_points(self.sample_length, split_data)
            self.edge_sample_texture_read.deactivate()  # otherwise the binding positions are wrong next time TODO better solution
            self.edge_sample_texture_write.deactivate()  # otherwise the binding positions are wrong next time TODO better solution
            self.edge_sample_texture_read, self.edge_sample_texture_write = self.edge_sample_texture_write, self.edge_sample_texture_read
        else:
            for edge in self.edges:
                edge.sample(self.sample_length)
        self.sampled = True

    def generate_buffer_data(self):
        buffer_data: List[Vector3] = []
        if not self.sampled:
            pass  # self.sample_edges()

        for edge in self.edges:
            for point in edge.sample_points:
                buffer_data.extend(point)

        return np.array(buffer_data, dtype=np.float32)

    def get_points(self):
        return sum([len(edge.sample_points) for edge in self.edges])

    def sample_noise(self, strength: float = 1.0):
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
