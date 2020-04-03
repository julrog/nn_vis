from random import random
from typing import List
from pyrr import Vector3
import numpy as np


class Edge:
    def __init__(self, start_position: Vector3, end_position: Vector3):
        self.start: Vector3 = start_position
        self.end: Vector3 = end_position
        self.sample_points: List[Vector3] | None = None
        self.sample_length: float = 0.0
        self.squared_sample_length: float = 0.0

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
    def __init__(self, sample_length: float):
        self.edges: List[Edge] = []
        self.sampled: bool = False
        self.sample_length: float = sample_length

    def set_data(self, node_positions_layer_one: List[float], node_positions_layer_two: List[float]):
        def group_wise(it):
            it = iter(it)
            while True:
                yield next(it), next(it), next(it)

        vec_nodes_layer_one = [Vector3([x, y, z]) for x, y, z in group_wise(node_positions_layer_one)]
        vec_nodes_layer_two = [Vector3([x, y, z]) for x, y, z in group_wise(node_positions_layer_two)]
        self.edges = []
        for node_one in vec_nodes_layer_one:
            for node_two in vec_nodes_layer_two:
                self.edges.append(Edge(node_one, node_two))

    def sample_edges(self):
        for edge in self.edges:
            edge.sample(self.sample_length)
        self.sampled = True

    def generate_buffer_data(self):
        buffer_data: List[Vector3] = []
        if not self.sampled:
            self.sample_edges()

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
        for edge in self.edges:
            edge.resample(sample_length)
