import math
from typing import List, Tuple
from pyrr import Vector3, Vector4, Matrix44, vector4, matrix44

from utility.performance import track_time


class Grid:
    def __init__(self, grid_cell_size: Vector3, bounding_volume: Tuple[Vector3, Vector3]):
        self.grid_cell_size: Vector3 = grid_cell_size

        self.bounding_volume: Tuple[Vector3, Vector3] = bounding_volume
        if self.bounding_volume[0].x > self.bounding_volume[1].x:
            self.bounding_volume[0].x, self.bounding_volume[1].x = bounding_volume[1].x, bounding_volume[0].x
        if self.bounding_volume[0].y > self.bounding_volume[1].y:
            self.bounding_volume[0].y, self.bounding_volume[1].y = bounding_volume[1].y, bounding_volume[0].y
        if self.bounding_volume[0].z > self.bounding_volume[1].z:
            self.bounding_volume[0].z, self.bounding_volume[1].z = bounding_volume[1].z, bounding_volume[0].z

        self.grid_cell_count: List[int] = [
            int((self.bounding_volume[1].x - self.bounding_volume[0].x) / self.grid_cell_size.x) + 2,
            int((self.bounding_volume[1].y - self.bounding_volume[0].y) / self.grid_cell_size.y) + 2,
            int((self.bounding_volume[1].z - self.bounding_volume[0].z) / self.grid_cell_size.z) + 2]
        self.grid_cell_count_overall: int = self.grid_cell_count[0] * self.grid_cell_count[1] * self.grid_cell_count[2]

        self.extends: List[Vector4] = [vector4.create_from_vector3(self.bounding_volume[0], 1.0),
                                       vector4.create_from_vector3(self.bounding_volume[1], 1.0)]
        self.extends.extend([
            Vector4([self.bounding_volume[1].x, self.bounding_volume[1].y, self.bounding_volume[0].z, 1.0]),
            Vector4([self.bounding_volume[1].x, self.bounding_volume[0].y, self.bounding_volume[0].z, 1.0]),
            Vector4([self.bounding_volume[0].x, self.bounding_volume[1].y, self.bounding_volume[0].z, 1.0]),
            Vector4([self.bounding_volume[1].x, self.bounding_volume[1].y, self.bounding_volume[1].z, 1.0]),
            Vector4([self.bounding_volume[1].x, self.bounding_volume[0].y, self.bounding_volume[1].z, 1.0]),
            Vector4([self.bounding_volume[0].x, self.bounding_volume[1].y, self.bounding_volume[1].z, 1.0])
        ])

    @track_time
    def get_near_far_from_view(self, view: Matrix44) -> Tuple[float, float]:
        nearest_view_z: float = -1000000
        farthest_view_z: float = 1000000

        for pos in self.extends:
            view_position = Vector4(matrix44.apply_to_vector(view, pos))
            if view_position.z > nearest_view_z:
                nearest_view_z = view_position.z
            if view_position.z < farthest_view_z:
                farthest_view_z = view_position.z

        return nearest_view_z, farthest_view_z


class Edge:
    def __init__(self, start_position: Vector3, end_position: Vector3):
        self.start: Vector4 = Vector4([start_position.x, start_position.y, start_position.z, 1.0])
        self.end: Vector4 = Vector4([end_position.x, end_position.y, end_position.z, 0.0])
        self.initial_data: List[float] = [start_position.x, start_position.y, start_position.z, 1.0, end_position.x,
                                          end_position.y, end_position.z, 0.0]
        self.sample_points: List[Vector4] = [self.start, self.end]


class NetworkModel:
    def __init__(self, layer: List[int], bounding_volume: Tuple[Vector3, Vector3]):
        self.layer: List[int] = layer
        self.node_positions: List[List[Vector3]] = []

        self.bounding_volume: Tuple[Vector3, Vector3] = bounding_volume
        self.bounding_mid: Vector3 = (self.bounding_volume[1] + self.bounding_volume[0]) / 2.0
        self.bounding_range: Vector3 = (self.bounding_volume[1] - self.bounding_volume[0]) / 2.0
        self.bounding_range = Vector3(
            [abs(self.bounding_range.x), abs(self.bounding_range.y), abs(self.bounding_range.z)])

        for layer, nodes in enumerate(self.layer):
            nodes_sqrt = math.ceil(math.sqrt(nodes))
            current_node_positions: List[Vector3] = []
            if nodes <= 1:
                position: Vector3 = Vector3(
                    [self.bounding_mid.x,
                     self.bounding_mid.y,
                     self.bounding_volume[0].z * (1 - layer / (len(self.layer) - 1)) + self.bounding_volume[
                         1].z * layer / (len(self.layer) - 1)])
                current_node_positions.append(position)
            else:
                for i in range(nodes):
                    position: Vector3 = Vector3(
                        [(((i % nodes_sqrt) / (
                                nodes_sqrt - 1.0)) * 2.0 - 1.0) * self.bounding_range.x + self.bounding_mid.x,
                         ((math.floor(
                             i / nodes_sqrt) / (
                                   nodes_sqrt - 1.0)) * 2.0 - 1.0) * self.bounding_range.y + self.bounding_mid.y,
                         self.bounding_volume[0].z * (1 - layer / (len(self.layer) - 1)) + self.bounding_volume[
                             1].z * layer / (len(self.layer) - 1)])
                    current_node_positions.append(position)
            self.node_positions.append(current_node_positions)

    def generate_edges(self) -> List[Edge]:
        edges: List[Edge] = []
        for i in range(len(self.layer) - 1):
            for node_one in self.node_positions[i]:
                for node_two in self.node_positions[i + 1]:
                    edges.append(Edge(node_one, node_two))
        return edges

    def generate_max_distance(self) -> float:
        max_distance: float = 0.0
        for i in range(len(self.layer) - 1):
            for node_one in self.node_positions[i]:
                for node_two in self.node_positions[i + 1]:
                    distance: float = (node_one - node_two).length
                    if max_distance < distance:
                        max_distance = distance
        return max_distance
