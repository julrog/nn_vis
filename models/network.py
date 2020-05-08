import math
from typing import List, Tuple

from pyrr import Vector3

from models.edge import Edge


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
                    if i % 2 == 0:
                        edges.append(Edge(node_one, node_two))
                    else:
                        edges.append(Edge(node_two, node_one))
        return edges

    def generate_edges_special(self) -> List[Edge]:
        edges: List[Edge] = []
        for i in range(len(self.layer) - 1):
            for i_one, node_one in enumerate(self.node_positions[i]):
                for i_two, node_two in enumerate(self.node_positions[i + 1]):
                    if ((node_one + node_two) / 2.0).y != self.bounding_mid.y:
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
