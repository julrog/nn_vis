import random
from typing import List

from pyrr import Vector4, Vector3

from models.node import Node


class Edge:
    def __init__(self, start_node: Node, end_node: Node, importance: float = None):
        self.start_node: Node = start_node
        self.end_node: Node = end_node
        self.start: Vector4 = Vector4([start_node.position.x, start_node.position.y, start_node.position.z, 1.0])
        self.end: Vector4 = Vector4([end_node.position.x, end_node.position.y, end_node.position.z, 0.0])
        self.initial_data: List[float] = [start_node.position.x, start_node.position.y, start_node.position.z, 1.0,
                                          end_node.position.x, end_node.position.y, end_node.position.z, 0.0]
        self.sample_points: List[Vector4] = [self.start, self.end]
        if importance is None:
            importance = 1.0  # random.random()
        self.data: List[float] = [2.0, start_node.output_edges, end_node.input_edges, importance, start_node.data[15],
                                  end_node.data[15], start_node.data[14], end_node.data[14]]
        self.data.extend(start_node.data[4:14])
        self.data.extend(end_node.data[4:14])
