import random
import numpy as np
from typing import List

from pyrr import Vector4

from definitions import vec4wise
from models.node import Node


class Edge:
    def __init__(self):
        self.data: List[float] = []
        self.sample_data: List[float] = []

    def data_init(self, data: np.array, sample_data: np.array):
        self.data = []
        for d in data:
            self.data.append(d)
        self.sample_data = []
        for sd in sample_data:
            self.sample_data.append(sd)
        return self

    def importance_init(self, start_node: Node, end_node: Node, layer_id: int, layer_edge_id: int, importance: float):
        self.data = []
        self.data = [2.0, layer_id, layer_edge_id, importance, start_node.data[15], end_node.data[15],
                     start_node.data[14], end_node.data[14]]
        self.data.extend(start_node.data[4:14])
        self.data.extend(end_node.data[4:14])
        self.sample_data = [start_node.position.x, start_node.position.y, start_node.position.z, 1.0,
                            end_node.position.x, end_node.position.y, end_node.position.z, 0.0]
        return self

    def random_importance_init(self, start_node: Node, end_node: Node, layer_id: int, layer_edge_id: int):
        importance: float = random.random()
        self.data = [2.0, layer_id, layer_edge_id, importance, start_node.data[15], end_node.data[15],
                     start_node.data[14], end_node.data[14]]
        self.data.extend(start_node.data[4:14])
        self.data.extend(end_node.data[4:14])
        self.sample_data = [start_node.position.x, start_node.position.y, start_node.position.z, 1.0,
                            end_node.position.x, end_node.position.y, end_node.position.z, 0.0]
        return self
