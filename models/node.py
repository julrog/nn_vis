import math
import random

import numpy as np
from typing import List, Tuple

from pyrr import Vector3


class Node:
    def __init__(self, position: Vector3, data: np.array = None):
        self.position: Vector3 = position
        self.data: List[float] = [position.x, position.y, position.z, 1.0]
        if data is None:
            for _ in range(12):
                self.data.append(random.random())
        else:
            for d in data:
                self.data.append(d)
            if len(data) % 4 != 0:
                for _ in range(len(data) % 4):
                    self.data.append(0.0)


def create_nodes(layer_nodes: List[int], center_position: Vector3, z_range: Tuple[float, float], node_size: float,
                 layer_data: List[np.array] = None) -> List[List[Node]]:
    node_positions: List[List[Node]] = []
    for layer, nodes in enumerate(layer_nodes):
        nodes_sqrt = math.ceil(math.sqrt(nodes))
        current_node_positions: List[Node] = []
        if nodes <= 1:
            position: Vector3 = Vector3(
                [center_position.x,
                 center_position.y,
                 z_range[0] * (1 - layer / (len(layer_nodes) - 1)) + z_range[1] * layer / (len(layer_nodes) - 1)])
            if layer_data is None:
                current_node_positions.append(Node(position))
            else:
                current_node_positions.append(Node(position, layer_data[layer][i]))
        else:
            for i in range(nodes):
                pos_x: float = (i % nodes_sqrt) - (nodes_sqrt - 1.0) / 2.0
                pos_y: float = (math.floor(i / nodes_sqrt)) - (nodes_sqrt - 1.0) / 2.0
                position: Vector3 = Vector3(
                    [pos_x * node_size + center_position.x,
                     pos_y * node_size + center_position.y,
                     z_range[0] * (1 - layer / (len(layer_nodes) - 1)) + z_range[1] * layer / (len(layer_nodes) - 1)])
                if layer_data is None:
                    current_node_positions.append(Node(position))
                else:
                    current_node_positions.append(Node(position, layer_data[layer][i]))
        node_positions.append(current_node_positions)
    return node_positions
