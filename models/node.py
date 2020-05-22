import math
import random

import numpy as np
from typing import List, Tuple

from pyrr import Vector3


class Node:
    def __init__(self, node_id: int, position: Vector3, input_edges: int = 0, output_edges: int = 0,
                 data: np.array = None):
        self.node_id: int = node_id
        self.position: Vector3 = position
        self.input_edges: int = input_edges
        self.output_edges: int = output_edges
        self.data: List[float] = [position.x, position.y, position.z, 1.0]
        if data is None:
            importance_sum: float = 0
            for _ in range(10):
                random_value: float = random.random()
                importance_sum += random_value
                self.data.append(random_value)
            self.data.append(importance_sum / 10.0)
            self.data.append(0.0)
        else:
            importance_sum: float = 0
            for d in data:
                importance_sum += d
                self.data.append(d)
            self.data.append(importance_sum / 10.0)
            self.data.append(0.0)

    def reset_position(self, position: Vector3):
        self.position: Vector3 = position
        self.data[0] = position.x
        self.data[1] = position.y
        self.data[2] = position.z


def create_nodes(layer_nodes: List[int], center_position: Vector3, z_range: Tuple[float, float], node_size: float,
                 layer_data: List[np.array] = None) -> List[List[Node]]:
    node_positions: List[List[Node]] = []
    for layer, nodes in enumerate(layer_nodes):
        input_edges: int = 0
        output_edges: int = 0
        if 0 < layer:
            input_edges = layer_nodes[layer - 1]
        if layer < len(layer_nodes) - 1:
            output_edges = layer_nodes[layer + 1]
        nodes_sqrt = math.ceil(math.sqrt(nodes))
        current_node_positions: List[Node] = []
        if nodes <= 1:
            position: Vector3 = Vector3(
                [center_position.x,
                 center_position.y,
                 z_range[0] * (1 - layer / (len(layer_nodes) - 1)) + z_range[1] * layer / (len(layer_nodes) - 1)])
            if layer_data is None:
                current_node_positions.append(Node(len(current_node_positions), position, input_edges, output_edges))
            else:
                current_node_positions.append(
                    Node(len(current_node_positions), position, input_edges, output_edges, layer_data[layer][0]))
        else:
            for i in range(nodes):
                pos_x: float = (i % nodes_sqrt) - (nodes_sqrt - 1.0) / 2.0
                pos_y: float = (math.floor(i / nodes_sqrt)) - (nodes_sqrt - 1.0) / 2.0
                position: Vector3 = Vector3(
                    [pos_x * node_size + center_position.x,
                     pos_y * node_size + center_position.y,
                     z_range[0] * (1 - layer / (len(layer_nodes) - 1)) + z_range[1] * layer / (len(layer_nodes) - 1)])
                if layer_data is None:
                    current_node_positions.append(
                        Node(len(current_node_positions), position, input_edges, output_edges))
                else:
                    current_node_positions.append(
                        Node(len(current_node_positions), position, input_edges, output_edges, layer_data[layer][i]))
        node_positions.append(current_node_positions)
    return node_positions
