import math
import random

import numpy as np
from typing import List, Tuple

from pyrr import Vector3


class Node:
    def __init__(self, node_id: int, input_edges: int = 0, output_edges: int = 0):
        self.node_id: int = node_id
        self.input_edges: int = input_edges
        self.output_edges: int = output_edges
        self.position: Vector3 = Vector3([0.0, 0.0, 0.0])
        self.data: List[float] = []

    def data_init(self, data: np.array):
        self.data = []
        self.position = Vector3([data[0], data[1], data[2]])

        for d in data:
            self.data.append(d)

        return self

    def importance_init(self, position: Vector3, data: np.array):
        self.data = []
        if position is not None:
            self.position = position
            self.data = [position.x, position.y, position.z, 1.0]

        importance_sum: float = 0
        importance_squared_sum: float = 0
        for d in data:
            importance_sum += d
            importance_squared_sum += d * d
            self.data.append(d)
        self.data.append(importance_sum / 10.0)
        self.data.append(math.sqrt(importance_squared_sum))

        return self

    def random_importance_init(self, position: Vector3):
        self.position = position
        self.data = [position.x, position.y, position.z, 1.0]

        importance_sum: float = 0
        importance_squared_sum: float = 0
        position_max_0: int = int(random.random() * 10)
        for i in range(10):
            if i == position_max_0:
                random_value: float = 1.0  # random.random()
                importance_sum += random_value
                importance_squared_sum += random_value * random_value
                self.data.append(random_value)
            else:
                random_value: float = 0.0  # random.random() / 5.0
                importance_sum += random_value
                importance_squared_sum += random_value * random_value
                self.data.append(random_value)
        self.data.append(importance_sum / 10.0)
        self.data.append(math.sqrt(importance_squared_sum))
        return self

    def reset_position(self, position: Vector3):
        self.position: Vector3 = position
        self.data[0] = position.x
        self.data[1] = position.y
        self.data[2] = position.z


def create_random_nodes(layer_nodes: List[int],
                        center_position: Vector3,
                        x_range: Tuple[float, float],
                        y_range: Tuple[float, float],
                        z_range: Tuple[float, float],
                        node_size: float = None) -> List[List[Node]]:
    nodes: List[List[Node]] = []
    for layer, node_count in enumerate(layer_nodes):
        input_edges: int = 0
        output_edges: int = 0
        if 0 < layer:
            input_edges = layer_nodes[layer - 1]
        if layer < len(layer_nodes) - 1:
            output_edges = layer_nodes[layer + 1]
        nodes_sqrt = math.ceil(math.sqrt(node_count))

        current_layer_nodes: List[Node] = []
        if node_count <= 1 and type != "from_complete_data":
            position: Vector3 = Vector3(
                [center_position.x,
                 center_position.y,
                 z_range[0] * (1 - layer / (len(layer_nodes) - 1)) + z_range[1] * layer / (len(layer_nodes) - 1)])
            current_layer_nodes.append(
                Node(len(current_layer_nodes), input_edges, output_edges).random_importance_init(position))
        else:
            node_size_x: float = node_size
            node_size_y: float = node_size
            if node_size is None:
                node_size_x = abs(x_range[1] - x_range[0]) / nodes_sqrt
                node_size_y = abs(y_range[1] - y_range[0]) / nodes_sqrt
            for i in range(node_count):
                pos_x: float = (i % nodes_sqrt) - (nodes_sqrt - 1.0) / 2.0
                pos_y: float = (math.floor(i / nodes_sqrt)) - (nodes_sqrt - 1.0) / 2.0
                position: Vector3 = Vector3(
                    [pos_x * node_size_x + center_position.x,
                     pos_y * node_size_y + center_position.y,
                     z_range[0] * (1 - layer / (len(layer_nodes) - 1)) + z_range[1] * layer / (
                             len(layer_nodes) - 1)])
                current_layer_nodes.append(
                    Node(len(current_layer_nodes), input_edges, output_edges).random_importance_init(position))

        nodes.append(current_layer_nodes)
    return nodes


def create_nodes_with_importance(layer_nodes: List[int],
                                 center_position: Vector3,
                                 x_range: Tuple[float, float],
                                 y_range: Tuple[float, float],
                                 z_range: Tuple[float, float],
                                 node_importance_data: List[np.array],
                                 node_size: float = None) -> List[List[Node]]:
    nodes: List[List[Node]] = []
    for layer, node_count in enumerate(layer_nodes):
        input_edges: int = 0
        output_edges: int = 0
        if 0 < layer:
            input_edges = layer_nodes[layer - 1]
        if layer < len(layer_nodes) - 1:
            output_edges = layer_nodes[layer + 1]
        nodes_sqrt = math.ceil(math.sqrt(node_count))

        current_layer_nodes: List[Node] = []
        if node_count <= 1:
            position: Vector3 = Vector3(
                [center_position.x,
                 center_position.y,
                 z_range[0] * (1 - layer / (len(layer_nodes) - 1)) + z_range[1] * layer / (len(layer_nodes) - 1)])
            current_layer_nodes.append(
                Node(len(current_layer_nodes), input_edges, output_edges).importance_init(position,
                                                                                          node_importance_data[layer][
                                                                                              0]))
        else:
            node_size_x: float = node_size
            node_size_y: float = node_size
            if node_size is None:
                node_size_x = abs(x_range[1] - x_range[0]) / nodes_sqrt
                node_size_y = abs(y_range[1] - y_range[0]) / nodes_sqrt
            for i in range(node_count):
                pos_x: float = (i % nodes_sqrt) - (nodes_sqrt - 1.0) / 2.0
                pos_y: float = (math.floor(i / nodes_sqrt)) - (nodes_sqrt - 1.0) / 2.0
                position: Vector3 = Vector3(
                    [pos_x * node_size_x + center_position.x,
                     pos_y * node_size_y + center_position.y,
                     z_range[0] * (1 - layer / (len(layer_nodes) - 1)) + z_range[1] * layer / (
                             len(layer_nodes) - 1)])
                current_layer_nodes.append(
                    Node(len(current_layer_nodes), input_edges, output_edges).importance_init(position,
                                                                                              node_importance_data[
                                                                                                  layer][i]))
        nodes.append(current_layer_nodes)
    return nodes


def create_nodes_from_data(layer_nodes: List[int],
                           node_data: List[np.array] = None) -> List[List[Node]]:
    nodes: List[List[Node]] = []
    for layer, node_count in enumerate(layer_nodes):
        input_edges: int = 0
        output_edges: int = 0
        if 0 < layer:
            input_edges = layer_nodes[layer - 1]
        if layer < len(layer_nodes) - 1:
            output_edges = layer_nodes[layer + 1]
        current_layer_nodes: List[Node] = []

        for i in range(node_count):
            current_layer_nodes.append(
                Node(len(current_layer_nodes), input_edges, output_edges).data_init(node_data[layer][i]))
        nodes.append(current_layer_nodes)
    return nodes
