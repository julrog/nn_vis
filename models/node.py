import math
from typing import List, Tuple

from pyrr import Vector3


class Node:
    def __init__(self, position: Vector3):
        self.position: Vector3 = position
        self.position_data: List[float] = [position.x, position.y, position.z, 1.0]


def create_nodes(layer_data: List[int], center_position: Vector3, z_range: Tuple[float, float], node_size: float) -> \
        List[List[Node]]:
    node_positions: List[List[Node]] = []
    for layer, nodes in enumerate(layer_data):
        nodes_sqrt = math.ceil(math.sqrt(nodes))
        current_node_positions: List[Node] = []
        if nodes <= 1:
            position: Vector3 = Vector3(
                [center_position.x,
                 center_position.y,
                 z_range[0] * (1 - layer / (len(layer_data) - 1)) + z_range[1] * layer / (len(layer_data) - 1)])
            current_node_positions.append(Node(position))
        else:
            for i in range(nodes):
                pos_x: float = (i % nodes_sqrt) - (nodes_sqrt - 1.0) / 2.0
                pos_y: float = (math.floor(i / nodes_sqrt)) - (nodes_sqrt - 1.0) / 2.0
                position: Vector3 = Vector3(
                    [pos_x * node_size + center_position.x,
                     pos_y * node_size + center_position.y,
                     z_range[0] * (1 - layer / (len(layer_data) - 1)) + z_range[1] * layer / (len(layer_data) - 1)])
                current_node_positions.append(Node(position))
        node_positions.append(current_node_positions)
    return node_positions
