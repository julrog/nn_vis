import math
from typing import List, Tuple
from pyrr import Vector3


class NetworkModel:
    def __init__(self, layer: List[int], bounding_volume: Tuple[Vector3, Vector3]):
        self.layer: List[int] = layer
        self.bounding_volume: Tuple[Vector3, Vector3] = bounding_volume
        self.node_positions: List[List[Vector3]] = []
        for nodes in self.layer:
            nodes_sqrt = math.ceil(math.sqrt(nodes))
            current_node_positions: List[Vector3] = []
            for i in range(nodes):
                position: Vector3 = Vector3(
                    [,
                        ((math.floor(i / nodes_sqrt)) / nodes_sqrt) * 2.0 - 1.0,
                        self.bounding_volume[0].z * (1 - i / len(self.layer)) + self.bounding_volume[1].z * i])
                current_node_positions.append(position)
                layer_one.append(((math.floor(i / nodes_sqrt)) / nodes_sqrt) * 2.0 - 1.0)
                layer_one.append(-1.0)
            self.node_positions.append(current_node_positions)
