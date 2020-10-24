import math
import numpy as np
from typing import List, Tuple

from pyrr import Vector3

from data.data_handler import ImportanceDataHandler, ProcessedNNHandler
from models.edge import Edge, split_edges_for_buffer, create_edges_processed, create_edges_random, \
    create_edges_importance
from models.node import Node, create_random_nodes, create_nodes_from_data, create_nodes_with_importance

LOG_SOURCE: str = "NETWORK_MODEL"


class NetworkModel:
    def __init__(self, layer: List[int], layer_width: float, layer_distance: float,
                 importance_data: ImportanceDataHandler = None, processed_nn: ProcessedNNHandler = None,
                 prune_percentage: float = 0.1):
        self.layer: List[int] = layer
        self.layer_width: float = layer_width
        self.layer_distance: float = layer_distance
        self.prune_percentage: float = prune_percentage

        self.bounding_volume: Tuple[Vector3, Vector3] = (
            Vector3(
                [-self.layer_width / 2.0, -self.layer_width / 2.0, -(len(self.layer) - 1) * self.layer_distance / 2.0]),
            Vector3(
                [self.layer_width / 2.0, self.layer_width / 2.0, (len(self.layer) - 1) * self.layer_distance / 2.0]))
        self.bounding_mid: Vector3 = (self.bounding_volume[1] + self.bounding_volume[0]) / 2.0
        self.bounding_range: Vector3 = (self.bounding_volume[1] - self.bounding_volume[0]) / 2.0
        self.bounding_range = Vector3(
            [abs(self.bounding_range.x), abs(self.bounding_range.y), abs(self.bounding_range.z)])

        self.layer_nodes: List[List[Node]] = []
        self.edge_data: np.array = []
        self.sample_data: np.array = []
        self.edge_importance_only: bool = False

        if importance_data is not None:
            self.layer_nodes: List[List[Node]] = create_nodes_with_importance(self.layer, self.bounding_mid,
                                                                              (self.bounding_volume[0].x,
                                                                               self.bounding_volume[1].x),
                                                                              (self.bounding_volume[0].y,
                                                                               self.bounding_volume[1].y),
                                                                              (self.bounding_volume[0].z,
                                                                               self.bounding_volume[1].z),
                                                                              importance_data.node_importance_data)
            self.edge_data = importance_data.edge_importance_data
            self.edge_importance_only = True
        elif processed_nn is not None:
            self.layer_nodes: List[List[Node]] = create_nodes_from_data(self.layer, processed_nn.node_data)
            self.edge_data = processed_nn.edge_data
            self.sample_data = processed_nn.sample_data
        else:
            self.layer_nodes: List[List[Node]] = create_random_nodes(self.layer, self.bounding_mid,
                                                                     (self.bounding_volume[0].x,
                                                                      self.bounding_volume[1].x),
                                                                     (self.bounding_volume[0].y,
                                                                      self.bounding_volume[1].y),
                                                                     (self.bounding_volume[0].z,
                                                                      self.bounding_volume[1].z))
        self.edge_count: int = 0
        for i in range(len(self.layer) - 1):
            self.edge_count += len(self.layer_nodes[i]) * len(self.layer_nodes[i + 1])
        self.pruned_edges: int = 0
        self.average_node_distance: float = layer_width * 0.75  # self.get_average_node_distance()
        self.average_edge_distance: float = layer_width * 0.5  # self.get_average_edge_distance()

        self.node_min_importance: float = self.read_node_min_importance()
        self.node_max_importance: float = self.read_node_max_importance()
        self.edge_min_importance: float = 0.0
        self.edge_max_importance: float = 1.0

    def get_nodes(self) -> List[Node]:
        node_data: List[Node] = []
        for layer in self.layer_nodes:
            for node in layer:
                node_data.append(node)
        return node_data

    def set_nodes(self, node_data: List[Node]):
        read_node_index: int = 0
        for i, layer in enumerate(self.layer_nodes):
            new_nodes = []
            for _ in layer:
                new_nodes.append(node_data[read_node_index])
                read_node_index += 1
            self.layer_nodes[i] = new_nodes

        self.edge_count = 0
        for i in range(len(self.layer) - 1):
            self.edge_count += len(self.layer_nodes[i]) * len(self.layer_nodes[i + 1])

        self.node_min_importance = self.read_node_min_importance()
        self.node_max_importance = self.read_node_max_importance()

    def create_edges(self) -> List[List[Edge]]:
        edges: List[List[Edge]] = []
        for i in range(len(self.layer) - 1):
            layer_edge: List[Edge] = []
            for node_one_i, node_one in enumerate(self.layer_nodes[i]):
                for node_two_i, node_two in enumerate(self.layer_nodes[i + 1]):
                    if len(self.edge_data) == 0:
                        new_edge: Edge = Edge().random_importance_init(node_one, node_two, i, node_one_i * len(
                            self.layer_nodes[i + 1]) + node_two_i)
                    else:
                        new_edge: Edge = Edge().importance_init(node_one, node_two, i, node_one_i * len(
                            self.layer_nodes[i + 1]) + node_two_i, self.edge_data[i][node_one_i][node_two_i])
                    layer_edge.append(new_edge)
            edges.append(layer_edge)
        return edges

    def set_edges(self) -> List[List[List[Edge]]]:
        self.edge_count = 0
        edges: List[List[List[Edge]]] = []
        for layer_edge_data, layer_sample_data in zip(self.edge_data, self.sample_data):
            layer_edges: List[List[Edge]] = []
            for container_edge_data, container_sample_data in zip(layer_edge_data, layer_sample_data):
                container_edges: List[Edge] = []
                for edge_data, sample_data in zip(container_edge_data, container_sample_data):
                    edge: Edge = Edge().data_init(edge_data, sample_data)
                    container_edges.append(edge)
                    if self.edge_min_importance > edge.data[3] * edge.data[6]:
                        self.edge_min_importance = edge.data[3] * edge.data[6]
                    if self.edge_max_importance < edge.data[3] * edge.data[6]:
                        self.edge_max_importance = edge.data[3] * edge.data[6]
                self.edge_count += len(container_edges)
                layer_edges.append(container_edges)
            edges.append(layer_edges)

        max_edge_count: int = 0
        for i in range(len(self.layer) - 1):
            max_edge_count += self.layer[i] * self.layer[i + 1]

        self.pruned_edges = max_edge_count - self.edge_count
        return edges

    def generate_filtered_edges(self, edge_container_size: int = 500) -> List[List[List[Edge]]]:
        self.pruned_edges = 0
        self.edge_min_importance = 10000.0
        self.edge_max_importance = 0.0

        edges: List[List[Edge]] = create_edges_random(self.layer_nodes) if len(self.edge_data) == 0 \
            else create_edges_importance(self.layer_nodes, self.edge_data) if self.edge_importance_only \
            else create_edges_processed(self.edge_data, self.sample_data)

        existing_edges: int = 0
        for layer_edge in edges:
            existing_edges += len(layer_edge)
        self.pruned_edges = self.edge_count - existing_edges

        edge_importance_values: List[float] = []
        for layer_edge in edges:
            for edge in layer_edge:
                edge_importance_values.append(edge.data[3] * edge.data[6])

        importance_prune_threshold: float = -1.0
        if self.prune_percentage > 0.0:
            sorted_importance_list = np.sort(np.array(edge_importance_values))
            lowest_importance: float = sorted_importance_list[0]
            highest_importance: float = sorted_importance_list[sorted_importance_list.shape[0] - 1]
            if not lowest_importance == highest_importance:
                importance_prune_threshold: float = sorted_importance_list[
                    int(len(edge_importance_values) * self.prune_percentage)]
            else:
                print("[NETWORK] Pruning ignored, because all importance values are equal.")

        filtered_edges: List[List[Edge]] = []
        for layer_edge in edges:
            filtered_layer_edges: List[Edge] = []
            for edge in layer_edge:
                if edge.data[3] * edge.data[6] > importance_prune_threshold:
                    if self.edge_min_importance > edge.data[3] * edge.data[6]:
                        self.edge_min_importance = edge.data[3] * edge.data[6]
                    if self.edge_max_importance < edge.data[3] * edge.data[6]:
                        self.edge_max_importance = edge.data[3] * edge.data[6]
                    filtered_layer_edges.append(edge)
                else:
                    self.pruned_edges += 1
            filtered_edges.append(filtered_layer_edges)
        return split_edges_for_buffer(filtered_edges, edge_container_size)

    def generate_max_distance(self) -> float:
        max_distance: float = 0.0
        for i in range(len(self.layer) - 1):
            for node_one in self.layer_nodes[i]:
                for node_two in self.layer_nodes[i + 1]:
                    distance: float = (node_one.position - node_two.position).length
                    if max_distance < distance:
                        max_distance = distance
        return max_distance

    def get_average_edge_distance(self) -> float:
        distance_sum: float = 0.0
        distance_value_count: int = 0
        edge_positions: List[List[Vector3]] = []
        for i in range(len(self.layer) - 1):
            layer_edge_position: List[Vector3] = []
            for node_one in self.layer_nodes[i]:
                for node_two in self.layer_nodes[i + 1]:
                    layer_edge_position.append((node_one.position + node_two.position) / 2.0)
            edge_positions.append(layer_edge_position)
            distance_value_count += (len(layer_edge_position) * (len(layer_edge_position) - 1))
        for i in range(len(self.layer) - 1):
            for edge_one in edge_positions[i]:
                for edge_two in edge_positions[i]:
                    distance_sum += math.sqrt(
                        (edge_one.x - edge_two.x) * (edge_one.x - edge_two.x)
                        + (edge_one.y - edge_two.y) * (edge_one.y - edge_two.y)
                        + (edge_one.z - edge_two.z) * (edge_one.z - edge_two.z))
        return distance_sum / distance_value_count

    def get_average_node_distance(self) -> float:
        distance_sum: float = 0.0
        distance_value_count: int = 0
        for i in range(len(self.layer)):
            distance_value_count += len(self.layer_nodes[i]) * (len(self.layer_nodes[i]) - 1)
        for i in range(len(self.layer)):
            layer_distance_sum: float = 0.0
            for node_one in self.layer_nodes[i]:
                for node_two in self.layer_nodes[i]:
                    layer_distance_sum += math.sqrt(
                        (node_one.position.x - node_two.position.x) * (node_one.position.x - node_two.position.x)
                        + (node_one.position.y - node_two.position.y) * (node_one.position.y - node_two.position.y)
                        + (node_one.position.z - node_two.position.z) * (node_one.position.z - node_two.position.z))
            distance_sum += layer_distance_sum / float(distance_value_count)
        return distance_sum

    def get_node_mid(self) -> Vector3:
        node_position_min_x: float = 0.0
        node_position_max_x: float = 0.0
        mid_position_x: float = 0.0
        node_position_min_y: float = 0.0
        node_position_max_y: float = 0.0
        mid_position_y: float = 0.0
        position_count: int = 0
        for i in range(len(self.layer)):
            position_count += len(self.layer_nodes[i])
        for i in range(len(self.layer)):
            for node_one in self.layer_nodes[i]:
                mid_position_x += node_one.position.x
                mid_position_y += node_one.position.y
                node_position_min_x = node_one.position.x if node_one.position.x < node_position_min_x else node_position_min_x
                node_position_min_y = node_one.position.y if node_one.position.y < node_position_min_y else node_position_min_y
                node_position_max_x = node_one.position.x if node_one.position.x > node_position_max_x else node_position_max_x
                node_position_max_y = node_one.position.y if node_one.position.y > node_position_max_y else node_position_max_y
        return Vector3(
            [(node_position_min_x + node_position_max_x) * 0.25 + 0.5 * mid_position_x / position_count,
             (node_position_min_y + node_position_max_y) * 0.25 + 0.5 * mid_position_y / position_count,
             0.0])

    def read_node_min_importance(self) -> float:
        min_importance: float = 10000.0
        for nodes in self.layer_nodes:
            for node in nodes:
                if node.data[14] < min_importance:
                    min_importance = node.data[14]
        return min_importance

    def read_node_max_importance(self) -> float:
        max_importance: float = 0.0
        for nodes in self.layer_nodes:
            for node in nodes:
                if node.data[14] > max_importance:
                    max_importance = node.data[14]
        return max_importance
