import numpy as np
from typing import List
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
        importance: float = 1.0  # random.random()
        self.data = [2.0, layer_id, layer_edge_id, importance, start_node.data[15], end_node.data[15],
                     start_node.data[14], end_node.data[14]]
        self.data.extend(start_node.data[4:14])
        self.data.extend(end_node.data[4:14])
        self.sample_data = [start_node.position.x, start_node.position.y, start_node.position.z, 1.0,
                            end_node.position.x, end_node.position.y, end_node.position.z, 0.0]
        return self


def split_edges_for_buffer(edges: List[List[Edge]], edge_container_size: int = 1000) -> List[List[List[Edge]]]:
    split_edges: List[List[List[Edge]]] = []
    for layer_edges in edges:
        split_layer_edge_container: List[List[Edge]] = []
        if len(layer_edges) > edge_container_size:
            current_container: List[Edge] = []
            current_container_edge_count: int = 0
            for edge in layer_edges:
                current_container_edge_count += 1
                if current_container_edge_count > edge_container_size:
                    current_container_edge_count = 1
                    split_layer_edge_container.append(current_container)
                    current_container = []
                else:
                    current_container.append(edge)
            split_layer_edge_container.append(current_container)
        else:
            split_layer_edge_container.append(layer_edges)
        split_edges.append(split_layer_edge_container)

    return split_edges


def create_edges_random(layer_nodes: List[List[Node]]) -> List[List[Edge]]:
    edges: List[List[Edge]] = []
    for i in range(len(layer_nodes) - 1):
        layer_edges: List[Edge] = []
        for node_one_i, node_one in enumerate(layer_nodes[i]):
            for node_two_i, node_two in enumerate(layer_nodes[i + 1]):
                new_edge: Edge = Edge().random_importance_init(node_one, node_two, i, node_one_i * len(
                    layer_nodes[i + 1]) + node_two_i)
                layer_edges.append(new_edge)
        edges.append(layer_edges)
    return edges


def create_edges_importance(layer_nodes: List[List[Node]], edge_data: np.array) -> List[List[Edge]]:
    edges: List[List[Edge]] = []
    for i in range(len(layer_nodes) - 1):
        layer_edges: List[Edge] = []
        for node_one_i, node_one in enumerate(layer_nodes[i]):
            for node_two_i, node_two in enumerate(layer_nodes[i + 1]):
                new_edge: Edge = Edge().importance_init(node_one, node_two, i, node_one_i * len(
                    layer_nodes[i + 1]) + node_two_i, edge_data[i][node_one_i][node_two_i])
                layer_edges.append(new_edge)
        edges.append(layer_edges)
    return edges


def create_edges_processed(edge_data: np.array, sample_data: np.array) -> List[List[Edge]]:
    edges: List[List[Edge]] = []
    for layer_edge_data, layer_sample_data in zip(edge_data, sample_data):
        layer_edges: List[Edge] = []
        for container_edge_data, container_sample_data in zip(layer_edge_data, layer_sample_data):
            for edge_data, sample_data in zip(container_edge_data, container_sample_data):
                edge: Edge = Edge().data_init(edge_data, sample_data)
                layer_edges.append(edge)
        edges.append(layer_edges)
    return edges


'''def create_edges_processed(edge_data: np.array, sample_data: np.array) -> Tuple[
    List[List[List[Edge]]], int, float, float]:
    edge_count: int = 0
    edge_min_importance: float = 10000.0
    edge_max_importance: float = 0.0

    edges: List[List[List[Edge]]] = []
    for layer_edge_data, layer_sample_data in zip(self.edge_data, self.sample_data):
        layer_edges: List[List[Edge]] = []
        for container_edge_data, container_sample_data in zip(layer_edge_data, layer_sample_data):
            container_edges: List[Edge] = []
            for edge_data, sample_data in zip(container_edge_data, container_sample_data):
                edge: Edge = Edge().data_init(edge_data, sample_data)
                container_edges.append(edge)
                if edge_min_importance > edge.data[3] * edge.data[6]:
                    edge_min_importance = edge.data[3] * edge.data[6]
                if edge_max_importance < edge.data[3] * edge.data[6]:
                    edge_max_importance = edge.data[3] * edge.data[6]
            edge_count += len(container_edges)
            layer_edges.append(container_edges)
        edges.append(layer_edges)
    return edges, edge_count, edge_min_importance, edge_max_importance'''
