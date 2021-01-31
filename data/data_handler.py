from typing import List

import numpy as np

from definitions import ADDITIONAL_NODE_BUFFER_DATA, ADDITIONAL_EDGE_BUFFER_DATA
from opengl_helper.buffer import get_buffer_object_size


class ImportanceDataHandler:
    def __init__(self, path: str):
        node_importance_data, edge_importance_data = np.load(path, allow_pickle=True)["arr_0"]
        self.node_importance_data: List[List[List[float]]] = node_importance_data
        self.edge_importance_data: List[List[List[float]]] = edge_importance_data
        self.layer_data: List[int] = []
        for layer_nodes in self.node_importance_data:
            self.layer_data.append(len(layer_nodes))


class ProcessedNNHandler:
    def __init__(self, path: str):
        layer_data, node_data, edge_data, sample_data, max_sample_points = np.load(path, allow_pickle=True)["arr_0"]
        self.layer_data: List[int] = layer_data
        num_classes: int = layer_data[len(layer_data) - 1]

        self.node_data: List[np.array] = []
        raw_node_data: np.array = np.array(node_data).reshape(
            -1, get_buffer_object_size(num_classes, ADDITIONAL_NODE_BUFFER_DATA))
        node_data_offset: int = 0
        for i, nodes in enumerate(self.layer_data):
            self.node_data.append(raw_node_data[node_data_offset:(node_data_offset + nodes)])
            node_data_offset += nodes

        self.edge_data: List[List[np.array]] = edge_data
        for i, layer_edge_data in enumerate(self.edge_data):
            for j, container_edge_data in enumerate(layer_edge_data):
                self.edge_data[i][j] = container_edge_data.reshape(
                    -1, get_buffer_object_size(num_classes * 2, ADDITIONAL_EDGE_BUFFER_DATA))

        self.sample_data: np.array = sample_data
        for i, layer_sample_data in enumerate(self.sample_data):
            for j, container_sample_data in enumerate(layer_sample_data):
                self.sample_data[i][j] = container_sample_data.reshape(-1, max_sample_points * 4)

    def get_all_samples(self) -> np.array:
        samples: np.array = np.array([])
        for layer_edges in self.sample_data:
            for container_edges in layer_edges:
                for edge_samples in container_edges:
                    samples = np.append(samples, edge_samples[:int(edge_samples[3] * 4)])
        return samples.reshape(-1, 4)
