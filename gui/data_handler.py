from typing import List

import numpy as np


class ImportanceDataHandler:
    def __init__(self, path: str):
        node_importance_data, edge_importance_data = np.load(path, allow_pickle=True)['arr_0']
        self.node_importance_data: List[List[float]] = node_importance_data
        self.edge_importance_data: List[List[List[float]]] = edge_importance_data
        self.layer_data: List[int] = []
        for layer_nodes in self.node_importance_data:
            self.layer_data.append(len(layer_nodes))


class ProcessedNNHandler:
    def __init__(self, path: str):
        layer_data, node_data, edge_data, sample_data = np.load(path, allow_pickle=True)['arr_0']
        self.layer_data: List[int] = layer_data

        self.node_data: List[np.array] = []
        raw_node_data: np.array = np.array(node_data).reshape(-1, 16)
        node_data_offset: int = 0
        for i, nodes in enumerate(self.layer_data):
            self.node_data.append(raw_node_data[node_data_offset:(node_data_offset + nodes)])
            node_data_offset += nodes

        self.edge_data: List[List[List[float]]] = edge_data
        self.sample_data: List[List[float]] = sample_data
