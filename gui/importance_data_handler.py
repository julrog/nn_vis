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
