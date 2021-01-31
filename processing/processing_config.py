from typing import Dict, List, Tuple

from utility.config import BaseConfig


class ProcessingConfig(BaseConfig):
    def __init__(self, name: str = None):
        if name is None:
            super().__init__("processing")
        else:
            super().__init__("processing", name)

        self.label: Dict[str, str] = dict()
        self.value_type: Dict[str, str] = dict()
        self.set_defaults()

    def set_defaults(self):
        setting_items: List[Tuple[str, str, str, any]] = []
        setting_items.extend([("layer_distance", "Layer distance", "float", 1.0),
                              ("layer_width", "Layer width", "float", 1.0),
                              ("sampling_rate", "Sampling rate", "float", 10.0),
                              ("prune_percentage", "Prune percentage", "float", 0.0),
                              ("node_bandwidth_reduction", "Node bandwidth reduction", "float", 0.95),
                              ("edge_bandwidth_reduction", "Edge bandwidth reduction", "float", 0.90),
                              ("edge_importance_type", "Edge importance type", "int", 0)])

        for key, label, valueType, value in setting_items:
            self.label[key] = label
            self.value_type[key] = valueType
            self.setdefault(key, value)

        phase_setting_items: List[Tuple[str, any]] = []
        phase_setting_items.extend([("smoothing", True),
                                    ("smoothing_iterations", 8)])

        for key, value in phase_setting_items:
            self.setdefault(key, value)
