from typing import Any, Dict, List, Optional, Tuple

from utility.config import BaseConfig


class ProcessingConfig(BaseConfig):
    def __init__(self, name: Optional[str] = None) -> None:
        if name is None:
            super().__init__('processing')
        else:
            super().__init__('processing', name)

        self.label: Dict[str, str] = dict()
        self.value_type: Dict[str, str] = dict()
        self.set_defaults()

    def set_defaults(self) -> None:
        setting_items: List[Tuple[str, str, str, Any]] = []
        setting_items.extend([('layer_distance', 'Layer distance', 'float', 0.5),
                              ('layer_width', 'Layer width', 'float', 1.0),
                              ('sampling_rate', 'Sampling rate', 'float', 15.0),
                              ('prune_percentage', 'Prune percentage', 'float', 0.0),
                              ('node_bandwidth_reduction',
                               'Node bandwidth reduction', 'float', 0.95),
                              ('edge_bandwidth_reduction',
                               'Edge bandwidth reduction', 'float', 0.90),
                              ('edge_importance_type', 'Edge importance type', 'int', 0)])

        for key, label, valueType, value in setting_items:
            self.label[key] = label
            self.value_type[key] = valueType
            self.setdefault(key, value)

        phase_setting_items: List[Tuple[str, Any]] = []
        phase_setting_items.extend([('smoothing', True),
                                    ('smoothing_iterations', 8)])

        for key, value in phase_setting_items:
            self.setdefault(key, value)
