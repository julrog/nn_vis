from enum import Enum
from typing import Dict, List, Tuple

from utility.config import BaseConfig


class RenderConfigType(Enum):
    GENERAL = 1
    SHADER = 2
    SELECTION = 3


class RenderingConfig(BaseConfig):
    def __init__(self, name: str = None):
        if name is None:
            super().__init__("rendering")
        else:
            super().__init__("rendering", name)

        self.shader_label: Dict[str, str] = dict()
        self.shader_name: Dict[str, str] = dict()
        self.selection_labels: Dict[str, List[str]] = dict()
        self.selection_values: Dict[str, List[any]] = dict()
        self.item_type: Dict[str, RenderConfigType] = dict()
        self.set_defaults()

    def set_defaults(self):
        shader_items: List[Tuple[str, str, str, any]] = []
        shader_items.extend([("screen_width", "screen_width", "Screen Width", 1920.0),
                             ("screen_height", "screen_height", "Screen Height", 1080.0),
                             ("edge_object_radius", "object_radius", "Size", 0.04),
                             ("edge_base_opacity", "base_opacity", "Base Opacity", 0.0),
                             ("edge_importance_opacity", "importance_opacity", "Importance Opacity", 1.1),
                             ("edge_depth_opacity", "depth_opacity", "Depth Opacity", 0.5),
                             ("edge_opacity_exponent", "opacity_exponent", "Depth Exponent", 0.5),
                             ("edge_importance_threshold", "importance_threshold", "Importance Threshold", 0.01),
                             ("node_object_radius", "object_radius", "Size", 0.05),
                             ("node_base_opacity", "base_opacity", "Base Opacity", 0.0),
                             ("node_importance_opacity", "importance_opacity", "Importance Opacity", 1.0),
                             ("node_depth_opacity", "depth_opacity", "Depth Opacity", 0.5),
                             ("node_opacity_exponent", "opacity_exponent", "Depth Exponent", 0.5),
                             ("node_importance_threshold", "importance_threshold", "Importance Threshold", 0.01)])

        for key, shader_name, label, value in shader_items:
            self.shader_label[key] = label
            self.shader_name[key] = shader_name
            self.item_type[key] = RenderConfigType.SHADER
            self.setdefault(key, value)

        selection_items: List[Tuple[str, List[str], List[any], int]] = []
        selection_items.extend([("grid_render_mode", ["None", "Cube", "Point"], [0, 1, 2], 0),
                                ("edge_render_mode",
                                 ["None", "Sphere", "Sphere_Transparent", "Ellipsoid_Transparent", "Line", "Point"],
                                 [0, 1, 2, 3, 4],
                                 3),
                                ("node_render_mode", ["None", "Sphere", "Sphere_Transparent", "Point"],
                                 [0, 1, 2, 3], 2)])

        for key, labels, values, default in selection_items:
            self.selection_labels[key] = labels
            self.selection_values[key] = values
            self.item_type[key] = RenderConfigType.SELECTION
            self.setdefault(key, default)
