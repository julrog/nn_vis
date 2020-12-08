from enum import IntEnum
from typing import List, Tuple

from utility.camera import CameraPose
from utility.config import BaseConfig


class ProcessRenderMode(IntEnum):
    NONE = 0
    FINAL = 1
    NODE_ITERATIONS = 10
    EDGE_ITERATIONS = 11
    NODE_EDGE_ITERATIONS = 12
    EDGE_ITERATIONS_SMOOTHING = 21
    NODE_EDGE_ITERATIONS_SMOOTHING = 22


class AutomationConfig(BaseConfig):
    def __init__(self, name: str = None):
        if name is None:
            super().__init__("automation")
        else:
            super().__init__("automation", name)

        self.set_defaults()

    def set_defaults(self):
        render_setting_items: List[Tuple[str, any]] = []
        render_setting_items.extend([("screenshot_width", 1600),
                                     ("screenshot_height", 900),
                                     ("screenshot_mode", ProcessRenderMode.EDGE_ITERATIONS_SMOOTHING),
                                     ("camera_pose_final", CameraPose.UPPER_BACK_RIGHT),
                                     ("camera_pose_list", [CameraPose.UPPER_BACK_RIGHT]),
                                     ("camera_rotation", True),
                                     ("camera_rotation_speed", 0.5),
                                     ("class_list", [0])])

        for key, value in render_setting_items:
            self.setdefault(key, value)
