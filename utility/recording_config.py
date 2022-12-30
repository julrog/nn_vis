from typing import Any, List, Tuple

from utility.camera import CameraPose
from utility.config import BaseConfig
from utility.types import ProcessRenderMode


class RecordingConfig(BaseConfig):
    def __init__(self, name: str = None):
        if name is None:
            super().__init__('recording')
        else:
            super().__init__('recording', name)

        self.set_defaults()

    def set_defaults(self):
        render_setting_items: List[Tuple[str, Any]] = []
        render_setting_items.extend([('screenshot_width', 1600),
                                     ('screenshot_height', 900),
                                     ('screenshot_mode', ProcessRenderMode.FINAL |
                                      ProcessRenderMode.NODE_ITERATIONS),
                                     ('camera_pose_final',
                                      CameraPose.UPPER_BACK_RIGHT),
                                     ('camera_pose_list', [
                                      CameraPose.UPPER_BACK_RIGHT]),
                                     ('camera_rotation', True),
                                     ('camera_rotation_speed', 0.5),
                                     ('class_list', [0])])

        for key, value in render_setting_items:
            self.setdefault(key, value)
        self.store()
