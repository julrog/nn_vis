from typing import Any, List, Tuple

from utility.config import BaseConfig


class WindowConfig(BaseConfig):
    def __init__(self, name: str = None):
        if name is None:
            super().__init__('window')
        else:
            super().__init__('window', name)

        self.set_defaults()

    def set_defaults(self):
        render_setting_items: List[Tuple[str, Any]] = []
        render_setting_items.extend([('title', 'NNVis Render'),
                                     ('width', 1600),
                                     ('height', 900),
                                     ('screen_width', 1600),
                                     ('screen_height', 900),
                                     ('screen_x', 0),
                                     ('screen_y', 0),
                                     ('monitor_id', None),
                                     ('camera_rotation', True),
                                     ('camera_rotation_speed', 0.5)])

        for key, value in render_setting_items:
            self.setdefault(key, value)
        self.store()
