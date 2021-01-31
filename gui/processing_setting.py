from tkinter import LabelFrame
from typing import Dict

from gui.general_setting import SettingEntry
from processing.processing_config import ProcessingConfig


class ProcessingSetting:
    def __init__(self, processing_config: ProcessingConfig, root: LabelFrame):
        self.processing_config: ProcessingConfig = processing_config
        self.settings: Dict[str, SettingEntry] = {}

        for i, (key, item) in enumerate(self.processing_config.items()):
            if key in self.processing_config.value_type.keys():
                self.settings[key] = SettingEntry(root, self.processing_config.label[key], row=i, column=0,
                                                  variable_type=self.processing_config.value_type[key])

    def set(self):
        for key, setting in self.settings.items():
            setting.set(self.processing_config[key])

    def update_config(self):
        for key, setting in self.settings.items():
            self.processing_config[key] = setting.get()
