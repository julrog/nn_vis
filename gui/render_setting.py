from tkinter import LabelFrame, IntVar, Radiobutton, Button
from typing import List, Dict

from gui.general_setting import SettingEntry


class RenderSettings:
    def __init__(self, root: LabelFrame, name: str, change_setting_func, render_options: List[str],
                 default_value: int = 0, shader_settings: Dict[str, any] = None, row: int = 0, column: int = 0):
        self.name: str = name
        self.render_frame: LabelFrame = LabelFrame(root, text=self.name, width=60,
                                                   padx=1, pady=1)
        self.render_mode: IntVar = IntVar(value=default_value)
        self.render_radio_buttons: List[Radiobutton] = []
        self.shader_settings: List[SettingEntry] = []
        self.shader_setting_frame: LabelFrame = LabelFrame(self.render_frame, text="Shader Settings", padx=1, pady=1)

        def create_apply_func(function, inner_func):
            def command():
                function(inner_func)

            return command

        self.apply_settings: Button = Button(self.shader_setting_frame, text="Apply",
                                             command=create_apply_func(self.get_settings, change_setting_func))

        def create_radio_func(setting_value: int):
            def command():
                change_setting_func("render", self.name, setting_value)

            return command

        for i, option in enumerate(render_options):
            self.render_radio_buttons.append(
                Radiobutton(self.render_frame, text=option, variable=self.render_mode, value=i,
                            command=create_radio_func(i)))
            self.render_radio_buttons[i].grid(row=i, column=0)

        change_setting_func("render", self.name, default_value)

        if shader_settings is not None:
            for i, (setting, value) in enumerate(shader_settings.items()):
                self.shader_settings.append(SettingEntry(self.shader_setting_frame, setting, value, "float", i, 0))

            self.apply_settings.grid(row=len(self.shader_settings), column=0, columnspan=2)
            self.shader_setting_frame.grid(row=0, column=1, rowspan=len(self.render_radio_buttons), padx=1, pady=1)
        self.render_frame.grid(row=row, column=column, padx=1, pady=1)

        self.get_settings(change_setting_func)

    def get_settings(self, change_setting_func):
        render_settings: Dict[str, float] = dict()
        for shader_setting in self.shader_settings:
            render_settings[shader_setting.name] = shader_setting.get()
        change_setting_func("render_shader_setting", self.name, render_settings)