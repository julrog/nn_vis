from tkinter import LabelFrame, IntVar, Tk, Button, Label, Checkbutton
from typing import List, Callable, Dict, Tuple

from gui.general_setting import RadioButtons, SettingField
from gui.constants import STATISTIC_NAME, StatisticLink
from gui.processing_setting import ProcessingSetting
from gui.render_setting import RenderSettings
from processing.processing_config import ProcessingConfig
from rendering.rendering_config import RenderingConfig
from rendering.shader_uniforms import EDGE_SHADER_UNIFORM, NODE_SHADER_UNIFORM


def set_stat_frame(gui_root: Tk, settings: Dict[any, any]):
    stats_frame: LabelFrame = LabelFrame(gui_root, text="Statistics", width=60, padx=5, pady=5)
    stats_frame.grid(row=0, column=0, padx=5, pady=5)
    for stat in range(len(STATISTIC_NAME)):
        settings[StatisticLink(stat)] = SettingField(stats_frame, STATISTIC_NAME[stat] + ":", row=stat, column=0)


def set_render_frame(gui_root: Tk, change_render_config: Callable, change_setting: Callable,
                     render_config: RenderingConfig) -> RadioButtons:
    render_frame: LabelFrame = LabelFrame(gui_root, text="Render Settings", width=60, padx=5, pady=5)
    render_frame.grid(row=0, column=3, columnspan=2, rowspan=3, padx=5, pady=5)

    RenderSettings(render_frame, "Grid", change_render_config, render_config, "grid_render_mode",
                   None, row=0, column=0)
    RenderSettings(render_frame, "Edge", change_render_config, render_config, "edge_render_mode",
                   EDGE_SHADER_UNIFORM, row=1, column=0)
    RenderSettings(render_frame, "Node", change_render_config, render_config, "node_render_mode",
                   NODE_SHADER_UNIFORM, row=2, column=0)

    class_setting_frame: LabelFrame = LabelFrame(render_frame, text="Class Visibility", width=60,
                                                 padx=5, pady=5)
    class_setting_frame.grid(row=0, column=1, rowspan=3, padx=5, pady=5)
    class_show: IntVar = IntVar(value=0)

    show_class_names: List[str] = ["Independent", "All"]
    for class_id in range(9):
        show_class_names.append("Class " + str(class_id))
    return RadioButtons(class_setting_frame, show_class_names, class_show, command=change_setting,
                        option="show", sub_option="class", row=0, column=0, width=10, height=2)


def set_architecture_frame(gui_root: Tk, save_processed_nn_file: Callable,
                           open_processed_nn_file: Callable, open_importance_file: Callable,
                           generate: Callable, add_layer: Callable, clear_layer: Callable) -> Tuple[LabelFrame, Label]:
    architecture_frame: LabelFrame = LabelFrame(gui_root, text="Neural Network Architecture", width=60,
                                                padx=5, pady=5)
    architecture_frame.grid(row=1, column=0, rowspan=2, padx=5, pady=5)
    save_processed_button: Button = Button(architecture_frame, text="Save Processed Network", width=20,
                                           command=save_processed_nn_file)
    save_processed_button.grid(row=0, column=0, columnspan=3)
    load_processed_button: Button = Button(architecture_frame, text="Load Processed Network", width=20,
                                           command=open_processed_nn_file)
    load_processed_button.grid(row=1, column=0, columnspan=3)
    load_button: Button = Button(architecture_frame, text="Load Network", width=20,
                                 command=open_importance_file)
    load_button.grid(row=2, column=0, columnspan=3)
    generate_button: Button = Button(architecture_frame, text="Generate Network", width=20,
                                     command=generate)
    generate_button.grid(row=3, column=0, columnspan=3)
    add_layer_button: Button = Button(architecture_frame, text="Add Layer", command=add_layer)
    add_layer_button.grid(row=4, column=1)
    clear_layer_button: Button = Button(architecture_frame, text="Clear Layer", command=clear_layer)
    clear_layer_button.grid(row=4, column=2)
    layer_label: Label = Label(architecture_frame, text="Modify:")
    layer_label.grid(row=4, column=0)
    return architecture_frame, layer_label


def set_processing_frame(gui_root: Tk, processing_config: ProcessingConfig, change_setting: Callable,
                         change_processing_config: Callable) -> Tuple[RadioButtons, ProcessingSetting]:
    processing_frame: LabelFrame = LabelFrame(gui_root, text="Processing", width=60, padx=5, pady=5)
    processing_frame.grid(row=0, column=1, columnspan=2, rowspan=3, padx=5, pady=5)

    action_frame: LabelFrame = LabelFrame(processing_frame, text="Actions", width=60,
                                          padx=5, pady=5)
    action_frame.grid(row=1, column=0, rowspan=2, padx=5, pady=5)
    sample_button: Button = Button(action_frame, text="Resample Edges", width=15,
                                   command=lambda: change_setting("trigger_network", "sample", 1, True))
    sample_button.grid(row=0, column=0)
    action_state: IntVar = IntVar(value=0)
    action_buttons: RadioButtons = RadioButtons(action_frame,
                                                ["Stop Everything", "Node Advect", "Node Diverge",
                                                 "Node Noise", "Edge Advect", "Edge Diverge", "Edge Noise"],
                                                action_state, command=change_setting,
                                                option="action", sub_option="state", row=2, column=0)

    smoothing_status: IntVar = IntVar(value=processing_config["smoothing"])
    smoothing_checkbox: Checkbutton = Checkbutton(action_frame, text="Smoothing",
                                                  variable=smoothing_status,
                                                  command=lambda: change_processing_config(
                                                      "smoothing",
                                                      smoothing_status.get()))
    change_setting("edge", "smoothing", smoothing_status.get())
    smoothing_checkbox.grid(row=1, column=0)

    setting_frame: LabelFrame = LabelFrame(processing_frame, text="Settings", width=60,
                                           padx=5, pady=5)
    setting_frame.grid(row=0, column=0, padx=5, pady=5)
    processing_setting: ProcessingSetting = ProcessingSetting(processing_config, setting_frame)

    return action_buttons, processing_setting
