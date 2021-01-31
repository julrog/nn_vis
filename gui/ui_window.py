import ntpath
from tkinter import *
from tkinter import filedialog, messagebox
from typing import List, Dict

from data.data_handler import ImportanceDataHandler, ProcessedNNHandler
from definitions import DATA_PATH
from gui.frame_building import set_render_frame, set_architecture_frame, set_stat_frame, set_processing_frame
from gui.general_setting import RadioButtons
from gui.neural_network_setting import LayerSettings
from gui.processing_setting import ProcessingSetting
from processing.processing_config import ProcessingConfig
from rendering.rendering_config import RenderingConfig
from utility.window_config import WindowConfig


class OptionGui:
    def __init__(self):
        self.window_config: WindowConfig = WindowConfig("ui")

        self.gui_root: Tk = Tk()
        self.layer_settings: List[LayerSettings] = []
        self.settings: Dict[any, any] = {"Closed": False, "current_layer_data": []}
        self.render_config: RenderingConfig = RenderingConfig()
        self.processing_config: ProcessingConfig = ProcessingConfig()

        self.gui_root.title("NNVIS Options")

        set_stat_frame(self.gui_root, self.settings)

        architecture, layer_label = set_architecture_frame(self.gui_root, self.save_processed_nn_file,
                                                           self.open_processed_nn_file, self.open_importance_file,
                                                           self.generate, self.add_layer, self.clear_layer)
        self.architecture_frame: LabelFrame = architecture
        self.layer_label: Label = layer_label
        self.class_show_options: RadioButtons = set_render_frame(self.gui_root, self.change_render_config,
                                                                 self.change_setting, self.render_config)

        action_buttons, processing_setting = set_processing_frame(self.gui_root, self.processing_config,
                                                                  self.change_setting, self.change_processing_config)
        self.action_buttons: RadioButtons = action_buttons
        self.processing_setting: ProcessingSetting = processing_setting

        self.gui_root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.gui_root.geometry("+%d+%d" % (self.window_config["screen_x"], self.window_config["screen_y"]))
        self.gui_root.bind("<Configure>", self.handle_configure)

    def start(self, layer_data: List[int] = None):
        if layer_data is None:
            default_layer_data = [4, 9, 9]
            for nodes in default_layer_data:
                self.add_layer(nodes)
        else:
            for nodes in layer_data:
                self.add_layer(nodes)

        self.processing_setting.set()
        self.generate()

        self.gui_root.mainloop()
        self.settings["Closed"] = True

    def handle_configure(self, event):
        self.window_config["screen_x"] = self.gui_root.winfo_x()
        self.window_config["screen_y"] = self.gui_root.winfo_y()
        self.window_config.store()

    def save_processed_nn_file(self):
        filename = filedialog.asksaveasfilename()
        if not filename:
            return
        self.settings["save_processed_nn_path"] = filename + ".pro.npz"
        self.settings["save_file"] = True

    def open_processed_nn_file(self):
        filename = filedialog.askopenfilename(initialdir=DATA_PATH, title="Select A File",
                                              filetypes=(("processed nn files", "*.pro.npz"),))
        data_loader: ProcessedNNHandler = ProcessedNNHandler(filename)
        self.settings["network_name"] = ntpath.basename(filename) + "_processed"
        self.update_layer(data_loader.layer_data, processed_nn=data_loader)

    def open_importance_file(self):
        filename = filedialog.askopenfilename(initialdir=DATA_PATH, title="Select A File",
                                              filetypes=(("importance files", "*.imp.npz"),))
        data_loader: ImportanceDataHandler = ImportanceDataHandler(filename)
        self.settings["network_name"] = ntpath.basename(filename) + "_raw"
        self.update_layer(data_loader.layer_data, importance_data=data_loader)

    def update_layer(self, layer_data: List[int], importance_data: ImportanceDataHandler = None,
                     processed_nn: ProcessedNNHandler = None):
        self.clear_layer()

        for nodes in layer_data:
            self.add_layer(nodes)

        self.generate(importance_data, processed_nn)
        self.set_classes(layer_data[len(layer_data) - 1])

    def add_layer(self, nodes: int = 9):
        layer_id: int = len(self.layer_settings)
        self.layer_settings.append(LayerSettings(self.architecture_frame, layer_id, 5, 0, self.remove_layer))
        self.layer_settings[layer_id].set_neurons(nodes)

    def clear_layer(self):
        for ls in self.layer_settings:
            ls.remove()
        self.layer_settings = []

    def remove_layer(self, layer_id: int):
        self.layer_settings[layer_id].remove()
        self.layer_settings.remove(self.layer_settings[layer_id])

        for i, ls in enumerate(self.layer_settings):
            ls.layer_id = i
            ls.grid()
        self.layer_label.grid(row=0, column=0)

    def generate(self, importance_data: ImportanceDataHandler = None, processed_nn: ProcessedNNHandler = None):
        self.action_buttons.press(0)
        layer_data: List[int] = []
        for ls in self.layer_settings:
            layer_data.append(ls.get_neurons())
        self.settings["current_layer_data"] = layer_data
        self.settings["importance_data"] = importance_data
        self.settings["processed_nn"] = processed_nn
        self.processing_setting.update_config()
        self.settings["update_model"] = True
        self.processing_config.store()

    def change_setting(self, setting_type: str, sub_type: str, value: int, stop_action: bool = False):
        if stop_action:
            self.action_buttons.press(0)
        self.settings[setting_type + "_" + sub_type] = value

    def change_render_config(self, name: str, value: int, stop_action: bool = False):
        if stop_action:
            self.action_buttons.press(0)
        self.render_config[name] = value
        self.render_config.store()

    def change_processing_config(self, name: str, value: int):
        self.processing_config[name] = value
        self.processing_config.store()

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.settings["Closed"] = True

    def set_classes(self, num_classes: int):
        show_class_names: List[str] = ["Independent", "All"]
        for class_id in range(num_classes):
            show_class_names.append("Class " + str(class_id))
        self.class_show_options.set_buttons(show_class_names)

    def destroy(self):
        self.gui_root.destroy()
