import ntpath
from tkinter import *
from tkinter import filedialog
from typing import List, Dict

from definitions import DATA_PATH
from data.data_handler import ImportanceDataHandler, ProcessedNNHandler


class LayerSettings:
    def __init__(self, root: LabelFrame, layer_id: int, row: int, column: int, remove_func):
        self.layer_id: int = layer_id
        self.row: int = row
        self.column: int = column
        self.remove_button: Button = Button(root, text="Remove", command=lambda: remove_func(self.layer_id))
        self.neuron_count_entry: Entry = Entry(root, width=5)
        self.neuron_count_entry.insert(0, "9")
        self.layer_label: Label = Label(root, text="Layer " + str(self.layer_id + 1))
        self.grid()

    def grid(self):
        self.layer_label.config(text="Layer " + str(self.layer_id + 1))
        self.remove_button.grid(row=self.row + self.layer_id, column=self.column + 2)
        self.neuron_count_entry.grid(row=self.row + self.layer_id, column=self.column + 1)
        self.layer_label.grid(row=self.row + self.layer_id, column=self.column)

    def set_neurons(self, neurons: int):
        self.neuron_count_entry.delete(0, END)
        self.neuron_count_entry.insert(0, str(neurons))

    def get_neurons(self) -> int:
        return int(self.neuron_count_entry.get())

    def remove(self):
        self.remove_button.destroy()
        self.neuron_count_entry.destroy()
        self.layer_label.destroy()


class SettingField:
    def __init__(self, root: LabelFrame, name: str, variable_value: any = 0, variable_type: str = "string",
                 row: int = 0, column: int = 0):
        self.name: str = name
        self.variable_type: str = variable_type
        if variable_type is "int":
            self.variable: IntVar = IntVar(value=int(variable_value))
        elif variable_type is "float":
            self.variable: DoubleVar = DoubleVar(value=float(variable_value))
        else:
            self.variable: StringVar = StringVar(value=str(variable_value))
        self.label: Label = Label(root, text=name)
        self.label.grid(row=row, column=column)
        self.variable_field: Label = Label(root, textvariable=self.variable)
        self.variable_field.grid(row=row, column=column + 1)

    def set(self, value: any):
        if self.variable_type is "int":
            self.variable.set(int(value))
        elif self.variable_type is "float":
            self.variable.set(float(value))
        else:
            self.variable.set(str(value))

    def get(self) -> any:
        if self.variable_type is "int":
            return int(self.variable.get())
        elif self.variable_type is "float":
            return float(self.variable.get())
        else:
            return str(self.variable.get())


class SettingEntry:
    def __init__(self, root: LabelFrame, name: str, variable_value: any = 0, variable_type: str = "string",
                 row: int = 0, column: int = 0):
        self.name: str = name
        self.variable_type: str = variable_type
        self.variable: StringVar = StringVar(value=str(variable_value))
        self.label: Label = Label(root, text=name)
        self.label.grid(row=row, column=column)
        self.variable_entry: Entry = Entry(root, width=5, textvariable=self.variable)
        self.variable_entry.grid(row=row, column=column + 1)

    def set(self, value: any):
        if self.variable_type is "int":
            self.variable.set(int(value))
        elif self.variable_type is "float":
            self.variable.set(float(value))
        else:
            self.variable.set(str(value))

    def get(self) -> any:
        if self.variable_type is "int":
            return int(self.variable.get())
        elif self.variable_type is "float":
            return float(self.variable.get())
        else:
            return str(self.variable.get())


class RadioButtons:
    def __init__(self, root: LabelFrame, names: List[str], shared_variable: IntVar, command, option: str,
                 sub_option: str, row: int = 0, column: int = 0, width: int = 15, height: int = 1):
        self.names: List[str] = names
        self.option: str = option
        self.sub_option: str = sub_option
        self.variable: IntVar = shared_variable
        self.command = command
        self.buttons: List[Button] = []
        for i, name in enumerate(self.names):
            def generate_press_function(current_index: int):
                def press_function():
                    self.press(current_index)

                return press_function

            new_button: Button = Button(root, text=name, width=width, height=height, command=generate_press_function(i))
            new_button.grid(row=row + i, column=column)
            self.buttons.append(new_button)
        self.press(0)

    def press(self, button_id: int):
        self.set(button_id)
        for i, button in enumerate(self.buttons):
            if i is not button_id:
                button.config(relief=RAISED)
            else:
                button.config(relief=SUNKEN)
        self.command(self.option, self.sub_option, self.variable.get())

    def set(self, value: any):
        self.variable.set(int(value))

    def get(self) -> any:
        return int(self.variable.get())


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


class OptionGui:
    def __init__(self):
        self.test: bool = False

        self.gui_root: Tk = Tk()
        self.layer_settings: List[LayerSettings] = []
        self.settings: Dict[str, any] = {"Closed": False, "current_layer_data": []}

        self.gui_root.title("NNVIS Options")

        self.stats_frame: LabelFrame = LabelFrame(self.gui_root, text="Statistics", width=60,
                                                  padx=5, pady=5)
        self.stats_frame.grid(row=0, column=0, padx=5, pady=5)
        self.edge_count_setting: SettingField = SettingField(self.stats_frame, "Edges:", row=0, column=0)
        self.settings["edge_count"] = self.edge_count_setting
        self.sample_count_setting: SettingField = SettingField(self.stats_frame, "Samples:", row=1, column=0)
        self.settings["sample_count"] = self.sample_count_setting
        self.cell_count_setting: SettingField = SettingField(self.stats_frame, "Grid Cells:", row=2, column=0)
        self.settings["cell_count"] = self.cell_count_setting
        self.pruned_edges: SettingField = SettingField(self.stats_frame, "Pruned Edges:", row=3, column=0)
        self.settings["pruned_edges"] = self.pruned_edges
        self.frame_time: SettingField = SettingField(self.stats_frame, "FPS:", row=4, column=0)
        self.settings["fps"] = self.frame_time

        self.architecture_frame: LabelFrame = LabelFrame(self.gui_root, text="Neural Network Architecture", width=60,
                                                         padx=5, pady=5)
        self.architecture_frame.grid(row=1, column=0, rowspan=2, padx=5, pady=5)
        self.save_processed_button: Button = Button(self.architecture_frame, text="Save Processed Network", width=20,
                                                    command=self.save_processed_nn_file)
        self.save_processed_button.grid(row=0, column=0, columnspan=3)
        self.load_processed_button: Button = Button(self.architecture_frame, text="Load Processed Network", width=20,
                                                    command=self.open_processed_nn_file)
        self.load_processed_button.grid(row=1, column=0, columnspan=3)
        self.load_button: Button = Button(self.architecture_frame, text="Load Network", width=20,
                                          command=self.open_importance_file)
        self.load_button.grid(row=2, column=0, columnspan=3)
        self.generate_button: Button = Button(self.architecture_frame, text="Generate Network", width=20,
                                              command=self.generate)
        self.generate_button.grid(row=3, column=0, columnspan=3)
        self.layer_label: Label = Label(self.architecture_frame, text="Modify:")
        self.add_layer_button: Button = Button(self.architecture_frame, text="Add Layer", command=self.add_layer)
        self.clear_layer_button: Button = Button(self.architecture_frame, text="Clear Layer", command=self.clear_layer)
        self.layer_label.grid(row=4, column=0)
        self.add_layer_button.grid(row=4, column=1)
        self.clear_layer_button.grid(row=4, column=2)

        self.render_frame: LabelFrame = LabelFrame(self.gui_root, text="Render Settings", width=60,
                                                   padx=5, pady=5)
        self.render_frame.grid(row=0, column=2, columnspan=2, rowspan=3, padx=5, pady=5)

        self.grid_render_settings: RenderSettings = RenderSettings(self.render_frame, "Grid", self.change_setting,
                                                                   ["None", "Cube", "Point"], 0, row=0, column=0)
        edge_shader_settings: Dict[str, any] = {"Size": 0.1, "Base Opacity": 0.0, "Importance Opacity": 1.0,
                                                "Density Exponent": 0.1, "Importance Threshold": 0.01}
        self.edge_render_settings: RenderSettings = RenderSettings(self.render_frame, "Edge", self.change_setting,
                                                                   ["None", "Sphere", "Sphere_Transparent",
                                                                    "Ellipsoid_Transparent", "Line", "Point"],
                                                                   3, edge_shader_settings, row=1, column=0)
        node_shader_settings: Dict[str, any] = {"Size": 0.05, "Base Opacity": 0.0, "Importance Opacity": 1.0,
                                                "Density Exponent": 0.1, "Importance Threshold": 0.01}
        self.node_render_settings: RenderSettings = RenderSettings(self.render_frame, "Node", self.change_setting,
                                                                   ["None", "Sphere", "Sphere_Transparent", "Point"], 2,
                                                                   node_shader_settings, row=2, column=0)
        self.class_setting_frame: LabelFrame = LabelFrame(self.render_frame, text="Class Visibility", width=60,
                                                          padx=5, pady=5)
        self.class_setting_frame.grid(row=0, column=1, rowspan=3, padx=5, pady=5)
        self.class_show: IntVar = IntVar(value=0)
        self.class_show_options: RadioButtons = RadioButtons(self.class_setting_frame,
                                                             ["Independent", "All", "Class 0", "Class 1", "Class 2",
                                                              "Class 3", "Class 4", "Class 5", "Class 6", "Class 7",
                                                              "Class 8", "Class 9"], self.class_show,
                                                             command=self.change_setting, option="show",
                                                             sub_option="class", row=0, column=0, width=10, height=2)

        self.action_frame: LabelFrame = LabelFrame(self.gui_root, text="Actions", width=60,
                                                   padx=5, pady=5)
        self.action_frame.grid(row=1, column=1, rowspan=2, padx=5, pady=5)
        self.sample_button: Button = Button(self.action_frame, text="Resample Edges", width=15,
                                            command=lambda: self.change_setting("trigger_network", "sample", 1, True))
        self.sample_button.grid(row=0, column=0)
        self.action_state: IntVar = IntVar(value=0)
        self.action_buttons: RadioButtons = RadioButtons(self.action_frame,
                                                         ["Stop Everything", "Node Advect", "Node Diverge",
                                                          "Node Noise", "Edge Advect", "Edge Diverge", "Edge Noise"],
                                                         self.action_state, command=self.change_setting,
                                                         option="action", sub_option="state", row=2, column=0)

        self.smoothing_status: IntVar = IntVar(value=1)
        self.smoothing_checkbox: Checkbutton = Checkbutton(self.action_frame, text="Smoothing",
                                                           variable=self.smoothing_status,
                                                           command=lambda: self.change_setting("edge",
                                                                                               "smoothing",
                                                                                               self.smoothing_status.get()))
        self.change_setting("edge", "smoothing", self.smoothing_status.get())
        self.smoothing_checkbox.grid(row=1, column=0)

        self.setting_frame: LabelFrame = LabelFrame(self.gui_root, text="Settings", width=60,
                                                    padx=5, pady=5)
        self.setting_frame.grid(row=0, column=1, padx=5, pady=5)
        self.layer_distance: SettingEntry = SettingEntry(self.setting_frame, "Layer distance:", row=0, column=0,
                                                         variable_type="float")
        self.layer_width: SettingEntry = SettingEntry(self.setting_frame, "Layer width:", row=1, column=0,
                                                      variable_type="float")
        self.sampling_rate: SettingEntry = SettingEntry(self.setting_frame, "Sampling rate:", row=2, column=0,
                                                        variable_type="float")
        self.prune_percentage: SettingEntry = SettingEntry(self.setting_frame, "Prune percentage:", row=3,
                                                           column=0, variable_type="float")
        self.node_bandwidth_reduction: SettingEntry = SettingEntry(self.setting_frame, "Node Bandwidth reduction:",
                                                                   row=4, column=0, variable_type="float")
        self.edge_bandwidth_reduction: SettingEntry = SettingEntry(self.setting_frame, "Edge Bandwidth reduction:",
                                                                   row=5, column=0, variable_type="float")
        self.edge_importance_type: SettingEntry = SettingEntry(self.setting_frame, "Edge Importance Type:",
                                                               row=6, column=0, variable_type="int")

    def start(self, layer_data: List[int] = None, layer_distance: float = 1.0, node_size: float = 1.0,
              sampling_rate: float = 10.0, prune_percentage: float = 0.0, node_bandwidth_reduction: float = 0.98,
              edge_bandwidth_reduction: float = 0.9, edge_importance_type: int = 0):
        if layer_data is None:
            default_layer_data = [4, 9, 4]
            for nodes in default_layer_data:
                self.add_layer(nodes)
        else:
            for nodes in layer_data:
                self.add_layer(nodes)

        self.layer_distance.set(layer_distance)
        self.layer_width.set(node_size)
        self.sampling_rate.set(sampling_rate)
        self.prune_percentage.set(prune_percentage)
        self.node_bandwidth_reduction.set(node_bandwidth_reduction)
        self.edge_bandwidth_reduction.set(edge_bandwidth_reduction)
        self.edge_importance_type.set(edge_importance_type)
        self.generate()

        self.gui_root.mainloop()
        self.settings["Closed"] = True

    def save_processed_nn_file(self):
        filename = filedialog.asksaveasfilename()
        if not filename:
            return
        self.settings["save_processed_nn_path"] = filename
        self.settings["save_file"] = True

    def open_processed_nn_file(self):
        filename = filedialog.askopenfilename(initialdir=DATA_PATH, title="Select A File",
                                              filetypes=(("processed nn files", "*.npz"),))
        data_loader: ProcessedNNHandler = ProcessedNNHandler(filename)
        self.settings['network_name'] = ntpath.basename(filename) + "_processed"
        self.update_layer(data_loader.layer_data, processed_nn=data_loader)

    def open_importance_file(self):
        filename = filedialog.askopenfilename(initialdir=DATA_PATH, title="Select A File",
                                              filetypes=(("importance files", "*.npz"),))
        data_loader: ImportanceDataHandler = ImportanceDataHandler(filename)
        self.settings['network_name'] = ntpath.basename(filename) + "_raw"
        self.update_layer(data_loader.layer_data, importance_data=data_loader)

    def update_layer(self, layer_data: List[int], importance_data: ImportanceDataHandler = None,
                     processed_nn: ProcessedNNHandler = None):
        self.clear_layer()

        for nodes in layer_data:
            self.add_layer(nodes)
        self.generate(importance_data, processed_nn)

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
        self.settings["layer_distance"] = self.layer_distance.get()
        self.settings["layer_width"] = self.layer_width.get()
        self.settings["sampling_rate"] = self.sampling_rate.get()
        self.settings["prune_percentage"] = self.prune_percentage.get()
        self.settings["node_bandwidth_reduction"] = self.node_bandwidth_reduction.get()
        self.settings["edge_bandwidth_reduction"] = self.edge_bandwidth_reduction.get()
        self.settings["edge_importance_type"] = self.edge_importance_type.get()
        self.settings["update_model"] = True

    def change_setting(self, setting_type: str, sub_type: str, value: int, stop_action: bool = False):
        if stop_action:
            self.action_buttons.press(0)
        self.settings[setting_type + "_" + sub_type] = value
