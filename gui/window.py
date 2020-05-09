from tkinter import *
from typing import List, Dict


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
    def __init__(self, root: LabelFrame, names: List[str], shared_variable: IntVar, command, row: int = 0,
                 column: int = 0):
        self.names: List[str] = names
        self.variable: IntVar = shared_variable
        self.buttons: List[Button] = []

        for i, name in enumerate(self.names):
            def generate_press_function(current_index: int):
                def press_function():
                    self.press(current_index, command)

                return press_function

            new_button: Button = Button(root, text=name, width=10, command=generate_press_function(i))
            new_button.grid(row=row + i, column=column)
            self.buttons.append(new_button)
        self.press(0, command)

    def press(self, button_id: int, command):
        self.set(button_id)
        for i, button in enumerate(self.buttons):
            if i is not button_id:
                button.config(relief=RAISED)
            else:
                button.config(relief=SUNKEN)
        command("action", "state", self.variable.get())

    def set(self, value: any):
        self.variable.set(int(value))

    def get(self) -> any:
        return int(self.variable.get())


class OptionGui:
    def __init__(self):
        self.test: bool = False

        self.gui_root: Tk = Tk()
        self.layer_settings: List[LayerSettings] = []
        self.settings: Dict[str, any] = {"Closed": False, "current_layer_data": []}

        self.gui_root.title("NNVIS Options")

        self.architecture_frame: LabelFrame = LabelFrame(self.gui_root, text="Neural Network Architecture", width=60,
                                                         padx=5, pady=5)
        self.architecture_frame.grid(row=0, column=0, rowspan=3, padx=5, pady=5)
        self.layer_label: Label = Label(self.architecture_frame, text="Actions:")
        self.add_layer_button: Button = Button(self.architecture_frame, text="Add Layer", command=self.add_layer)
        self.clear_layer_button: Button = Button(self.architecture_frame, text="Clear Layer", command=self.clear_layer)
        self.layer_label.grid(row=0, column=0)
        self.add_layer_button.grid(row=0, column=1)
        self.clear_layer_button.grid(row=0, column=2)

        self.stats_frame: LabelFrame = LabelFrame(self.gui_root, text="Statistics", width=60,
                                                  padx=5, pady=5)
        self.stats_frame.grid(row=0, column=1, padx=5, pady=5)
        self.edge_count_setting: SettingField = SettingField(self.stats_frame, "Edges:", row=0, column=0)
        self.settings["edge_count"] = self.edge_count_setting
        self.sample_count_setting: SettingField = SettingField(self.stats_frame, "Samples:", row=1, column=0)
        self.settings["sample_count"] = self.sample_count_setting
        self.cell_count_setting: SettingField = SettingField(self.stats_frame, "Grid Cells:", row=2, column=0)
        self.settings["cell_count"] = self.cell_count_setting

        self.render_frame: LabelFrame = LabelFrame(self.gui_root, text="Render Settings", width=60,
                                                   padx=5, pady=5)
        self.render_frame.grid(row=1, column=1, rowspan=2, padx=5, pady=5)

        self.grid_render_frame: LabelFrame = LabelFrame(self.render_frame, text="Grid", width=60,
                                                        padx=5, pady=5)
        self.grid_render_mode = IntVar(value=1)
        self.settings["render_grid"] = 1
        self.rb_grid_render_cube: Radiobutton = Radiobutton(self.grid_render_frame, text="Cube",
                                                            variable=self.grid_render_mode, value=1,
                                                            command=lambda: self.change_setting("render", "grid", 1))
        self.rb_grid_render_none: Radiobutton = Radiobutton(self.grid_render_frame, text="None",
                                                            variable=self.grid_render_mode, value=0,
                                                            command=lambda: self.change_setting("render", "grid", 0))
        self.grid_render_frame.grid(row=1, column=0, padx=5, pady=5)
        self.rb_grid_render_cube.grid(row=0, column=0)
        self.rb_grid_render_none.grid(row=1, column=0)

        self.edge_render_frame: LabelFrame = LabelFrame(self.render_frame, text="Edge", width=60,
                                                        padx=5, pady=5)
        self.edge_render_mode = IntVar(value=2)
        self.settings["render_edge"] = 2
        self.rb_edge_render_transparent: Radiobutton = Radiobutton(self.edge_render_frame, text="Transparent",
                                                                   variable=self.edge_render_mode, value=2,
                                                                   command=lambda: self.change_setting("render", "edge",
                                                                                                       2))
        self.rb_edge_render_sphere: Radiobutton = Radiobutton(self.edge_render_frame, text="Sphere",
                                                              variable=self.edge_render_mode, value=1,
                                                              command=lambda: self.change_setting("render", "edge", 1))
        self.rb_edge_render_none: Radiobutton = Radiobutton(self.edge_render_frame, text="None",
                                                            variable=self.edge_render_mode, value=0,
                                                            command=lambda: self.change_setting("render", "edge", 0))
        self.edge_render_frame.grid(row=2, column=0, padx=5, pady=5)
        self.rb_edge_render_transparent.grid(row=0, column=0)
        self.rb_edge_render_sphere.grid(row=1, column=0)
        self.rb_edge_render_none.grid(row=2, column=0)

        self.action_frame: LabelFrame = LabelFrame(self.gui_root, text="Settings", width=60,
                                                   padx=5, pady=5)
        self.action_frame.grid(row=0, column=2, rowspan=2, padx=5, pady=5)
        self.generate_button: Button = Button(self.action_frame, text="Generate", width=10, command=self.generate)
        self.generate_button.grid(row=0, column=0)
        self.action_state: IntVar = IntVar(0)
        self.action_buttons: RadioButtons = RadioButtons(self.action_frame, ["Freeze", "Advect", "Diverge", "Random"],
                                                         self.action_state, command=self.change_setting, row=1,
                                                         column=0)

        self.setting_frame: LabelFrame = LabelFrame(self.gui_root, text="Settings", width=60,
                                                    padx=5, pady=5)
        self.setting_frame.grid(row=2, column=2, padx=5, pady=5)
        self.layer_distance: SettingEntry = SettingEntry(self.setting_frame, "Layer distance:", row=0, column=0,
                                                         variable_type="float")
        self.neuron_size: SettingEntry = SettingEntry(self.setting_frame, "Neuron size:", row=1, column=0,
                                                      variable_type="float")
        self.sampling_rate: SettingEntry = SettingEntry(self.setting_frame, "Sampling rate:", row=2, column=0,
                                                        variable_type="float")

    def start(self, layer_data: List[int] = None, layer_distance: float = 1.0, node_size: float = 0.3,
              sampling_rate: float = 10.0):
        if layer_data is None:
            default_layer_data = [4, 9, 4]
            for nodes in default_layer_data:
                self.add_layer(nodes)
        else:
            for nodes in layer_data:
                self.add_layer(nodes)

        self.layer_distance.set(layer_distance)
        self.neuron_size.set(node_size)
        self.sampling_rate.set(sampling_rate)
        self.generate()

        self.gui_root.mainloop()
        self.settings["Closed"] = True

    def add_layer(self, nodes: int = 9):
        layer_id: int = len(self.layer_settings)
        self.layer_settings.append(LayerSettings(self.architecture_frame, layer_id, 1, 0, self.remove_layer))
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

    def generate(self):
        layer_data: List[int] = []
        for ls in self.layer_settings:
            layer_data.append(ls.get_neurons())
        self.settings["current_layer_data"] = layer_data
        self.settings["layer_distance"] = self.layer_distance.get()
        self.settings["node_size"] = self.neuron_size.get()
        self.settings["sampling_rate"] = self.sampling_rate.get()
        print("Generated network: " + str(layer_data))

    def change_setting(self, setting_type: str, sub_type: str, value: int):
        self.settings[setting_type + "_" + sub_type] = value
