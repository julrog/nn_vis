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


class OptionGui:
    def __init__(self):
        self.test: bool = False

        self.gui_root: Tk = Tk()
        self.layer_settings: List[LayerSettings] = []
        self.settings: Dict[str, any] = {"Closed": False, "current_layer_data": []}

        self.gui_root.title("NNVIS Options")

        self.architecture_frame: LabelFrame = LabelFrame(self.gui_root, text="Neural Network Architecture", width=60,
                                                         padx=5, pady=5)
        self.architecture_frame.grid(row=0, column=0, rowspan=2, padx=5, pady=5)

        self.layer_label: Label = Label(self.architecture_frame, text="Actions:")
        self.add_layer_button: Button = Button(self.architecture_frame, text="Add Layer", command=self.add_layer)
        self.clear_layer_button: Button = Button(self.architecture_frame, text="Clear Layer", command=self.clear_layer)
        self.generate_button: Button = Button(self.architecture_frame, text="Generate", command=self.generate)
        self.layer_label.grid(row=0, column=0)
        self.add_layer_button.grid(row=0, column=1)
        self.clear_layer_button.grid(row=0, column=2)
        self.generate_button.grid(row=0, column=3)

        self.stats_frame: LabelFrame = LabelFrame(self.gui_root, text="Statistics", width=60,
                                                  padx=5, pady=5)

        self.stats_frame.grid(row=0, column=1, padx=5, pady=5)
        self.edge_count: StringVar = StringVar(value="0")
        self.settings["edge_count"] = self.edge_count
        self.stat_edge_text_label: Label = Label(self.stats_frame, text="Edges:")
        self.stat_edge_text_label.grid(row=0, column=0)
        self.stat_edge_count_label: Label = Label(self.stats_frame, textvariable=self.edge_count)
        self.stat_edge_count_label.grid(row=0, column=1)

        self.sample_count: StringVar = StringVar(value="0")
        self.settings["sample_count"] = self.sample_count
        self.stat_sample_text_label: Label = Label(self.stats_frame, text="Samples:")
        self.stat_sample_text_label.grid(row=1, column=0)
        self.stat_sample_count_label: Label = Label(self.stats_frame, textvariable=self.sample_count)
        self.stat_sample_count_label.grid(row=1, column=1)

        self.cell_count: StringVar = StringVar(value="0")
        self.settings["cell_count"] = self.cell_count
        self.stat_sample_text_label: Label = Label(self.stats_frame, text="Grid Cells:")
        self.stat_sample_text_label.grid(row=2, column=0)
        self.stat_sample_count_label: Label = Label(self.stats_frame, textvariable=self.cell_count)
        self.stat_sample_count_label.grid(row=2, column=1)

        self.render_frame: LabelFrame = LabelFrame(self.gui_root, text="Render Settings", width=60,
                                                   padx=5, pady=5)

        self.render_frame.grid(row=1, column=1, padx=5, pady=5)
        self.render_info_label: Label = Label(self.render_frame, text="FPS: ")
        self.render_info_label.grid(row=0, column=0)

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

    def start(self, layer_data: List[int] = None):
        if layer_data is None:
            default_layer_data = [4, 9, 4]
            for nodes in default_layer_data:
                self.add_layer(nodes)
            self.settings["current_layer_data"] = default_layer_data
        else:
            for nodes in layer_data:
                self.add_layer(nodes)
            self.settings["current_layer_data"] = layer_data
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
        self.test = True
        print("Generated network: " + str(layer_data))

    def change_setting(self, setting_type: str, render_object: str, value: int):
        self.settings[setting_type + "_" + render_object] = value
