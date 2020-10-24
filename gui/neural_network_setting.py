from tkinter import LabelFrame, Button, Entry, Label, END


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