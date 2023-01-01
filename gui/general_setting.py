import logging
from tkinter import (RAISED, SUNKEN, Button, DoubleVar, Entry, IntVar, Label,
                     LabelFrame, StringVar, TclError)
from typing import Any, Callable, List, Union


class SettingBase:
    def __init__(self, root: LabelFrame, name: str, variable_value: Any = 0, variable_type: str = 'string',
                 row: int = 0, column: int = 0) -> None:
        self.name: str = name
        self.variable_type: str = variable_type
        self.variable: Union[IntVar, DoubleVar, StringVar]
        if variable_type == 'int':
            self.variable = IntVar(value=int(variable_value))
        elif variable_type == 'float':
            self.variable = DoubleVar(value=float(variable_value))
        else:
            self.variable = StringVar(value=str(variable_value))
        self.label: Label = Label(root, text=name)
        self.label.grid(row=row, column=column)

    def set(self, value: Any) -> None:
        if isinstance(self.variable, IntVar):
            self.variable.set(int(value))
        elif isinstance(self.variable, DoubleVar):
            self.variable.set(float(value))
        else:
            self.variable.set(str(value))


class SettingField(SettingBase):
    def __init__(self, root: LabelFrame, name: str, variable_value: Any = 0, variable_type: str = 'string',
                 row: int = 0, column: int = 0) -> None:
        super().__init__(root, name, variable_value, variable_type, row, column)
        self.variable_field: Label = Label(root, textvariable=self.variable)
        self.variable_field.grid(row=row, column=column + 1)


class SettingEntry(SettingBase):
    def __init__(self, root: LabelFrame, name: str, variable_value: Any = 0, variable_type: str = 'string',
                 row: int = 0, column: int = 0) -> None:
        super().__init__(root, name, variable_value, variable_type, row, column)
        self.old_variable: Union[IntVar, DoubleVar, StringVar]
        if variable_type == 'int':
            self.old_variable = IntVar(value=int(variable_value))
        elif variable_type == 'float':
            self.old_variable = DoubleVar(
                value=float(variable_value))
        else:
            self.old_variable = StringVar(value=str(variable_value))
        self.variable_entry: Entry = Entry(
            root, width=5, textvariable=self.variable)
        self.variable_entry.grid(row=row, column=column + 1)

    def set(self, value: Any) -> None:
        super().set(value)
        if isinstance(self.old_variable, IntVar):
            self.old_variable.set(int(value))
        elif isinstance(self.old_variable, DoubleVar):
            self.old_variable.set(float(value))
        else:
            self.old_variable.set(str(value))

    def get(self) -> Any:
        try:
            if isinstance(self.variable, IntVar):
                return int(self.variable.get())
            elif isinstance(self.variable, DoubleVar):
                return float(self.variable.get())
            else:
                return str(self.variable.get())
        except TclError:
            self.variable.set(self.old_variable.get())  # type: ignore
            logging.error('wrong value entered.. using old one')
            if isinstance(self.variable, IntVar):
                return int(self.variable.get())
            elif isinstance(self.variable, DoubleVar):
                return float(self.variable.get())
            else:
                return str(self.variable.get())


class RadioButtons:
    def __init__(self, root: LabelFrame, names: List[str], shared_variable: IntVar, command: Callable, option: str,
                 sub_option: str, row: int = 0, column: int = 0, width: int = 15, height: int = 1) -> None:
        self.root: LabelFrame = root
        self.row: int = row
        self.column: int = column
        self.width: int = width
        self.height: int = height
        self.names: List[str] = names
        self.option: str = option
        self.sub_option: str = sub_option
        self.variable: IntVar = shared_variable
        self.command: Callable = command
        self.buttons: List[Button] = []
        self.set_buttons(names)
        self.press(0)

    def press(self, button_id: int) -> None:
        self.set(button_id)
        for i, button in enumerate(self.buttons):
            if i is not button_id:
                button.config(relief=RAISED)
            else:
                button.config(relief=SUNKEN)
        self.command(self.option, self.sub_option, self.variable.get())

    def set(self, value: Any) -> None:
        self.variable.set(int(value))

    def get(self) -> Any:
        return int(self.variable.get())

    def set_buttons(self, button_names: List[str]) -> None:
        if self.buttons:
            for button in self.buttons:
                button.destroy()

        self.buttons = []
        self.names = button_names
        for i, name in enumerate(self.names):
            def generate_press_function(current_index: int) -> Callable:
                def press_function() -> None:
                    self.press(current_index)

                return press_function

            new_button: Button = Button(self.root, text=name, width=self.width, height=self.height,
                                        command=generate_press_function(i))
            new_button.grid(row=self.row + i, column=self.column)
            self.buttons.append(new_button)
