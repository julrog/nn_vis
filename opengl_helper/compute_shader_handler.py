import os
from typing import Dict, List

from definitions import BASE_PATH
from opengl_helper.compute_shader import ComputeShader
from utility.singleton import Singleton

SHADER_STATIC_VAR: List[str] = [
    'num_classes'
]

SHADER_DYNAMIC_VAR: List[str] = [
    'r_class_id',
    'r_edgebuffer_padding_id',
    'r_densitybuffer_padding_id',
    'r_nodebuffer_padding_id'
]


class ComputeShaderHandler(metaclass=Singleton):
    def __init__(self) -> None:
        self.shader_dir: str = os.path.join(BASE_PATH, 'shader_src/compute')
        self.shader_list: Dict[str, ComputeShader] = dict()
        self.num_classes: int = 10  # default value
        self.edgebuffer_padding: int = 0  # will be calculated
        self.densitybuffer_padding: int = 0  # will be calculated
        self.nodebuffer_padding: int = 0  # will be calculated
        self.static_var_map: Dict[str, str] = dict()

        self.set_classification_number(self.num_classes)

    def set_classification_number(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.static_var_map['$num_classes$'] = str(num_classes)
        self.edgebuffer_padding = (4 - ((self.num_classes * 2) % 4)) % 4
        self.densitybuffer_padding = (4 - ((self.num_classes + 1) % 4)) % 4
        self.nodebuffer_padding = (4 - ((self.num_classes + 2) % 4)) % 4
        self.shader_list = dict()

    def create(self, shader_name: str, shader_file_path: str) -> ComputeShader:
        if shader_name in self.shader_list.keys():
            return self.shader_list[shader_name]
        shader_src = self.get_processed_src(
            os.path.join(self.shader_dir, shader_file_path))
        self.shader_list[shader_name] = ComputeShader(shader_src)
        return self.shader_list[shader_name]

    def get(self, shader_name: str) -> ComputeShader:
        return self.shader_list[shader_name]

    def get_processed_src(self, path: str) -> str:
        processed_src: str = ''
        with open(path, 'r') as src:
            for line in src:
                processed_src = processed_src + self.process_line(line)
        return processed_src

    def process_line(self, line: str) -> str:
        parsed_lines: str = ''
        processed_line: str = line

        for static, value in self.static_var_map.items():
            processed_line = processed_line.replace(static, value)

        if '$$' in processed_line:
            new_line: str = ''
            added: bool = False

            for padding_id in range(self.edgebuffer_padding):
                new_line = processed_line
                added = False
                if '$r_edgebuffer_padding_id$' in new_line:
                    new_line = new_line.replace(
                        '$r_edgebuffer_padding_id$', str(padding_id))
                    added = True
                if added:
                    parsed_lines = parsed_lines + \
                        new_line.replace('//$$', '').replace('$$', '')

            for padding_id in range(self.densitybuffer_padding):
                new_line = processed_line
                added = False
                if '$r_densitybuffer_padding_id$' in new_line:
                    new_line = new_line.replace(
                        '$r_densitybuffer_padding_id$', str(padding_id))
                    added = True
                if added:
                    parsed_lines = parsed_lines + \
                        new_line.replace('//$$', '').replace('$$', '')

            for padding_id in range(self.nodebuffer_padding):
                new_line = processed_line
                added = False
                if '$r_nodebuffer_padding_id$' in new_line:
                    new_line = new_line.replace(
                        '$r_nodebuffer_padding_id$', str(padding_id))
                    added = True
                if added:
                    parsed_lines = parsed_lines + \
                        new_line.replace('//$$', '').replace('$$', '')

            for class_id in range(self.num_classes):
                new_line = processed_line
                added = False
                if '$r_class_id$' in new_line:
                    new_line = new_line.replace('$r_class_id$', str(class_id))
                    added = True
                if added:
                    parsed_lines = parsed_lines + \
                        new_line.replace('//$$', '').replace('$$', '')
        else:
            return processed_line.replace('//$', '')
        return parsed_lines
