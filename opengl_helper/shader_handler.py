import math
import os
from typing import Dict, List, Optional

from definitions import (ADDITIONAL_EDGE_BUFFER_DATA,
                         ADDITIONAL_NODE_BUFFER_DATA, BASE_PATH)
from opengl_helper.shader import RenderShader, ShaderSetting
from utility.singleton import Singleton

SHADER_STATIC_VAR: List[str] = [
    'num_classes',
    'nodebuffer_average',
    'nodebuffer_length',
    'edgebuffer_samples',
    'edgebuffer_layer',
    'edgebuffer_edge',
    'edgebuffer_importance',
    'edgebuffer_start_length',
    'edgebuffer_end_length',
    'edgebuffer_start_average',
    'edgebuffer_end_average'
]

SHADER_DYNAMIC_VAR: List[str] = [
    'r_nodebuffer_group_id',
    'r_nodebuffer_group_location',
    'r_edgebuffer_group_id',
    'r_edgebuffer_group_location',
    'r_class_color',
    'r_class_id',
    'r_nodebuffer_class_importance',
    'r_edgebuffer_start_class_importance',
    'r_edgebuffer_end_class_importance'
]

BUFFER_GROUP_VALUE: List[str] = ['x', 'y', 'z', 'w']

CLASS_COLOR: List[str] = [
    'vec3(0.133, 0.545, 0.133)',
    'vec3(0, 0, 0.545)',
    'vec3(0.69, 0.188, 0.376)',
    'vec3(1, 0.271, 0)',
    'vec3(1, 1, 0)',
    'vec3(0.871, 0.722, 0.529)',
    'vec3(0, 1, 0)',
    'vec3(0, 1, 1)',
    'vec3(1, 0, 1)',
    'vec3(0.392, 0.584, 0.929)'
]


def get_buffer_id(position: int) -> str:
    number: int = position % 4
    return str(int((position - number) / 4)) + '.' + BUFFER_GROUP_VALUE[number]


class RenderShaderHandler(metaclass=Singleton):
    def __init__(self) -> None:
        self.shader_dir: str = os.path.join(BASE_PATH, 'shader_src')
        self.shader_list: Dict[str, RenderShader] = dict()
        self.num_classes: int = 10  # default value
        self.static_var_map: Dict[str, str] = dict()

        self.set_classification_number(self.num_classes)

    def set_classification_number(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.static_var_map['$num_classes$'] = str(num_classes)
        self.static_var_map['$nodebuffer_average$'] = get_buffer_id(
            num_classes)
        self.static_var_map['$nodebuffer_length$'] = get_buffer_id(
            num_classes + 1)
        self.static_var_map['$edgebuffer_samples$'] = get_buffer_id(0)
        self.static_var_map['$edgebuffer_layer$'] = get_buffer_id(1)
        self.static_var_map['$edgebuffer_edge$'] = get_buffer_id(2)
        self.static_var_map['$edgebuffer_importance$'] = get_buffer_id(3)
        self.static_var_map['$edgebuffer_start_length$'] = get_buffer_id(4)
        self.static_var_map['$edgebuffer_end_length$'] = get_buffer_id(5)
        self.static_var_map['$edgebuffer_start_average$'] = get_buffer_id(6)
        self.static_var_map['$edgebuffer_end_average$'] = get_buffer_id(7)
        self.shader_list = dict()

    def create(self, shader_setting: ShaderSetting) -> RenderShader:
        if shader_setting.id_name in self.shader_list.keys():
            return self.shader_list[shader_setting.id_name]
        vertex_src: str = self.get_processed_src(
            os.path.join(self.shader_dir, shader_setting.vertex))
        fragment_src: str = self.get_processed_src(
            os.path.join(self.shader_dir, shader_setting.fragment))
        geometry_src: Optional[str] = None
        if shader_setting.geometry is not None:
            geometry_src = self.get_processed_src(
                os.path.join(self.shader_dir, shader_setting.geometry))
        self.shader_list[shader_setting.id_name] = RenderShader(vertex_src, fragment_src, geometry_src,
                                                                shader_setting.uniform_labels)
        return self.shader_list[shader_setting.id_name]

    def get(self, shader_name: str) -> RenderShader:
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

            for node_buffer_group in range(
                    int(math.ceil((self.num_classes + (ADDITIONAL_NODE_BUFFER_DATA - 4)) / 4.0))):
                new_line = processed_line
                added = False
                if '$r_nodebuffer_group_id$' in new_line:
                    new_line = new_line.replace(
                        '$r_nodebuffer_group_id$', str(node_buffer_group))
                    added = True
                if '$r_nodebuffer_group_location$' in new_line:
                    new_line = new_line.replace(
                        '$r_nodebuffer_group_location$', str(node_buffer_group + 1))
                    added = True
                if added:
                    parsed_lines = parsed_lines + \
                        new_line.replace('//$$', '').replace('$$', '')

            for edge_buffer_group in range(int(math.ceil((self.num_classes * 2 + ADDITIONAL_EDGE_BUFFER_DATA) / 4))):
                new_line = processed_line
                added = False
                if '$r_edgebuffer_group_id$' in new_line:
                    new_line = new_line.replace(
                        '$r_edgebuffer_group_id$', str(edge_buffer_group))
                    added = True
                if '$r_edgebuffer_group_location$' in new_line:
                    new_line = new_line.replace(
                        '$r_edgebuffer_group_location$', str(edge_buffer_group + 2))
                    added = True
                if added:
                    parsed_lines = parsed_lines + \
                        new_line.replace('//$$', '').replace('$$', '')

            for class_id in range(self.num_classes):
                new_line = processed_line
                added = False
                if '$r_class_color$' in new_line:
                    new_line = new_line.replace(
                        '$r_class_color$', CLASS_COLOR[class_id])
                    added = True
                if '$r_class_id$' in new_line:
                    new_line = new_line.replace('$r_class_id$', str(class_id))
                    added = True
                if '$r_nodebuffer_class_importance$' in new_line:
                    new_line = new_line.replace(
                        '$r_nodebuffer_class_importance$', get_buffer_id(class_id))
                    added = True
                if '$r_edgebuffer_start_class_importance$' in new_line:
                    new_line = new_line.replace('$r_edgebuffer_start_class_importance$',
                                                get_buffer_id(class_id + ADDITIONAL_EDGE_BUFFER_DATA))
                    added = True
                if '$r_edgebuffer_end_class_importance$' in new_line:
                    new_line = new_line.replace('$r_edgebuffer_end_class_importance$',
                                                get_buffer_id(
                                                    class_id + self.num_classes + ADDITIONAL_EDGE_BUFFER_DATA))
                    added = True
                if added:
                    parsed_lines = parsed_lines + \
                        new_line.replace('//$$', '').replace('$$', '')
        else:
            return processed_line.replace('//$', '')
        return parsed_lines
