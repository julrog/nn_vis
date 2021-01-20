import os
from typing import Dict, List

from opengl_helper.shader import RenderShader, ShaderSetting
from utility.singleton import Singleton
from definitions import BASE_PATH

SHADER_STATIC_VAR: List[str] = [
    'num_classes',
    'nodebuffer_average',
    'nodebuffer_length',
    'nodebuffer_samples',
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


class RenderShaderHandler(metaclass=Singleton):
    def __init__(self):
        self.shader_dir: str = os.path.join(BASE_PATH, 'shader_src')
        self.shader_list: Dict[str, RenderShader] = dict()
        self.num_classes: int = 10  # default value
        self.static_var_map: Dict[str, str] = dict()
        self.node_buffer_group_count: int = 0
        self.class_color: Dict[int, str] = dict()
        self.nb_importance: Dict[int, str] = dict()
        self.eb_start_importance: Dict[int, str] = dict()
        self.eb_end_importance: Dict[int, str] = dict()

        self.set_classification_number(self.num_classes)

    def set_classification_number(self, num_classes: int):
        self.num_classes: int = num_classes

    def create(self, shader_setting: ShaderSetting) -> RenderShader:
        if shader_setting.id_name in self.shader_list.keys():
            return self.shader_list[shader_setting.id_name]
        vertex_src: str = open(os.path.join(self.shader_dir, shader_setting.vertex), 'r').read()
        fragment_src: str = open(os.path.join(self.shader_dir, shader_setting.fragment), 'r').read()
        geometry_src: str or None = None
        if shader_setting.geometry is not None:
            geometry_src = open(os.path.join(self.shader_dir, shader_setting.geometry), 'r').read()
        self.shader_list[shader_setting.id_name] = RenderShader(vertex_src, fragment_src, geometry_src,
                                                                shader_setting.uniform_labels)
        return self.shader_list[shader_setting.id_name]

    def get(self, shader_name: str) -> RenderShader:
        return self.shader_list[shader_name]

    def get_processed_src(self, path) -> str:
        processed_src: str = ""
        with open(path, 'r') as src:

        return processed_src
