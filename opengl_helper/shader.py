from typing import List, Tuple, Dict
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from rendering.rendering_config import RenderingConfig
from opengl_helper.texture import Texture

LOG_SOURCE: str = "SHADER"


def uniform_setter_function(uniform_setter: str):
    if uniform_setter is "float":
        def uniform_func(location, data):
            glUniform1f(location, data)

        return uniform_func
    if uniform_setter is "vec3":
        def uniform_func(location, data):
            glUniform3fv(location, 1, data)

        return uniform_func
    if uniform_setter is "mat4":
        def uniform_func(location, data):
            glUniformMatrix4fv(location, 1, GL_FALSE, data)

        return uniform_func
    if uniform_setter is "int":
        def uniform_func(location, data):
            glUniform1i(location, data)

        return uniform_func
    if uniform_setter is "ivec3":
        def uniform_func(location, data):
            glUniform3iv(location, 1, data)

        return uniform_func
    raise Exception("[%s] Uniform setter function for '%s' not defined." % (LOG_SOURCE, uniform_setter))


class ShaderSetting:
    def __init__(self, id_name: str, shader_paths: List[str], uniform_labels: List[str] = None):
        self.id_name: str = id_name
        if len(shader_paths) < 2 or len(shader_paths) > 3:
            raise Exception(
                "[%s] Can't handle number of shaders for a program (either 2 or 3 with geometry shader)." % LOG_SOURCE)
        self.vertex: str = shader_paths[0]
        self.fragment: str = shader_paths[1]
        self.geometry: str = None if len(shader_paths) < 3 else shader_paths[2]
        self.uniform_labels: List[str] = uniform_labels if uniform_labels is not None else []


class BaseShader:
    def __init__(self):
        self.shader_handle: int = 0
        self.textures: List[Tuple[Texture, str, int]] = []
        self.uniform_cache: Dict[str, Tuple[int, any, any]] = dict()
        self.uniform_labels: List[str] = []
        self.uniform_ignore_labels: List[str] = []

    def set_uniform_label(self, data: List[str]):
        for setting in data:
            self.uniform_labels.append(setting)

    def set_uniform_labeled_data(self, config: RenderingConfig):
        if config is not None:
            uniform_data = []
            for setting, shader_name in config.shader_name.items():
                if setting in self.uniform_labels:
                    uniform_data.append((shader_name, config[setting], "float"))
            self.set_uniform_data(uniform_data)

    def set_uniform_data(self, data: List[Tuple[str, any, any]]):
        program_is_set: bool = False
        for uniform_name, uniform_data, uniform_setter in data:
            if uniform_name not in self.uniform_ignore_labels:
                if uniform_name not in self.uniform_cache.keys():
                    if not program_is_set:
                        glUseProgram(self.shader_handle)
                        program_is_set = True
                    uniform_location = glGetUniformLocation(self.shader_handle, uniform_name)
                    if uniform_location != -1:
                        self.uniform_cache[uniform_name] = (
                            uniform_location, uniform_data, uniform_setter_function(uniform_setter))
                    else:
                        self.uniform_ignore_labels.append(uniform_name)
                else:
                    uniform_location, _, setter = self.uniform_cache[uniform_name]
                    self.uniform_cache[uniform_name] = (uniform_location, uniform_data, setter)

    def set_textures(self, textures: List[Tuple[Texture, str, int]]):
        self.textures: List[Tuple[Texture, str, int]] = textures

    def use(self):
        pass


class RenderShader(BaseShader):
    def __init__(self, vertex_src: str, fragment_src: str, geometry_src: str = None,
                 uniform_labels: List[str] = None):
        BaseShader.__init__(self)
        if geometry_src is None:
            self.shader_handle = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                                                compileShader(fragment_src, GL_FRAGMENT_SHADER))
        else:
            self.shader_handle = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                                                compileShader(fragment_src, GL_FRAGMENT_SHADER),
                                                compileShader(geometry_src, GL_GEOMETRY_SHADER))
        if uniform_labels is not None:
            self.set_uniform_label(uniform_labels)

    def use(self):
        for texture, _, texture_position in self.textures:
            texture.bind_as_texture(texture_position)
        glUseProgram(self.shader_handle)

        for uniform_location, uniform_data, uniform_setter in self.uniform_cache.values():
            uniform_setter(uniform_location, uniform_data)
