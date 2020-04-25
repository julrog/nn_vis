import os
from typing import List, Tuple, Dict

from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

from definitions import BASE_PATH
from utility.singleton import Singleton
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
    raise Exception("[%s] Uniform setter function for '%s' not defined." % (LOG_SOURCE, uniform_setter))


class BaseShader:
    def __init__(self):
        self.shader_handle: int = 0
        self.textures: List[Tuple[Texture, str, int]] = []
        self.uniform_cache: Dict[str, Tuple[int, any, any]] = dict()

    def set_uniform_data(self, data: List[Tuple[str, any, any]]):
        program_is_set: bool = False
        for uniform_name, uniform_data, uniform_setter in data:
            if uniform_name not in self.uniform_cache.keys():
                if not program_is_set:
                    glUseProgram(self.shader_handle)
                    program_is_set = True
            uniform_location = glGetUniformLocation(self.shader_handle, uniform_name)
            if uniform_location != -1:
                self.uniform_cache[uniform_name] = (
                    uniform_location, uniform_data, uniform_setter_function(uniform_setter))
            else:
                print(["[%s] Uniform variable '%s' not used in shader_src." % (LOG_SOURCE, uniform_name)])

    def set_textures(self, textures: List[Tuple[Texture, str, int]]):
        self.textures: List[Tuple[Texture, str, int]] = textures

    def use(self):
        pass


class RenderShader(BaseShader):
    def __init__(self, vertex_src: str, fragment_src: str, geometry_src: str = None):
        BaseShader.__init__(self)
        if geometry_src is None:
            self.shader_handle = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                                                compileShader(fragment_src, GL_FRAGMENT_SHADER))
        else:
            self.shader_handle = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                                                compileShader(fragment_src, GL_FRAGMENT_SHADER),
                                                compileShader(geometry_src, GL_GEOMETRY_SHADER))

    def use(self):
        for texture, _, texture_position in self.textures:
            texture.bind_as_texture(texture_position)
        glUseProgram(self.shader_handle)

        for uniform_location, uniform_data, uniform_setter in self.uniform_cache.values():
            uniform_setter(uniform_location, uniform_data)


class RenderShaderHandler(metaclass=Singleton):
    def __init__(self):
        self.shader_dir: str = os.path.join(BASE_PATH, 'shader_src')
        self.shader_list: Dict[str, RenderShader] = dict()

    def create(self, shader_name: str, vertex_file_path: str = None, fragment_file_path: str = None,
               geometry_file_path: str = None) -> RenderShader:
        vertex_src: str = open(os.path.join(self.shader_dir, vertex_file_path), 'r').read()
        fragment_src: str = open(os.path.join(self.shader_dir, fragment_file_path), 'r').read()
        geometry_src: str or None = None
        if geometry_file_path is not None:
            geometry_src = open(os.path.join(self.shader_dir, geometry_file_path), 'r').read()
        self.shader_list[shader_name] = RenderShader(vertex_src, fragment_src, geometry_src)
        return self.shader_list[shader_name]

    def get(self, shader_name: str) -> RenderShader:
        return self.shader_list[shader_name]
