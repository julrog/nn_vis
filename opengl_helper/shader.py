from typing import Any, Callable, Dict, List, Optional, Tuple

from OpenGL.GL import (GL_FALSE, GL_FRAGMENT_SHADER, GL_GEOMETRY_SHADER,
                       GL_VERTEX_SHADER, glGetUniformLocation, glUniform1f,
                       glUniform1i, glUniform3fv, glUniform3iv,
                       glUniformMatrix4fv, glUseProgram)
from OpenGL.GL.shaders import compileProgram, compileShader

from opengl_helper.texture import Texture
from rendering.rendering_config import RenderingConfig


def uniform_setter_function(uniform_setter: str) -> Callable:
    if uniform_setter == 'float':

        def uniform_func_1f(location: int, data: float) -> None:
            glUniform1f(location, data)

        return uniform_func_1f
    if uniform_setter == 'vec3':

        def uniform_func_3fv(location: int, data: List[float]) -> None:
            glUniform3fv(location, 1, data)

        return uniform_func_3fv
    if uniform_setter == 'mat4':

        def uniform_func_4fv(location: int, data: List[float]) -> None:
            glUniformMatrix4fv(location, 1, GL_FALSE, data)

        return uniform_func_4fv
    if uniform_setter == 'int':

        def uniform_func_1i(location: int, data: int) -> None:
            glUniform1i(location, data)

        return uniform_func_1i
    if uniform_setter == 'ivec3':

        def uniform_func_3iv(location: int, data: List[int]) -> None:
            glUniform3iv(location, 1, data)

        return uniform_func_3iv
    raise Exception(
        "Uniform setter function for '%s' not defined." % uniform_setter)


class ShaderSetting:
    def __init__(
        self, id_name: str, shader_paths: List[str], uniform_labels: Optional[List[str]] = None
    ) -> None:
        self.id_name: str = id_name
        if len(shader_paths) < 2 or len(shader_paths) > 3:
            raise Exception(
                "Can't handle number of shaders for a program (either 2 or 3 with geometry shader)."
            )
        self.vertex: str = shader_paths[0]
        self.fragment: str = shader_paths[1]
        self.geometry: Optional[str] = None if len(
            shader_paths) < 3 else shader_paths[2]
        self.uniform_labels: List[str] = (
            uniform_labels if uniform_labels is not None else []
        )


class BaseShader:
    def __init__(self) -> None:
        self.shader_handle: int = 0
        self.textures: List[Tuple[Texture, str, int]] = []
        self.uniform_cache: Dict[str, Tuple[int, Any, Callable]] = dict()
        self.uniform_labels: List[str] = []
        self.uniform_ignore_labels: List[str] = []

    def set_uniform_label(self, data: List[str]) -> None:
        for setting in data:
            self.uniform_labels.append(setting)

    def set_uniform_labeled_data(self, config: RenderingConfig) -> None:
        uniform_data = []
        for setting, shader_name in config.shader_name.items():
            if setting in self.uniform_labels:
                uniform_data.append(
                    (shader_name, config[setting], 'float'))
        self.set_uniform_data(uniform_data)

    def set_uniform_data(self, data: List[Tuple[str, Any, str]]) -> None:
        program_is_set: bool = False
        for uniform_name, uniform_data, uniform_setter in data:
            if uniform_name not in self.uniform_ignore_labels:
                if uniform_name not in self.uniform_cache.keys():
                    if not program_is_set:
                        glUseProgram(self.shader_handle)
                        program_is_set = True
                    uniform_location = glGetUniformLocation(
                        self.shader_handle, uniform_name
                    )
                    if uniform_location != -1:
                        self.uniform_cache[uniform_name] = (
                            uniform_location,
                            uniform_data,
                            uniform_setter_function(uniform_setter),
                        )
                    else:
                        self.uniform_ignore_labels.append(uniform_name)
                else:
                    uniform_location, _, setter = self.uniform_cache[uniform_name]
                    self.uniform_cache[uniform_name] = (
                        uniform_location,
                        uniform_data,
                        setter,
                    )

    def set_textures(self, textures: List[Tuple[Texture, str, int]]) -> None:
        self.textures = textures

    def use(self) -> None:
        pass


class RenderShader(BaseShader):
    def __init__(
        self,
        vertex_src: str,
        fragment_src: str,
        geometry_src: Optional[str] = None,
        uniform_labels: Optional[List[str]] = None,
    ) -> None:
        BaseShader.__init__(self)
        if geometry_src is None:
            self.shader_handle = compileProgram(
                compileShader(vertex_src, GL_VERTEX_SHADER),
                compileShader(fragment_src, GL_FRAGMENT_SHADER),
            )
        else:
            self.shader_handle = compileProgram(
                compileShader(vertex_src, GL_VERTEX_SHADER),
                compileShader(fragment_src, GL_FRAGMENT_SHADER),
                compileShader(geometry_src, GL_GEOMETRY_SHADER),
            )
        if uniform_labels is not None:
            self.set_uniform_label(uniform_labels)

    def use(self) -> None:
        for texture, _, texture_position in self.textures:
            texture.bind_as_texture(texture_position)
        glUseProgram(self.shader_handle)

        for (
            uniform_location,
            uniform_data,
            uniform_setter,
        ) in self.uniform_cache.values():
            uniform_setter(uniform_location, uniform_data)
