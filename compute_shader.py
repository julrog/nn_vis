import os
from typing import Dict, Tuple, List

from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

from texture import Texture


def uniform_setter_function(uniform_setter: str):
    if uniform_setter is "float":
        def usf_single_float(location, data):
            glUniform1f(location, data)

        return usf_single_float


class ComputeShader:
    def __init__(self, shader_src: str):
        self.shader_handle: int = compileProgram(compileShader(shader_src, GL_COMPUTE_SHADER))
        self.textures: List[Tuple[Texture, str, int]] = []
        self.uniform_cache: Dict[str, Tuple[int, any, any]] = dict()

    def set_textures(self, textures: List[Tuple[Texture, str, int]]):
        self.textures: List[Tuple[Texture, str, int]] = textures

    def set_uniform_data(self, data: List[Tuple[str, any, any]]):
        glUseProgram(self.shader_handle)
        for uniform_name, uniform_data, uniform_setter in data:
            # TODO add check for update, to not always update uniform data
            uniform_location = glGetUniformLocation(self.shader_handle, uniform_name)
            self.uniform_cache[uniform_name] = (
                uniform_location, uniform_data, uniform_setter_function(uniform_setter))

    def use(self, width: int):
        for texture, flag, image_position in self.textures:
            texture.bind_as_image(flag, image_position)
        glUseProgram(self.shader_handle)

        for uniform_location, uniform_data, uniform_setter in self.uniform_cache.values():
            uniform_setter(uniform_location, uniform_data)

        glDispatchCompute(width, 1, 1)
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ComputeShaderHandler(metaclass=Singleton):
    def __init__(self):
        self.shader_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'shader/compute')
        self.shader_list = dict()

    def create(self, shader_name: str, shader_file_path: str) -> ComputeShader:
        shader_src = open(os.path.join(self.shader_dir, shader_file_path), 'r').read()
        self.shader_list[shader_name] = ComputeShader(shader_src)
        return self.shader_list[shader_name]

    def get(self, shader_name: str) -> ComputeShader:
        return self.shader_list[shader_name]


'''data = []
print(bool(glGetIntegeri_v))
glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, data)
print(data)
glGetIntegeri_v( GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &data); 1024
glGetIntegeri_v( GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &data); 64
glGetIntegeri_v( GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &data); 2.147.483.647
glGetIntegeri_v( GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &data); 65.535
glGetIntegeri_v( GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &data); 65.535
glGetIntegerv( GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &data); 1536
glGetIntegerv( GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, &data);'''
