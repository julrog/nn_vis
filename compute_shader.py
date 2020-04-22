import os
from typing import Dict, Tuple, List

from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

from shader import BaseShader
from singleton import Singleton
from texture import Texture


class ComputeShader(BaseShader):
    def __init__(self, shader_src: str):
        BaseShader.__init__(self)
        self.shader_handle: int = compileProgram(compileShader(shader_src, GL_COMPUTE_SHADER))
        self.textures: List[Tuple[Texture, str, int]] = []
        self.uniform_cache: Dict[str, Tuple[int, any, any]] = dict()
        self.max_workgroup_size: int = glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0)[0]

    def compute(self, width: int):
        if width > self.max_workgroup_size:
            raise Exception("Workgroup size is too big!")
        for texture, flag, image_position in self.textures:
            texture.bind_as_image(flag, image_position)
        glUseProgram(self.shader_handle)

        for uniform_location, uniform_data, uniform_setter in self.uniform_cache.values():
            uniform_setter(uniform_location, uniform_data)

        glDispatchCompute(width, 1, 1)
        glMemoryBarrier(GL_ALL_BARRIER_BITS)


class ComputeShaderHandler(metaclass=Singleton):
    def __init__(self):
        self.shader_dir: str = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'shader/compute')
        self.shader_list: Dict[str, ComputeShader] = dict()

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
