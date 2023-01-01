import math
from typing import Any, Dict, List, Tuple

from OpenGL.GL import (GL_ALL_BARRIER_BITS, GL_COMPUTE_SHADER,
                       GL_MAX_COMPUTE_WORK_GROUP_COUNT, glDispatchCompute,
                       glGetIntegeri_v, glMemoryBarrier, glUseProgram)
from OpenGL.GL.shaders import compileProgram, compileShader

from opengl_helper.shader import BaseShader
from opengl_helper.texture import Texture


class ComputeShader(BaseShader):
    def __init__(self, shader_src: str) -> None:
        BaseShader.__init__(self)
        self.shader_handle: int = compileProgram(
            compileShader(shader_src, GL_COMPUTE_SHADER))
        self.textures: List[Tuple[Texture, str, int]] = []
        self.uniform_cache: Dict[str, Tuple[int, Any, Any]] = dict()
        self.max_workgroup_size: int = glGetIntegeri_v(
            GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0)[0]

    def compute(self, width: int, barrier: bool = False) -> None:
        for i in range(math.ceil(width / self.max_workgroup_size)):
            self.set_uniform_data(
                [('work_group_offset', i * self.max_workgroup_size, 'int')])

            for texture, flag, image_position in self.textures:
                texture.bind_as_image(flag, image_position)
            glUseProgram(self.shader_handle)

            for uniform_location, uniform_data, uniform_setter in self.uniform_cache.values():
                uniform_setter(uniform_location, uniform_data)

            if i == math.ceil(width / self.max_workgroup_size) - 1:
                glDispatchCompute(width % self.max_workgroup_size, 1, 1)
            else:
                glDispatchCompute(self.max_workgroup_size, 1, 1)
        if barrier:
            self.barrier()

    @staticmethod
    def barrier() -> None:
        glMemoryBarrier(GL_ALL_BARRIER_BITS)
