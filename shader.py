import os
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader


class Shader:
    def __init__(self, vertex_src, fragment_src):
        self.shader_handle = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                                            compileShader(fragment_src, GL_FRAGMENT_SHADER))


class ShaderHandler:
    def __init__(self):
        self.shader_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'shader')
        self.shader_list = dict()

    def create(self, shader_name: str, vertex_file_path: str = None, fragment_file_path: str = None) -> Shader:
        vertex_src = open(os.path.join(self.shader_dir, vertex_file_path), 'r').read()
        fragment_src = open(os.path.join(self.shader_dir, fragment_file_path), 'r').read()
        self.shader_list[shader_name] = Shader(vertex_src, fragment_src)
        return self.shader_list[shader_name]
