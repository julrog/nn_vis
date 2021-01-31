from OpenGL.GL import *

from utility.singleton import Singleton


class Texture:
    def __init__(self, width: int, height: int):
        self.texture_handler: TextureHandler = TextureHandler()
        self.width: int = width
        self.height: int = height
        self.active_index: int or None = None
        self.ogl_handle: int = glGenTextures(1)
        self.texture_position: int = -1
        self.image_position: int = -1

    def setup(self, position: int = None, data=None):
        self.bind_as_texture(position)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, self.width, self.height, 0, GL_RGBA, GL_FLOAT, data)

    def bind_as_texture(self, position=None):
        if position is None:
            if self.texture_position is -1:
                raise Exception("No texture position configured.")
        else:
            self.texture_position = position
        self.texture_handler.activate(self.texture_position)
        glBindTexture(GL_TEXTURE_2D, self.ogl_handle)

    def bind_as_image(self, flag: str, position=None):
        if position is None:
            if self.image_position is -1:
                raise Exception("No image position configured!")
        else:
            self.image_position = position
        ogl_flag = GL_WRITE_ONLY if flag is "write" else GL_READ_ONLY if flag is "read" else GL_READ_WRITE
        glBindImageTexture(self.image_position, self.ogl_handle, 0, GL_FALSE, 0, ogl_flag, GL_RGBA32F)

    def read(self):
        self.bind_as_texture()
        data = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT)
        return data

    def delete(self):
        glDeleteTextures(1, [self.ogl_handle])


class TextureHandler(metaclass=Singleton):
    def __init__(self):
        self.max_textures: int = glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS)

    def activate(self, position: int):
        if position < 0 or position > self.max_textures:
            raise Exception("OGL Texture position '%d' not available." % position)
        glActiveTexture(GL_TEXTURE0 + position)
