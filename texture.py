from typing import List

from OpenGL.GL import *


class Texture:
    def __init__(self, width: int, height: int):
        self.texture_handler: TextureHandler = TextureHandler()
        self.width: int = width
        self.height: int = height
        self.active_index: int or None = None
        self.ogl_handle: int = glGenTextures(1)

    def activate(self):
        self.texture_handler.activate(self)

    def setup(self, data=None):
        self.activate()
        glBindTexture(GL_TEXTURE_2D, self.ogl_handle)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, self.width, self.height, 0, GL_RGBA, GL_FLOAT, data)

    def bind(self):
        glBindImageTexture(self.active_index, self.ogl_handle, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F)

    def read(self):
        self.activate()
        glBindTexture(GL_TEXTURE_2D, self.ogl_handle)
        data = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT)
        self.deactivate()
        return data

    def deactivate(self):
        self.texture_handler.deactivate(self)

    def delete(self):
        texture_handler: TextureHandler = TextureHandler()
        texture_handler.deactivate(self)
        glDeleteTextures(1, self.ogl_handle)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class TextureHandler(metaclass=Singleton):
    def __init__(self):
        self.max_textures: int = glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS)
        self.active_textures: List[Texture] = []

    def activate(self, texture: Texture) -> Texture:
        if texture.active_index is not None:
            glActiveTexture(GL_TEXTURE0 + texture.active_index)
            return texture

        texture_active_slots = [texture.active_index for texture in self.active_textures]
        for i in range(self.max_textures):
            if i not in texture_active_slots:
                texture.active_index = i
                self.active_textures.append(texture)
                glActiveTexture(GL_TEXTURE0 + texture.active_index)
                return texture
        raise Exception("[TEXTURE] Can't activate texture! No free texture slot left.")

    def deactivate(self, texture: Texture):
        if texture.active_index is not None:
            texture.active_index = None
            self.active_textures.remove(texture)
