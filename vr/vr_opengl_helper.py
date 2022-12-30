from OpenGL.GL import (GL_COLOR_ATTACHMENT0, GL_DEPTH24_STENCIL8,
                       GL_DEPTH_STENCIL_ATTACHMENT, GL_FRAMEBUFFER, GL_LINEAR,
                       GL_RENDERBUFFER, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                       GL_TEXTURE_MIN_FILTER, glBindFramebuffer,
                       glBindRenderbuffer, glFramebufferRenderbuffer,
                       glFramebufferTexture2D, glRenderbufferStorage,
                       glTexParameteri)

from opengl_helper.frame_buffer import FrameBufferObject
from opengl_helper.texture import Texture


class VRFrameBufferObject(FrameBufferObject):
    def __init__(self, width: int, height: int, texture: Texture) -> None:
        self.texture: Texture = texture
        super().__init__(width, height)

    def load(self) -> None:
        glBindFramebuffer(GL_FRAMEBUFFER, self.handle)

        glBindRenderbuffer(GL_RENDERBUFFER, self.depth_handle)
        glRenderbufferStorage(
            GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, self.width, self.height
        )
        glFramebufferRenderbuffer(
            GL_FRAMEBUFFER,
            GL_DEPTH_STENCIL_ATTACHMENT,
            GL_RENDERBUFFER,
            self.depth_handle,
        )

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

        glFramebufferTexture2D(
            GL_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D,
            self.texture.ogl_handle,
            0,
        )
        # glBindFramebuffer(GL_FRAMEBUFFER, 0)
