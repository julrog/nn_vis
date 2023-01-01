from typing import Any

from OpenGL.GL import (GL_COLOR_ATTACHMENT0, GL_DEPTH_ATTACHMENT,
                       GL_DEPTH_COMPONENT, GL_FRAMEBUFFER, GL_PACK_ALIGNMENT,
                       GL_RENDERBUFFER, GL_RGBA, GL_RGBA8, GL_UNSIGNED_BYTE,
                       glBindFramebuffer, glBindRenderbuffer,
                       glDeleteFramebuffers, glDeleteRenderbuffers,
                       glFramebufferRenderbuffer, glGenFramebuffers,
                       glGenRenderbuffers, glPixelStorei, glReadBuffer,
                       glReadPixels, glRenderbufferStorage)


class FrameBufferObject:
    def __init__(self, width: int, height: int) -> None:
        self.handle: int = glGenFramebuffers(1)
        self.color_handle: int = glGenRenderbuffers(1)
        self.depth_handle: int = glGenRenderbuffers(1)
        self.width: int = width
        self.height: int = height
        self.load()

    def load(self) -> None:
        glBindFramebuffer(GL_FRAMEBUFFER, self.handle)

        glBindRenderbuffer(GL_RENDERBUFFER, self.color_handle)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8,
                              self.width, self.height)
        glFramebufferRenderbuffer(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, self.color_handle)

        glBindRenderbuffer(GL_RENDERBUFFER, self.depth_handle)
        glRenderbufferStorage(
            GL_RENDERBUFFER, GL_DEPTH_COMPONENT, self.width, self.height)
        glFramebufferRenderbuffer(
            GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.depth_handle)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def read(self) -> Any:
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        glReadBuffer(GL_COLOR_ATTACHMENT0)
        data = glReadPixels(0, 0, self.width, self.height,
                            GL_RGBA, GL_UNSIGNED_BYTE)
        return data

    def bind(self) -> None:
        glBindFramebuffer(GL_FRAMEBUFFER, self.handle)

    def delete(self) -> None:
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glDeleteRenderbuffers(1, [self.color_handle])
        glDeleteRenderbuffers(1, [self.depth_handle])
        glDeleteFramebuffers(1, [self.handle])
