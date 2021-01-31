from OpenGL.GL import *


class FrameBufferObject:
    def __init__(self, width: int, height: int):
        self.handle: int = glGenFramebuffers(1)
        self.color_handle: int = glGenRenderbuffers(1)
        self.depth_handle: int = glGenRenderbuffers(1)
        self.width: int = width
        self.height: int = height
        self.load()

    def load(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.handle)

        glBindRenderbuffer(GL_RENDERBUFFER, self.color_handle)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, self.width, self.height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, self.color_handle)

        glBindRenderbuffer(GL_RENDERBUFFER, self.depth_handle)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, self.width, self.height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.depth_handle)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def read(self) -> any:
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        glReadBuffer(GL_COLOR_ATTACHMENT0)
        data = glReadPixels(0, 0, self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE)
        return data

    def bind(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.handle)

    def delete(self):
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glDeleteRenderbuffers(1, [self.color_handle])
        glDeleteRenderbuffers(1, [self.depth_handle])
        glDeleteFramebuffers(1, [self.handle])
