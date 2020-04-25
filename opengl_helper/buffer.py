from OpenGL.GL import *


class BufferObject:
    def __init__(self, ssbo: bool = False):
        self.handle: int = glGenBuffers(1)
        self.location: int = 0
        self.ssbo: bool = ssbo
        if self.ssbo:
            self.size: int = 0
            self.max_ssbo_size: int = glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE)

    def load(self, data: any):
        glBindVertexArray(0)

        self.size = data.nbytes
        if self.ssbo:
            if data.nbytes > self.max_ssbo_size:
                raise Exception("Data to big for SSBO (%d bytes, max %d bytes)." % (data.nbytes, self.max_ssbo_size))

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.handle)
            glBufferData(GL_SHADER_STORAGE_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        else:
            glBindBuffer(GL_ARRAY_BUFFER, self.handle)
            glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)

    def read(self) -> any:
        if self.ssbo:
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.handle)
            return glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, self.size)

    def bind(self, location: int, rendering: bool = False):
        if self.ssbo:
            if rendering:
                glBindBuffer(GL_ARRAY_BUFFER, self.handle)
                glEnableVertexAttribArray(location)
                glVertexAttribPointer(location, 4, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
            else:
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, location, self.handle)
        else:
            glBindBuffer(GL_ARRAY_BUFFER, self.handle)
            glEnableVertexAttribArray(location)
            glVertexAttribPointer(location, 4, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))


class SwappingBufferObject(BufferObject):
    def __init__(self, ssbo: bool = False):
        super().__init__(ssbo)
        self.swap_handle: int = glGenBuffers(1)

    def swap(self):
        self.handle, self.swap_handle = self.swap_handle, self.handle

    def bind(self, location: int, rendering: bool = False):
        if self.ssbo:
            if rendering:
                glBindBuffer(GL_ARRAY_BUFFER, self.handle)
                glEnableVertexAttribArray(location)
                glVertexAttribPointer(location, 4, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
            else:
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, location, self.handle)
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, location + 1, self.swap_handle)
        else:
            glBindBuffer(GL_ARRAY_BUFFER, self.handle)
            glEnableVertexAttribArray(location)
            glVertexAttribPointer(location, 4, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
