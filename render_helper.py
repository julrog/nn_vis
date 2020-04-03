from OpenGL.GL import *


class VertexDataHandler:
    def __init__(self):
        self.VAO = glGenVertexArrays(1)
        self.VBO = glGenBuffers(1)

    def load_data(self, data):
        self.set()

        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)

        # cube vertices
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, data.itemsize * 3, ctypes.c_void_p(0))

    def set(self):
        glBindVertexArray(self.VAO)