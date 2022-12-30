import openvr
from OpenGL.GL import GL_RGBA8, GL_UNSIGNED_BYTE

from opengl_helper.texture import Texture
from vr.vr_opengl_helper import VRFrameBufferObject


class VRRenderTarget:
    def __init__(
        self,
        vr_eye_id: int,
        eye_id: int,
        width: int,
        height: int,
    ) -> None:
        self.vr_eye_id: int = vr_eye_id
        self.eye_id: int = eye_id
        self.texture: Texture = Texture(width, height)
        self.texture.setup(eye_id, internalformat=GL_RGBA8,
                           data_type=GL_UNSIGNED_BYTE)
        self.frame_buffer: VRFrameBufferObject = VRFrameBufferObject(
            width, height, self.texture
        )

        self.vr_texture: openvr.Texture_t = openvr.Texture_t()
        self.vr_texture.handle = int(self.texture.ogl_handle)
        self.vr_texture.eType = openvr.TextureType_OpenGL
        self.vr_texture.eColorSpace = openvr.ColorSpace_Gamma
