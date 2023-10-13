import math
from typing import List, Optional

import numpy as np
import pyrr
from pyrr import Matrix44, Vector3, Vector4

from utility.camera import BaseCamera

INITIAL_SCALE: float = 0.1


def convert_projection_matrix(proj: np.ndarray) -> np.ndarray:
    conv_proj = np.array(
        (
            (proj[0][0], proj[1][0], proj[2][0], proj[3][0]),
            (proj[0][1], proj[1][1], proj[2][1], proj[3][1]),
            (proj[0][2], proj[1][2], proj[2][2], proj[3][2]),
            (proj[0][3], proj[1][3], proj[2][3], proj[3][3]),
        ),
        dtype=np.float32,
    )
    return conv_proj


def convert_view_matrix(pose: np.ndarray) -> np.ndarray:
    return np.array(
        (
            (pose[0][0], pose[1][0], pose[2][0], 0.0),
            (pose[0][1], pose[1][1], pose[2][1], 0.0),
            (pose[0][2], pose[1][2], pose[2][2], 0.0),
            (pose[0][3], pose[1][3], pose[2][3], 1.0),
        ),
        dtype=np.float32,
    )


class VRCamera(BaseCamera):
    def __init__(
        self,
        width: int,
        height: int,
        initial_pos: Optional[Vector3] = None,
    ) -> None:
        super().__init__(width, height)
        self.base: Vector3 = Vector3([0.0, 0.0, 0.0])
        self.initial_offset: Vector3 = (
            Vector3([0.0, 0.0, -2.0]) if initial_pos is None else initial_pos
        )

        self.eye_view: Matrix44 = Matrix44.identity()
        self.head_to_camera: Matrix44 = Matrix44.identity()

        self.object_scale: float = INITIAL_SCALE
        self.translation: Optional[Matrix44] = None
        self.rotation: Matrix44 = Matrix44.identity()
        self.rotation_offset: Matrix44 = pyrr.matrix44.create_from_axis_rotation(
            [0.0, 1.0, 0.0], -math.pi / 2.0
        )
        self.input_rotation: Matrix44 = Matrix44.identity()
        self.input_rotation_angle: List[float] = [0.0, 0.0]

    def generate_view(self) -> None:
        if self.translation is None:
            self.translation = Matrix44.identity()
            self.translation[3][0] = self.base.x + self.initial_offset.x
            self.translation[3][1] = self.base.y + self.initial_offset.y
            self.translation[3][2] = self.base.z + self.initial_offset.z

        self.view = (
            self.eye_view
            * self.translation
            * self.rotation
            * self.input_rotation
            * self.rotation_offset
        )

    def set_position(self, pose: np.ndarray) -> None:
        pos: List[float] = [pose[0][3], pose[1][3], pose[2][3]]

        if self.translation is None:
            self.translation = Matrix44.identity()
        self.translation[3][0] = self.base.x + pos[0]
        self.translation[3][1] = self.base.y + pos[1]
        self.translation[3][2] = self.base.z + pos[2]

        rotation: np.ndarray = convert_view_matrix(pose)
        rotation[3] = [0, 0, 0, 1]
        self.rotation = Matrix44(rotation)

    def update_projection(self, projection: np.ndarray) -> None:
        self.projection: Matrix44 = Matrix44(
            convert_projection_matrix(projection))

    def update_eye_to_head(self, eye_to_head: np.ndarray) -> None:
        self.head_to_eye: Matrix44 = Matrix44(
            np.linalg.inv(convert_view_matrix(eye_to_head))
        )

    def update_head(self, head_to_world: np.ndarray) -> None:
        world_to_head = Matrix44(np.linalg.inv(
            convert_view_matrix(head_to_world)))
        self.eye_view = self.head_to_eye * world_to_head

    def apply_input(
        self, scaling: float, rotation: List[float], grabbed: bool, reset: bool = False
    ) -> None:
        self.object_scale *= scaling
        self.input_rotation_angle[0] = rotation[0] / \
            10.0  # make rotation slower
        self.input_rotation_angle[1] = rotation[1] / \
            10.0  # make rotation slower

        # change rotation matrix only with actual changes
        if abs(rotation[0]) > 0 or abs(rotation[1]) > 0:
            if grabbed:  # only rotate when grabbed
                # rotate around current view by using the inverse rotation matrix on the rotation (x-)axis
                rotated_axis = (
                    pyrr.matrix44.inverse(self.input_rotation)
                    * Vector4([1.0, 0.0, 0.0, 1.0])
                )[:3]
                self.input_rotation = self.input_rotation * Matrix44(
                    pyrr.matrix44.create_from_axis_rotation(
                        rotated_axis, -self.input_rotation_angle[1]
                    )
                )

                # rotate around current view by using the inverse rotation matrix on the rotation (z-)axis (z)
                rotated_axis = (
                    pyrr.matrix44.inverse(self.input_rotation)
                    * Vector4([0.0, 0.0, 1.0, 1.0])
                )[:3]
                self.input_rotation = self.input_rotation * Matrix44(
                    pyrr.matrix44.create_from_axis_rotation(
                        rotated_axis, -self.input_rotation_angle[0]
                    )
                )

        if reset:
            self.object_scale = INITIAL_SCALE
            self.rotation = Matrix44.identity()
            self.translation = None
            self.input_rotation = Matrix44.identity()
