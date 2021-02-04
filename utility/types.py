from enum import IntEnum, IntFlag, auto
from typing import List

from pyrr import Vector3


class ProcessRenderMode(IntFlag):
    FINAL = auto()
    NODE_ITERATIONS = auto()
    EDGE_ITERATIONS = auto()
    SMOOTHING = auto()


class CameraPose(IntEnum):
    FRONT = 0
    RIGHT = 1
    LEFT = 2
    LOWER_BACK_RIGHT = 3
    BACK_RIGHT = 4
    UPPER_BACK_LEFT = 5
    UPPER_BACK_RIGHT = 6
    DEFAULT = 7


CAMERA_POSE_POSITION: List[Vector3] = [
    Vector3([3.0, 0.0, 0.0]),       #z
    Vector3([0.0, 0.0, 2.5]),       #x
    Vector3([0.0, 0.0, -2.5]),      #xz
    Vector3([-2.75, -1.0, 1.25]),   #x
    Vector3([-2.5, 0.0, 2.5]),      #x
    Vector3([-2.0, 2.0, -2.0]),     #xz
    Vector3([-2.0, 2.0, 2.0]),      #x
    Vector3([-3.5, 0.0, 0.0])
]
