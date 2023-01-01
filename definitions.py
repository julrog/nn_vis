import os
from enum import IntEnum, IntFlag, auto
from typing import Generator, Iterable, List

from pyrr import Vector3

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = BASE_PATH + '/storage/data/'
SCREENSHOT_PATH = BASE_PATH + '/storage/screenshots/'
ADDITIONAL_NODE_BUFFER_DATA: int = 6
ADDITIONAL_EDGE_BUFFER_DATA: int = 8


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
    BACK = 7
    DEFAULT = 8


CAMERA_POSE_POSITION: List[Vector3] = [
    Vector3([3.5, 0.0, 0.0]),
    Vector3([0.0, 0.0, 2.5]),
    Vector3([0.0, 0.0, -2.5]),
    Vector3([-2.75, -1.0, 1.25]),
    Vector3([-2.5, 0.0, 2.5]),
    Vector3([-2.0, 2.0, -2.0]),
    Vector3([-2.0, 2.0, 2.0]),
    Vector3([-4.0, 0.0, 0.0]),
    Vector3([-3.5, 0.0, 0.0])
]


def pairwise(it: Iterable, size: int) -> Generator:
    it = iter(it)
    while True:
        try:
            yield next(it)
            for _ in range(size - 1):
                next(it)
        except StopIteration:
            return


def vec4wise(it: Iterable) -> Generator:
    it = iter(it)
    while True:
        try:
            yield next(it), next(it), next(it), next(it),
        except StopIteration:
            return
