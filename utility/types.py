from enum import IntEnum, IntFlag, auto


class ProcessRenderMode(IntFlag):
    FINAL = auto()
    NODE_ITERATIONS = auto()
    EDGE_ITERATIONS = auto()
    SMOOTHING = auto()


class CameraPose(IntEnum):
    FRONT = 2
    RIGHT = 3
    LEFT = 4
    LOWER_BACK_RIGHT = 5
    BACK_RIGHT = 6
    UPPER_BACK_LEFT = 7
    UPPER_BACK_RIGHT = 8