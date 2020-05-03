from typing import List

from pyrr import Vector4, Vector3


class Edge:
    def __init__(self, start_position: Vector3, end_position: Vector3):
        self.start: Vector4 = Vector4([start_position.x, start_position.y, start_position.z, 1.0])
        self.end: Vector4 = Vector4([end_position.x, end_position.y, end_position.z, 0.0])
        self.initial_data: List[float] = [start_position.x, start_position.y, start_position.z, 1.0, end_position.x,
                                          end_position.y, end_position.z, 0.0]
        self.sample_points: List[Vector4] = [self.start, self.end]