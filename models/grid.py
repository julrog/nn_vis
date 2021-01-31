from typing import List, Tuple

from pyrr import Vector4, vector4, Matrix44, matrix44, Vector3


class Grid:
    def __init__(self, grid_cell_size: Vector3, bounding_volume: Tuple[Vector3, Vector3], layer_distance: float,
                 extend_by: int = 1.0):
        self.grid_cell_size: Vector3 = grid_cell_size
        self.layer_distance: float = layer_distance

        self.bounding_volume: Tuple[Vector3, Vector3] = bounding_volume
        if self.bounding_volume[0].x > self.bounding_volume[1].x:
            self.bounding_volume[0].x, self.bounding_volume[1].x = bounding_volume[1].x - extend_by, \
                                                                   bounding_volume[0].x + extend_by
        else:
            self.bounding_volume[0].x, self.bounding_volume[1].x = bounding_volume[0].x - extend_by, \
                                                                   bounding_volume[1].x + extend_by
        if self.bounding_volume[0].y > self.bounding_volume[1].y:
            self.bounding_volume[0].y, self.bounding_volume[1].y = bounding_volume[1].y - extend_by, \
                                                                   bounding_volume[0].y + extend_by
        else:
            self.bounding_volume[0].y, self.bounding_volume[1].y = bounding_volume[0].y - extend_by, \
                                                                   bounding_volume[1].y + extend_by

        if self.bounding_volume[0].z > self.bounding_volume[1].z:
            self.bounding_volume[0].z, self.bounding_volume[1].z = bounding_volume[1].z - extend_by, \
                                                                   bounding_volume[1].z + layer_distance + extend_by
        else:
            self.bounding_volume[0].z, self.bounding_volume[1].z = bounding_volume[0].z - extend_by, \
                                                                   bounding_volume[0].z + layer_distance + extend_by

        self.grid_cell_count: List[int] = [
            int((self.bounding_volume[1].x - self.bounding_volume[0].x) / self.grid_cell_size.x) + 1,
            int((self.bounding_volume[1].y - self.bounding_volume[0].y) / self.grid_cell_size.y) + 1,
            int((self.bounding_volume[1].z - self.bounding_volume[0].z) / self.grid_cell_size.z) + 1]
        self.grid_cell_count_overall: int = self.grid_cell_count[0] * self.grid_cell_count[1] * self.grid_cell_count[2]

        self.extends: List[Vector4] = [vector4.create_from_vector3(self.bounding_volume[0], 1.0),
                                       vector4.create_from_vector3(self.bounding_volume[1], 1.0)]
        self.extends.extend([
            Vector4([self.bounding_volume[1].x, self.bounding_volume[1].y, self.bounding_volume[0].z, 1.0]),
            Vector4([self.bounding_volume[1].x, self.bounding_volume[0].y, self.bounding_volume[0].z, 1.0]),
            Vector4([self.bounding_volume[0].x, self.bounding_volume[1].y, self.bounding_volume[0].z, 1.0]),
            Vector4([self.bounding_volume[1].x, self.bounding_volume[1].y, self.bounding_volume[1].z, 1.0]),
            Vector4([self.bounding_volume[1].x, self.bounding_volume[0].y, self.bounding_volume[1].z, 1.0]),
            Vector4([self.bounding_volume[0].x, self.bounding_volume[1].y, self.bounding_volume[1].z, 1.0])
        ])

    def get_near_far_from_view(self, view: Matrix44) -> Tuple[float, float]:
        nearest_view_z: float = -1000000
        farthest_view_z: float = 1000000

        for pos in self.extends:
            view_position = Vector4(matrix44.apply_to_vector(view, pos))
            if view_position.z > nearest_view_z:
                nearest_view_z = view_position.z
            if view_position.z < farthest_view_z:
                farthest_view_z = view_position.z
        if nearest_view_z > 0:
            nearest_view_z = 0
        return nearest_view_z, farthest_view_z
