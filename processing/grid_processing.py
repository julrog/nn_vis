import math
from typing import Tuple, List

import numpy as np
from pyrr import Vector3

from opengl_helper.buffer import BufferObject
from opengl_helper.compute_shader import ComputeShader, ComputeShaderHandler
from opengl_helper.render_utility import VertexDataHandler
from processing.edge_processing import EdgeProcessor
from utility.performance import track_time

LOG_SOURCE: str = "GRID_PROCESSING"


class GridProcessor:
    def __init__(self, grid_cell_size: Vector3, bounding_volume: Tuple[Vector3, Vector3],
                 edge_processor: EdgeProcessor, density_strength: float = 100.0, sample_radius_scale: float = 2.0):
        self.edge_processor: EdgeProcessor = edge_processor
        self.grid_cell_size: Vector3 = grid_cell_size

        self.bounding_volume: Tuple[Vector3, Vector3] = bounding_volume
        if self.bounding_volume[0].x > self.bounding_volume[1].x:
            self.bounding_volume[0].x, self.bounding_volume[1].x = bounding_volume[1].x, bounding_volume[0].x
        if self.bounding_volume[0].y > self.bounding_volume[1].y:
            self.bounding_volume[0].y, self.bounding_volume[1].y = bounding_volume[1].y, bounding_volume[0].y
        if self.bounding_volume[0].z > self.bounding_volume[1].z:
            self.bounding_volume[0].z, self.bounding_volume[1].z = bounding_volume[1].z, bounding_volume[0].z

        self.grid_cell_count: List[int] = [
            int((self.bounding_volume[1].x - self.bounding_volume[0].x) / self.grid_cell_size.x) + 2,
            int((self.bounding_volume[1].y - self.bounding_volume[0].y) / self.grid_cell_size.y) + 2,
            int((self.bounding_volume[1].z - self.bounding_volume[0].z) / self.grid_cell_size.z) + 2]

        self.density_compute_shader: ComputeShader = ComputeShaderHandler().create("sample_density",
                                                                                   "sample_density_map.comp")
        self.grid_density_buffer: BufferObject = BufferObject(ssbo=True)
        self.ssbo_handler: VertexDataHandler = VertexDataHandler(
            [(self.edge_processor.sample_buffer, 0), (self.grid_density_buffer, 2)])

        self.density_strength: float = density_strength
        self.sample_radius: float = sample_radius_scale

        self.empty_buffer_data = np.array(
            [0] * 4 * self.grid_cell_count[0] * self.grid_cell_count[1] * self.grid_cell_count[2],
            dtype=np.int32)
        self.set_density_buffer_data()

    @track_time
    def set_density_buffer_data(self):
        self.grid_density_buffer.load(self.empty_buffer_data)

    @track_time
    def calculate_density(self):
        self.set_density_buffer_data()
        self.ssbo_handler.set()

        self.density_compute_shader.set_uniform_data(
            [('max_sample_points', self.edge_processor.max_sample_points, 'int')])
        self.density_compute_shader.set_uniform_data([('density_strength', self.density_strength, 'float')])
        self.density_compute_shader.set_uniform_data([('sample_radius', self.sample_radius, 'float')])
        self.density_compute_shader.set_uniform_data(
            [('grid_cell_size', self.grid_cell_size, 'vec3')])
        self.density_compute_shader.set_uniform_data(
            [('grid_bounding_min', self.bounding_volume[0], 'vec3')])
        self.density_compute_shader.set_uniform_data(
            [('grid_cell_count', self.grid_cell_count, 'ivec3')])

        for i in range(math.ceil(len(self.edge_processor.edges) / self.density_compute_shader.max_workgroup_size)):
            self.density_compute_shader.set_uniform_data(
                [('work_group_offset', i * self.density_compute_shader.max_workgroup_size, 'int')])

            if i == math.ceil(len(self.edge_processor.edges) / self.density_compute_shader.max_workgroup_size) - 1:
                self.density_compute_shader.compute(
                    len(self.edge_processor.edges) % self.density_compute_shader.max_workgroup_size)
            else:
                self.density_compute_shader.compute(self.density_compute_shader.max_workgroup_size)

    @track_time
    def get_grid_count(self) -> int:
        return self.grid_cell_count[0] * self.grid_cell_count[1] * self.grid_cell_count[2]
