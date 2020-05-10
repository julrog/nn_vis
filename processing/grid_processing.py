import math

import numpy as np

from models.grid import Grid
from opengl_helper.buffer import BufferObject, OverflowingBufferObject
from opengl_helper.compute_shader import ComputeShader, ComputeShaderHandler
from opengl_helper.render_utility import VertexDataHandler, OverflowingVertexDataHandler
from processing.edge_processing import EdgeProcessor
from utility.performance import track_time

LOG_SOURCE: str = "GRID_PROCESSING"


class GridProcessor:
    def __init__(self, grid: Grid, edge_processor: EdgeProcessor, density_strength: float = 100.0,
                 sample_radius_scale: float = 2.0, advect_strength: float = 0.01):
        self.edge_processor: EdgeProcessor = edge_processor
        self.grid: Grid = grid
        self.grid_slice_size: int = grid.grid_cell_count[0] * grid.grid_cell_count[1]

        self.position_compute_shader: ComputeShader = ComputeShaderHandler().create("grid_position",
                                                                                    "grid_position.comp")
        self.clear_compute_shader: ComputeShader = ComputeShaderHandler().create("clear_grid",
                                                                                 "clear_grid.comp")
        self.density_compute_shader: ComputeShader = ComputeShaderHandler().create("sample_density",
                                                                                   "sample_density_map.comp")
        self.gradient_compute_shader: ComputeShader = ComputeShaderHandler().create("grid_gradient",
                                                                                    "grid_gradient.comp")
        self.advect_compute_shader: ComputeShader = ComputeShaderHandler().create("sample_advect",
                                                                                  "sample_advect.comp")

        def split_function_generation(split_grid: Grid):
            size_xy_slice = split_grid.grid_cell_count[0] * split_grid.grid_cell_count[1] * 4

            def split_grid_data(data, i, size):
                fitting_slices = math.floor(size / (4 * size_xy_slice)) - 1
                section_start = None
                section_end = None
                if i > 0:
                    section_start = i * fitting_slices * size_xy_slice
                if i < data.nbytes / (fitting_slices * size_xy_slice * 4) - 1:
                    # add one more slice at the edge for edge cases
                    section_end = (i + 1) * (fitting_slices + 1) * size_xy_slice
                return data[section_start:section_end]

            return split_grid_data

        self.grid_position_buffer: OverflowingBufferObject = OverflowingBufferObject(split_function_generation(grid))
        self.grid_density_buffer: OverflowingBufferObject = OverflowingBufferObject(split_function_generation(grid))
        self.grid_gradient_buffer: OverflowingBufferObject = OverflowingBufferObject(split_function_generation(grid))

        self.position_ssbo_handler: OverflowingVertexDataHandler = OverflowingVertexDataHandler(
            [], [(self.grid_position_buffer, 0)])
        self.density_ssbo_handler: OverflowingVertexDataHandler = OverflowingVertexDataHandler(
            [(self.edge_processor.sample_buffer, 0)], [(self.grid_density_buffer, 2)])
        self.gradient_ssbo_handler: OverflowingVertexDataHandler = OverflowingVertexDataHandler(
            [], [(self.grid_density_buffer, 0), (self.grid_gradient_buffer, 1)])
        self.advect_ssbo_handler: OverflowingVertexDataHandler = OverflowingVertexDataHandler(
            [(self.edge_processor.sample_buffer, 0)], [(self.grid_gradient_buffer, 2)])

        self.density_strength: float = density_strength
        self.sample_radius: float = sample_radius_scale
        self.advect_strength: float = advect_strength

        self.grid_position_buffer.load_empty(np.float32, self.grid_slice_size * grid.grid_cell_count[2],
                                             self.grid_slice_size)
        self.grid_density_buffer.load_empty(np.int32, self.grid_slice_size * grid.grid_cell_count[2],
                                            self.grid_slice_size)
        self.grid_gradient_buffer.load_empty(np.float32, self.grid_slice_size * grid.grid_cell_count[2],
                                             self.grid_slice_size)

        self.buffer_slice_count: int = math.floor(self.grid_gradient_buffer.size[0] / (16 * self.grid_slice_size)) - 1

    @track_time
    def clear_buffer(self):
        for i in range(len(self.grid_position_buffer.handle)):
            self.gradient_ssbo_handler.set(i)
            self.clear_compute_shader.compute(int(self.grid_position_buffer.size[i] / 16), barrier=False)
        self.clear_compute_shader.barrier()

    @track_time
    def calculate_position(self):
        for i in range(len(self.grid_position_buffer.handle)):
            self.position_ssbo_handler.set(i)
            print("SET slice_size %i, slice_count %i, buffer %i" % (self.grid_slice_size, self.buffer_slice_count, i))
            self.position_compute_shader.set_uniform_data([
                ('slice_size', self.grid_slice_size, 'int'),
                ('slice_count', self.buffer_slice_count, 'int'),
                ('current_buffer', i, 'int'),
                ('grid_cell_size', self.grid.grid_cell_size, 'vec3'),
                ('grid_bounding_min', self.grid.bounding_volume[0], 'vec3'),
                ('grid_cell_count', self.grid.grid_cell_count, 'ivec3')
            ])
            print("Calculate buffer %i" % int(self.grid_position_buffer.size[i] / 16))
            self.position_compute_shader.compute(int(self.grid_position_buffer.size[i] / 16), barrier=False)
        self.position_compute_shader.barrier()

    @track_time
    def calculate_density(self):
        for i in range(len(self.grid_position_buffer.handle)):
            self.density_ssbo_handler.set_range(i - 1, 3)

            self.density_compute_shader.set_uniform_data([
                ('slice_size', self.grid_slice_size, 'int'),
                ('slice_count', self.buffer_slice_count, 'int'),
                ('current_buffer', i, 'int'),
                ('max_sample_points', self.edge_processor.max_sample_points, 'int'),
                ('density_strength', self.density_strength, 'float'),
                ('sample_radius', self.sample_radius, 'float'),
                ('grid_cell_size', self.grid.grid_cell_size, 'vec3'),
                ('grid_bounding_min', self.grid.bounding_volume[0], 'vec3'),
                ('grid_cell_count', self.grid.grid_cell_count, 'ivec3')
            ])

            self.density_compute_shader.compute(len(self.edge_processor.edges), barrier=False)
        self.density_compute_shader.barrier()

    @track_time
    def calculate_gradient(self):
        for i in range(len(self.grid_position_buffer.handle)):
            self.gradient_ssbo_handler.set(i)

            self.gradient_compute_shader.set_uniform_data([
                ('grid_cell_count', self.grid.grid_cell_count, 'ivec3')
            ])

            self.gradient_compute_shader.compute(int(self.grid_position_buffer.size[i] / 16), barrier=False)
        self.gradient_compute_shader.barrier()

    @track_time
    def sample_advect(self):
        for i in range(len(self.grid_position_buffer.handle)):
            self.advect_ssbo_handler.set(i)

            self.advect_compute_shader.set_uniform_data([
                ('slice_count', self.buffer_slice_count, 'int'),
                ('current_buffer', i, 'int'),
                ('advect_strength', self.advect_strength, 'float'),
                ('grid_cell_count', self.grid.grid_cell_count, 'ivec3'),
                ('grid_bounding_min', self.grid.bounding_volume[0], 'vec3'),
                ('grid_cell_size', self.grid.grid_cell_size, 'vec3')
            ])

            self.advect_compute_shader.compute(self.edge_processor.get_buffer_points())
        self.advect_compute_shader.barrier()
        self.edge_processor.sample_buffer.swap()

    def delete(self):
        self.grid_position_buffer.delete()
        self.grid_density_buffer.delete()
        self.grid_gradient_buffer.delete()

        self.position_ssbo_handler.delete()
        self.density_ssbo_handler.delete()
        self.gradient_ssbo_handler.delete()
        self.advect_ssbo_handler.delete()
