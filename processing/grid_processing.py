import math
from typing import List
import numpy as np
from models.grid import Grid
from opengl_helper.buffer import OverflowingBufferObject
from opengl_helper.compute_shader import ComputeShader, ComputeShaderHandler
from opengl_helper.render_utility import OverflowingVertexDataHandler
from processing.advection_process import AdvectionProgress
from processing.edge_processing import EdgeProcessor
from processing.node_processing import NodeProcessor
from utility.performance import track_time
from OpenGL.GL import *


LOG_SOURCE: str = "GRID_PROCESSING"


class GridProcessor:
    def __init__(self, grid: Grid, node_processor: NodeProcessor, edge_processor: EdgeProcessor,
                 density_strength: float = 1000.0):
        self.node_processor: NodeProcessor = node_processor
        self.edge_processor: EdgeProcessor = edge_processor
        self.grid: Grid = grid
        self.grid_slice_size: int = grid.grid_cell_count[0] * grid.grid_cell_count[1]

        self.position_compute_shader: ComputeShader = ComputeShaderHandler().create("grid_position",
                                                                                    "grid/grid_position.comp")
        self.clear_compute_shader: ComputeShader = ComputeShaderHandler().create("clear_grid",
                                                                                 "grid/clear_grid.comp")
        self.node_density_compute_shader: ComputeShader = ComputeShaderHandler().create("node_density",
                                                                                        "grid/node_density_map.comp")
        self.sample_density_compute_shader: ComputeShader = ComputeShaderHandler().create("sample_density",
                                                                                          "grid/sample_density_map.comp")
        self.node_advect_compute_shader: ComputeShader = ComputeShaderHandler().create("node_advect",
                                                                                       "grid/node_advect.comp")
        self.sample_advect_compute_shader: ComputeShader = ComputeShaderHandler().create("sample_advect",
                                                                                         "grid/sample_advect.comp")

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

        self.grid_position_buffer: OverflowingBufferObject = OverflowingBufferObject(split_function_generation(grid),
                                                                                     object_size=12,
                                                                                     render_data_offset=[0],
                                                                                     render_data_size=[4])
        self.grid_density_buffer: OverflowingBufferObject = OverflowingBufferObject(split_function_generation(grid),
                                                                                    object_size=12,
                                                                                    render_data_offset=[0],
                                                                                    render_data_size=[1])

        self.position_ssbo_handler: OverflowingVertexDataHandler = OverflowingVertexDataHandler(
            [], [(self.grid_position_buffer, 0)])
        self.node_density_ssbo_handler: OverflowingVertexDataHandler = OverflowingVertexDataHandler(
            [(self.node_processor.node_buffer, 0)], [(self.grid_density_buffer, 2)])
        self.sample_density_ssbo_handler: List[List[OverflowingVertexDataHandler]] = [[OverflowingVertexDataHandler(
            [(self.edge_processor.sample_buffer[i][j], 0), (self.edge_processor.edge_buffer[i][j], 2)],
            [(self.grid_density_buffer, 3)]) for j in range(len(self.edge_processor.sample_buffer[i]))] for i in range(
            len(self.edge_processor.sample_buffer))]
        self.node_advect_ssbo_handler: OverflowingVertexDataHandler = OverflowingVertexDataHandler(
            [(self.node_processor.node_buffer, 0)], [(self.grid_density_buffer, 2)])
        self.sample_advect_ssbo_handler: List[List[OverflowingVertexDataHandler]] = [[OverflowingVertexDataHandler(
            [(self.edge_processor.sample_buffer[i][j], 0), (self.edge_processor.edge_buffer[i][j], 2)],
            [(self.grid_density_buffer, 3)]) for j in range(len(self.edge_processor.sample_buffer[i]))] for i in range(
            len(self.edge_processor.sample_buffer))]
        self.density_ssbo_handler: OverflowingVertexDataHandler = OverflowingVertexDataHandler(
            [], [(self.grid_density_buffer, 0)])

        self.density_strength: float = density_strength

        self.grid_position_buffer.load_empty(np.float32, self.grid_slice_size * grid.grid_cell_count[2],
                                             self.grid_slice_size)
        self.grid_density_buffer.load_empty(np.int32, self.grid_slice_size * grid.grid_cell_count[2],
                                            self.grid_slice_size)

        self.position_buffer_slice_count: int = math.floor(
            self.grid_position_buffer.size[0] / (self.grid_position_buffer.object_size * 4 * self.grid_slice_size)) - 1
        self.density_buffer_slice_count: int = math.floor(
            self.grid_density_buffer.size[0] / (self.grid_density_buffer.object_size * 4 * self.grid_slice_size)) - 1

    def set_node_processor(self, node_processor: NodeProcessor):
        self.node_processor = node_processor
        self.node_density_ssbo_handler.delete()
        self.node_density_ssbo_handler = OverflowingVertexDataHandler(
            [(self.node_processor.node_buffer, 0)], [(self.grid_density_buffer, 2)])
        self.sample_advect_ssbo_handler.delete()
        self.sample_advect_ssbo_handler = OverflowingVertexDataHandler(
            [(self.node_processor.node_buffer, 0)], [(self.grid_density_buffer, 2)])

    def set_new_edge_processor(self, edge_processor: EdgeProcessor):
        self.edge_processor = edge_processor

        for layer_ssbo_handler in self.sample_density_ssbo_handler:
            for container_ssbo_handler in layer_ssbo_handler:
                container_ssbo_handler.delete()
        self.sample_density_ssbo_handler = [[OverflowingVertexDataHandler(
            [(self.edge_processor.sample_buffer[i][j], 0), (self.edge_processor.edge_buffer[i][j], 2)],
            [(self.grid_density_buffer, 3)]) for j in range(len(self.edge_processor.sample_buffer[i]))] for i in range(
            len(self.edge_processor.sample_buffer))]

        for layer_ssbo_handler in self.sample_advect_ssbo_handler:
            for container_ssbo_handler in layer_ssbo_handler:
                container_ssbo_handler.delete()
        self.sample_advect_ssbo_handler = [[OverflowingVertexDataHandler(
            [(self.edge_processor.sample_buffer[i][j], 0), (self.edge_processor.edge_buffer[i][j], 2)],
            [(self.grid_density_buffer, 3)]) for j in range(len(self.edge_processor.sample_buffer[i]))] for i in range(
            len(self.edge_processor.sample_buffer))]

    @track_time
    def clear_buffer(self):
        for i in range(len(self.grid_density_buffer.handle)):
            self.density_ssbo_handler.set_buffer(i)
            self.density_ssbo_handler.set()
            self.clear_compute_shader.compute(self.grid_density_buffer.get_objects(i))
        self.clear_compute_shader.barrier()

    @track_time
    def calculate_position(self):
        print("[%s] Calculate grid positions." % LOG_SOURCE)
        for i in range(len(self.grid_position_buffer.handle)):
            self.position_ssbo_handler.set_buffer(i)
            self.position_ssbo_handler.set()
            self.position_compute_shader.set_uniform_data([
                ('slice_size', self.grid_slice_size, 'int'),
                ('slice_count', self.position_buffer_slice_count, 'int'),
                ('current_buffer', i, 'int'),
                ('grid_cell_size', self.grid.grid_cell_size, 'vec3'),
                ('grid_bounding_min', self.grid.bounding_volume[0], 'vec3'),
                ('grid_cell_count', self.grid.grid_cell_count, 'ivec3')
            ])
            self.position_compute_shader.compute(self.grid_position_buffer.get_objects(i))
        self.position_compute_shader.barrier()

    @track_time
    def calculate_node_density(self, advection_status: AdvectionProgress):
        self.node_density_ssbo_handler.set_buffer(0)
        self.node_density_ssbo_handler.set()

        self.node_density_compute_shader.set_uniform_data([
            ('density_strength', self.density_strength, 'float'),
            ('bandwidth', advection_status.current_bandwidth, 'float'),
            ('grid_cell_size', self.grid.grid_cell_size, 'vec3'),
            ('grid_bounding_min', self.grid.bounding_volume[0], 'vec3'),
            ('grid_cell_count', self.grid.grid_cell_count, 'ivec3')
        ])

        self.node_density_compute_shader.compute(len(self.node_processor.nodes))
        self.node_density_compute_shader.barrier()

    @track_time
    def calculate_edge_density(self, layer: int, advection_status: AdvectionProgress, wait_for_compute: bool = False):
        for i in range(len(self.grid_density_buffer.handle)):
            self.sample_density_compute_shader.set_uniform_data([
                ('max_sample_points', self.edge_processor.max_sample_points, 'int'),
                ('slice_size', self.grid_slice_size, 'int'),
                ('slice_count', self.density_buffer_slice_count, 'int'),
                ('current_buffer', i, 'int'),
                ('density_strength', self.density_strength, 'float'),
                ('bandwidth', advection_status.current_bandwidth, 'float'),
                ('grid_cell_size', self.grid.grid_cell_size, 'vec3'),
                ('grid_bounding_min', self.grid.bounding_volume[0], 'vec3'),
                ('grid_cell_count', self.grid.grid_cell_count, 'ivec3'),
                ('edge_importance_type', self.edge_processor.edge_importance_type, 'int')
            ])

            for container in range(len(self.edge_processor.sample_buffer[layer])):
                self.sample_density_compute_shader.set_uniform_data(
                    [('grid_layer_offset', self.grid.layer_distance * layer, 'float')])
                self.sample_density_ssbo_handler[layer][container].set_buffer(i - 1)
                self.sample_density_ssbo_handler[layer][container].set_range(3)
                self.sample_density_compute_shader.compute(self.edge_processor.get_buffer_points(layer, container))
                if wait_for_compute:
                    glFinish()
        self.sample_density_compute_shader.barrier()

    @track_time
    def node_advect(self, advection_status: AdvectionProgress):
        self.node_advect_ssbo_handler.set_buffer(0)
        self.node_advect_ssbo_handler.set()

        self.node_advect_compute_shader.set_uniform_data([
            ('advect_strength', advection_status.get_advection_strength(), 'float'),
            ('importance_similarity', advection_status.importance_similarity, 'float'),
            ('grid_cell_count', self.grid.grid_cell_count, 'ivec3'),
            ('grid_bounding_min', self.grid.bounding_volume[0], 'vec3'),
            ('grid_bounding_max', self.grid.bounding_volume[1], 'vec3'),
            ('grid_cell_size', self.grid.grid_cell_size, 'vec3')
        ])

        self.node_advect_compute_shader.compute(self.node_processor.get_buffer_points())
        self.node_advect_compute_shader.barrier()
        self.node_processor.node_buffer.swap()

    @track_time
    def sample_advect(self, layer: int, advection_status: AdvectionProgress, wait_for_compute: bool = False):
        for i in range(len(self.grid_density_buffer.handle)):
            self.sample_advect_compute_shader.set_uniform_data([
                ('max_sample_points', self.edge_processor.max_sample_points, 'int'),
                ('slice_size', self.grid_slice_size, 'int'),
                ('slice_count', self.density_buffer_slice_count, 'int'),
                ('current_buffer', i, 'int'),
                ('advect_strength', advection_status.get_advection_strength(), 'float'),
                ('importance_similarity', advection_status.importance_similarity, 'float'),
                ('grid_cell_count', self.grid.grid_cell_count, 'ivec3'),
                ('grid_bounding_min', self.grid.bounding_volume[0], 'vec3'),
                ('grid_cell_size', self.grid.grid_cell_size, 'vec3'),
                ('edge_importance_type', self.edge_processor.edge_importance_type, 'int')
            ])

            for container in range(len(self.edge_processor.sample_buffer[layer])):
                self.sample_advect_compute_shader.set_uniform_data(
                    [('grid_layer_offset', self.grid.layer_distance * layer, 'float')])
                self.sample_advect_ssbo_handler[layer][container].set_buffer(i)
                self.sample_advect_ssbo_handler[layer][container].set()
                self.sample_advect_compute_shader.compute(self.edge_processor.get_buffer_points(layer, container))
                self.edge_processor.sample_buffer[layer][container].swap()
                if wait_for_compute:
                    glFinish()

        self.sample_advect_compute_shader.barrier()

    def delete(self):
        self.grid_position_buffer.delete()

        self.grid_density_buffer.delete()

        self.position_ssbo_handler.delete()

        self.node_density_ssbo_handler.delete()
        for layer_ssbo_handler in self.sample_density_ssbo_handler:
            for container_ssbo_handler in layer_ssbo_handler:
                container_ssbo_handler.delete()
        self.sample_density_ssbo_handler = []

        self.density_ssbo_handler.delete()

        for layer_ssbo_handler in self.sample_advect_ssbo_handler:
            for container_ssbo_handler in layer_ssbo_handler:
                container_ssbo_handler.delete()
        self.sample_advect_ssbo_handler = []
