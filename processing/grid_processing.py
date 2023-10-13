import logging
import math
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from OpenGL.GL import glFinish

from models.grid import Grid
from opengl_helper.buffer import OverflowingBufferObject
from opengl_helper.compute_shader import ComputeShader
from opengl_helper.compute_shader_handler import ComputeShaderHandler
from opengl_helper.vertex_data_handler import OverflowingVertexDataHandler
from processing.advection_process import AdvectionProgress
from processing.edge_processing import EdgeProcessor
from processing.node_processing import NodeProcessor
from utility.performance import track_time


class GridProcessor:
    def __init__(self, grid: Grid, node_processor: NodeProcessor, edge_processor: EdgeProcessor,
                 density_strength: float = 1000.0) -> None:
        self.node_processor: NodeProcessor = node_processor
        self.edge_processor: EdgeProcessor = edge_processor
        self.grid: Grid = grid
        self.grid_slice_size: int = grid.grid_cell_count[0] * \
            grid.grid_cell_count[1]

        shader_settings: Dict[str, str] = {
            'grid_position': 'grid/grid_position.comp',
            'clear_grid': 'grid/clear_grid.comp',
            'node_density': 'grid/node_density_map.comp',
            'sample_density': 'grid/sample_density_map.comp',
            'node_advect': 'grid/node_advect.comp',
            'sample_advect': 'grid/sample_advect.comp',
        }
        for shader_name, path in shader_settings.items():
            ComputeShaderHandler().create(shader_name, path)

        def split_function_generation(split_grid: Grid) -> Callable:
            size_xy_slice = split_grid.grid_cell_count[0] * \
                split_grid.grid_cell_count[1] * 4

            def split_grid_data(data: np.array, i: int, size: int) -> np.array:
                fitting_slices: float = math.floor(
                    size / (4 * size_xy_slice)) - 1
                section_start: int = 0
                section_end: int = 0
                if i > 0:
                    section_start = int(i * fitting_slices * size_xy_slice)
                if i < data.nbytes / (fitting_slices * size_xy_slice * 4) - 1:
                    # add one more slice at the edge for edge cases
                    section_end = int((i + 1) *
                                      (fitting_slices + 1) * size_xy_slice)
                return data[section_start:section_end]

            return split_grid_data

        self.grid_position_buffer: OverflowingBufferObject = OverflowingBufferObject(split_function_generation(grid),
                                                                                     object_size=4,
                                                                                     render_data_offset=[
                                                                                         0],
                                                                                     render_data_size=[4])
        self.grid_density_buffer: OverflowingBufferObject = OverflowingBufferObject(split_function_generation(grid),
                                                                                    object_size=12,
                                                                                    render_data_offset=[
                                                                                        0],
                                                                                    render_data_size=[1])

        self.position_ssbo_handler: OverflowingVertexDataHandler = OverflowingVertexDataHandler(
            [], [(self.grid_position_buffer, 0)])
        self.node_density_ssbo_handler: OverflowingVertexDataHandler = OverflowingVertexDataHandler(
            [(self.node_processor.node_buffer, 0)], [(self.grid_density_buffer, 2)])
        self.sample_density_ssbo_handler: List[List[OverflowingVertexDataHandler]] = [[OverflowingVertexDataHandler(
            [(self.edge_processor.sample_buffer[i][j], 0),
             (self.edge_processor.edge_buffer[i][j], 2)],
            [(self.grid_density_buffer, 3)]) for j in range(len(self.edge_processor.sample_buffer[i]))] for i in range(
            len(self.edge_processor.sample_buffer))]
        self.node_advect_ssbo_handler: OverflowingVertexDataHandler = OverflowingVertexDataHandler(
            [(self.node_processor.node_buffer, 0)], [(self.grid_density_buffer, 2)])
        self.sample_advect_ssbo_handler: List[List[OverflowingVertexDataHandler]] = [[OverflowingVertexDataHandler(
            [(self.edge_processor.sample_buffer[i][j], 0),
             (self.edge_processor.edge_buffer[i][j], 2)],
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

    def set_new_edge_processor(self, edge_processor: EdgeProcessor) -> None:
        self.edge_processor = edge_processor

        for layer_ssbo_handler in self.sample_density_ssbo_handler:
            for container_ssbo_handler in layer_ssbo_handler:
                container_ssbo_handler.delete()
        self.sample_density_ssbo_handler = [[OverflowingVertexDataHandler(
            [(self.edge_processor.sample_buffer[i][j], 0),
             (self.edge_processor.edge_buffer[i][j], 2)],
            [(self.grid_density_buffer, 3)]) for j in range(len(self.edge_processor.sample_buffer[i]))] for i in range(
            len(self.edge_processor.sample_buffer))]

        for layer_ssbo_handler in self.sample_advect_ssbo_handler:
            for container_ssbo_handler in layer_ssbo_handler:
                container_ssbo_handler.delete()
        self.sample_advect_ssbo_handler = [[OverflowingVertexDataHandler(
            [(self.edge_processor.sample_buffer[i][j], 0),
             (self.edge_processor.edge_buffer[i][j], 2)],
            [(self.grid_density_buffer, 3)]) for j in range(len(self.edge_processor.sample_buffer[i]))] for i in range(
            len(self.edge_processor.sample_buffer))]

    def set_uniform(self, compute_shader: ComputeShader, uniforms: List[str]) -> None:
        uniform_data: List[Tuple[str, Any, Any]] = []
        if 'slice_size' in uniforms:
            uniform_data.append(('slice_size', self.grid_slice_size, 'int'))
        if 'slice_count' in uniforms:
            uniform_data.append(
                ('slice_count', self.position_buffer_slice_count, 'int'))
        if 'grid_cell_size' in uniforms:
            uniform_data.append(
                ('grid_cell_size', self.grid.grid_cell_size, 'vec3'))
        if 'grid_bounding_min' in uniforms:
            uniform_data.append(
                ('grid_bounding_min', self.grid.bounding_volume[0], 'vec3'))
        if 'grid_bounding_max' in uniforms:
            uniform_data.append(
                ('grid_bounding_max', self.grid.bounding_volume[1], 'vec3'))
        if 'grid_cell_count' in uniforms:
            uniform_data.append(
                ('grid_cell_count', self.grid.grid_cell_count, 'ivec3'))
        if 'density_strength' in uniforms:
            uniform_data.append(
                ('density_strength', self.density_strength, 'float'))
        if 'max_sample_points' in uniforms:
            uniform_data.append(
                ('max_sample_points', self.edge_processor.max_sample_points, 'int'))
        if 'edge_importance_type' in uniforms:
            uniform_data.append(
                ('edge_importance_type', self.edge_processor.edge_importance_type, 'int'))
        compute_shader.set_uniform_data(uniform_data)

    @track_time
    def clear_buffer(self) -> None:
        clear: ComputeShader = ComputeShaderHandler().get('clear_grid')
        for i in range(len(self.grid_density_buffer.handle)):
            self.density_ssbo_handler.set_buffer(i)
            self.density_ssbo_handler.set()
            clear.compute(self.grid_density_buffer.get_objects(i))
        clear.barrier()

    @track_time
    def calculate_position(self) -> None:
        logging.info('Calculate grid positions.')
        position: ComputeShader = ComputeShaderHandler().get('grid_position')
        self.set_uniform(position,
                         ['slice_size', 'slice_count', 'grid_cell_size', 'grid_bounding_min', 'grid_cell_count'])
        for i in range(len(self.grid_position_buffer.handle)):
            self.position_ssbo_handler.set_buffer(i)
            self.position_ssbo_handler.set()
            position.set_uniform_data([('current_buffer', i, 'int')])
            position.compute(self.grid_position_buffer.get_objects(i))
        position.barrier()

    @track_time
    def calculate_node_density(self, advection_status: AdvectionProgress) -> None:
        self.node_density_ssbo_handler.set_buffer(0)
        self.node_density_ssbo_handler.set()
        density: ComputeShader = ComputeShaderHandler().get('node_density')
        self.set_uniform(density, [
                         'density_strength', 'grid_cell_size', 'grid_bounding_min', 'grid_cell_count'])
        density.set_uniform_data(
            [('bandwidth', advection_status.current_bandwidth, 'float')])
        density.compute(len(self.node_processor.nodes))
        density.barrier()

    @track_time
    def calculate_edge_density(self, layer: int, advection_status: AdvectionProgress, wait_for_compute: bool = False) -> None:
        density: ComputeShader = ComputeShaderHandler().get('sample_density')
        self.set_uniform(density, ['max_sample_points', 'slice_size', 'slice_count', 'density_strength',
                                   'grid_cell_size', 'grid_bounding_min', 'grid_cell_count', 'edge_importance_type'])
        density.set_uniform_data(
            [('bandwidth', advection_status.current_bandwidth, 'float')])
        for i in range(len(self.grid_density_buffer.handle)):
            density.set_uniform_data([('current_buffer', i, 'int')])
            for container in range(len(self.edge_processor.sample_buffer[layer])):
                density.set_uniform_data(
                    [('grid_layer_offset', self.grid.layer_distance * layer, 'float')])
                self.sample_density_ssbo_handler[layer][container].set_buffer(
                    i - 1)
                self.sample_density_ssbo_handler[layer][container].set_range(3)
                density.compute(
                    self.edge_processor.get_buffer_points(layer, container))
                if wait_for_compute:
                    glFinish()
        density.barrier()

    @track_time
    def node_advect(self, advection_status: AdvectionProgress) -> None:
        self.node_advect_ssbo_handler.set_buffer(0)
        self.node_advect_ssbo_handler.set()
        advect: ComputeShader = ComputeShaderHandler().get('node_advect')
        self.set_uniform(advect, [
                         'grid_cell_size', 'grid_bounding_min', 'grid_bounding_max', 'grid_cell_count'])
        advect.set_uniform_data([
            ('advect_strength', advection_status.get_advection_strength(), 'float'),
            ('importance_similarity', advection_status.importance_similarity, 'float')
        ])
        advect.compute(self.node_processor.get_buffer_points())
        advect.barrier()
        self.node_processor.node_buffer.swap()

    @track_time
    def sample_advect(self, layer: int, advection_status: AdvectionProgress, wait_for_compute: bool = False) -> None:
        advect: ComputeShader = ComputeShaderHandler().get('sample_advect')
        self.set_uniform(advect, ['max_sample_points', 'slice_size', 'slice_count', 'grid_cell_size',
                                  'grid_bounding_min', 'grid_cell_count', 'edge_importance_type'])
        advect.set_uniform_data([
            ('advect_strength', advection_status.get_advection_strength(), 'float'),
            ('importance_similarity', advection_status.importance_similarity, 'float')
        ])
        for i in range(len(self.grid_density_buffer.handle)):
            advect.set_uniform_data([('current_buffer', i, 'int')])
            for container in range(len(self.edge_processor.sample_buffer[layer])):
                advect.set_uniform_data(
                    [('grid_layer_offset', self.grid.layer_distance * layer, 'float')])
                self.sample_advect_ssbo_handler[layer][container].set_buffer(i)
                self.sample_advect_ssbo_handler[layer][container].set()
                advect.compute(
                    self.edge_processor.get_buffer_points(layer, container))
                self.edge_processor.sample_buffer[layer][container].swap()
                if wait_for_compute:
                    glFinish()
        advect.barrier()

    def delete(self) -> None:
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
