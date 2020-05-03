import numpy as np

from models.grid import Grid
from opengl_helper.buffer import BufferObject
from opengl_helper.compute_shader import ComputeShader, ComputeShaderHandler
from opengl_helper.render_utility import VertexDataHandler
from processing.edge_processing import EdgeProcessor
from utility.performance import track_time

LOG_SOURCE: str = "GRID_PROCESSING"


class GridProcessor:
    def __init__(self, grid: Grid, edge_processor: EdgeProcessor, density_strength: float = 100.0,
                 sample_radius_scale: float = 2.0, advect_strength: float = 0.01):
        self.edge_processor: EdgeProcessor = edge_processor
        self.grid: Grid = grid

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

        self.grid_position_buffer: BufferObject = BufferObject(ssbo=True)
        self.grid_density_buffer: BufferObject = BufferObject(ssbo=True)
        self.grid_gradient_buffer: BufferObject = BufferObject(ssbo=True)

        self.position_ssbo_handler: VertexDataHandler = VertexDataHandler([(self.grid_position_buffer, 0)])
        self.density_ssbo_handler: VertexDataHandler = VertexDataHandler(
            [(self.edge_processor.sample_buffer, 0), (self.grid_density_buffer, 2)])
        self.gradient_ssbo_handler: VertexDataHandler = VertexDataHandler(
            [(self.grid_density_buffer, 0), (self.grid_gradient_buffer, 1)])
        self.advect_ssbo_handler: VertexDataHandler = VertexDataHandler(
            [(self.edge_processor.sample_buffer, 0), (self.grid_gradient_buffer, 2)])

        self.density_strength: float = density_strength
        self.sample_radius: float = sample_radius_scale
        self.advect_strength: float = advect_strength

        self.empty_buffer_data = np.zeros(4 * self.grid.grid_cell_count_overall, dtype=np.int32)
        self.empty_float_buffer_data = np.zeros(4 * self.grid.grid_cell_count_overall, dtype=np.float32)

        self.grid_position_buffer.load(self.empty_float_buffer_data)
        self.grid_density_buffer.load(self.empty_buffer_data)
        self.grid_gradient_buffer.load(self.empty_float_buffer_data)

    @track_time
    def clear_buffer(self):
        self.gradient_ssbo_handler.set()
        self.clear_compute_shader.compute(self.grid.grid_cell_count_overall)

    @track_time
    def calculate_position(self):
        self.position_ssbo_handler.set()

        self.position_compute_shader.set_uniform_data([
            ('grid_cell_size', self.grid.grid_cell_size, 'vec3'),
            ('grid_bounding_min', self.grid.bounding_volume[0], 'vec3'),
            ('grid_cell_count', self.grid.grid_cell_count, 'ivec3')
        ])
        self.position_compute_shader.compute(self.grid.grid_cell_count_overall)

    @track_time
    def calculate_density(self):
        self.density_ssbo_handler.set()

        self.density_compute_shader.set_uniform_data([
            ('max_sample_points', self.edge_processor.max_sample_points, 'int'),
            ('density_strength', self.density_strength, 'float'),
            ('sample_radius', self.sample_radius, 'float'),
            ('grid_cell_size', self.grid.grid_cell_size, 'vec3'),
            ('grid_bounding_min', self.grid.bounding_volume[0], 'vec3'),
            ('grid_cell_count', self.grid.grid_cell_count, 'ivec3')
        ])

        self.density_compute_shader.compute(len(self.edge_processor.edges))

    @track_time
    def calculate_gradient(self):
        self.gradient_ssbo_handler.set()

        self.gradient_compute_shader.set_uniform_data([('grid_cell_count', self.grid.grid_cell_count, 'ivec3')])

        self.gradient_compute_shader.compute(self.grid.grid_cell_count_overall)

    @track_time
    def sample_advect(self):
        self.advect_ssbo_handler.set()

        self.advect_compute_shader.set_uniform_data([
            ('advect_strength', self.advect_strength, 'float'),
            ('grid_cell_count', self.grid.grid_cell_count, 'ivec3'),
            ('grid_bounding_min', self.grid.bounding_volume[0], 'vec3'),
            ('grid_cell_size', self.grid.grid_cell_size, 'vec3')
        ])

        # print(self.edge_processor.get_buffer_points())
        self.advect_compute_shader.compute(self.edge_processor.get_buffer_points())

        self.edge_processor.sample_buffer.swap()
