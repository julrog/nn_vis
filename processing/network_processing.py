from typing import List

from pyrr import Vector3

from models.grid import Grid
from models.network import NetworkModel
from opengl_helper.render_utility import clear_screen
from processing.edge_processing import EdgeProcessor
from processing.grid_processing import GridProcessor
from rendering.edge_rendering import EdgeRenderer
from rendering.grid_rendering import GridRenderer
from utility.window import Window

LOG_SOURCE: str = "NETWORK_PROCESSING"


class NetworkProcessor:
    def __init__(self, layer_data: List[int], layer_distance: float = 1.0, node_size: float = 0.3,
                 sampling_rate: float = 10.0):
        self.layer_data = layer_data
        self.layer_distance = layer_distance
        self.node_size = node_size

        self.network = NetworkModel(self.layer_data, self.node_size, self.layer_distance)
        self.sample_length = self.network.max_layer_width / sampling_rate
        self.grid_cell_size = self.sample_length / 3.0
        self.sample_radius = self.sample_length * 2.0

        self.grid = Grid(Vector3([self.grid_cell_size, self.grid_cell_size, self.grid_cell_size]),
                         self.network.bounding_volume)

        self.edge_handler = EdgeProcessor(self.sample_length)
        self.edge_handler.set_data(self.network)
        self.edge_handler.init_sample_edge()
        self.edge_renderer = EdgeRenderer(self.edge_handler, self.grid)

        self.grid_processor = GridProcessor(self.grid, self.edge_handler, 10.0, self.sample_radius, 0.01)
        self.grid_processor.calculate_position()
        self.grid_processor.calculate_density()
        self.grid_processor.calculate_gradient()

        self.grid_renderer = GridRenderer(self.grid_processor)

    def render(self, window: Window, edge_render_mode: int, grid_render_mode: int):
        self.edge_handler.check_limits(window.cam.view)
        if not window.freeze:
            if window.gradient:
                self.grid_processor.clear_buffer()
                self.grid_processor.calculate_density()
                self.grid_processor.calculate_gradient()
                self.grid_processor.sample_advect()

            # edge_handler.sample_noise(0.5)
            self.edge_handler.sample_edges()
            # edge_handler.sample_smooth()

        clear_screen([1.0, 1.0, 1.0, 1.0])
        if window.gradient and grid_render_mode == 1:
            self.grid_renderer.render_cube(window, clear=False, swap=False)
        if edge_render_mode == 2:
            self.edge_renderer.render_transparent(window, clear=False, swap=False)
        elif edge_render_mode == 1:
            self.edge_renderer.render_sphere(window, clear=False, swap=False)

    def delete(self):
        self.edge_handler.delete()
        self.edge_renderer.delete()
        self.grid_processor.delete()
        self.grid_renderer.delete()
