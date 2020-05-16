from typing import List

from pyrr import Vector3

from models.grid import Grid
from models.network import NetworkModel
from opengl_helper.render_utility import clear_screen
from processing.edge_processing import EdgeProcessor
from processing.grid_processing import GridProcessor
from processing.node_processing import NodeProcessor
from rendering.edge_rendering import EdgeRenderer
from rendering.grid_rendering import GridRenderer
from rendering.node_rendering import NodeRenderer
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

        self.node_processor = NodeProcessor()
        self.node_processor.set_data(self.network)
        self.node_renderer = NodeRenderer(self.node_processor, self.grid)

        self.edge_processor = EdgeProcessor(self.sample_length)
        self.edge_processor.set_data(self.network)
        self.edge_processor.init_sample_edge()
        self.edge_renderer = EdgeRenderer(self.edge_processor, self.grid)

        self.grid_processor = GridProcessor(self.grid, self.node_processor, self.edge_processor, 10.0,
                                            self.sample_radius, 0.01)
        self.grid_processor.calculate_position()
        self.grid_processor.calculate_edge_density()
        self.grid_processor.calculate_gradient()

        self.grid_renderer = GridRenderer(self.grid_processor)

    def reset_edges(self):
        self.edge_processor.delete()
        self.edge_renderer.delete()

        self.node_processor.read_nodes_from_buffer()
        self.network.set_nodes(self.node_processor.nodes)
        self.edge_processor = EdgeProcessor(self.sample_length)
        self.edge_processor.set_data(self.network)
        self.edge_processor.init_sample_edge()
        self.edge_renderer = EdgeRenderer(self.edge_processor, self.grid)

        self.grid_processor.set_edge_processor(self.edge_processor)

    def process(self, window: Window, action_mode: int):
        self.edge_processor.check_limits(window.cam.view)
        if action_mode is not 0:
            if action_mode == 1:
                self.grid_processor.clear_buffer()
                self.grid_processor.calculate_node_density()
                self.grid_processor.calculate_gradient()
                if self.grid_processor.advect_strength < 0:
                    self.grid_processor.advect_strength = -self.grid_processor.advect_strength
                self.grid_processor.node_advect()
            elif action_mode == 2:
                self.grid_processor.clear_buffer()
                self.grid_processor.calculate_node_density()
                self.grid_processor.calculate_gradient()
                if self.grid_processor.advect_strength > 0:
                    self.grid_processor.advect_strength = -self.grid_processor.advect_strength
                self.grid_processor.node_advect()
            elif action_mode == 3:
                self.node_processor.node_noise(self.sample_length, 0.5)
            if action_mode == 4:
                self.grid_processor.clear_buffer()
                self.grid_processor.calculate_edge_density()
                self.grid_processor.calculate_gradient()
                if self.grid_processor.advect_strength < 0:
                    self.grid_processor.advect_strength = -self.grid_processor.advect_strength
                self.grid_processor.sample_advect()
            elif action_mode == 5:
                self.grid_processor.clear_buffer()
                self.grid_processor.calculate_edge_density()
                self.grid_processor.calculate_gradient()
                if self.grid_processor.advect_strength > 0:
                    self.grid_processor.advect_strength = -self.grid_processor.advect_strength
                self.grid_processor.sample_advect()
            elif action_mode == 6:
                self.edge_processor.sample_noise(0.5)

            self.edge_processor.sample_edges()
            self.edge_processor.sample_smooth()

    def render(self, window: Window, edge_render_mode: int, grid_render_mode: int, node_render_mode: int):
        clear_screen([1.0, 1.0, 1.0, 1.0])
        if window.gradient and grid_render_mode == 1:
            self.grid_renderer.render_cube(window, clear=False, swap=False)
        elif window.gradient and grid_render_mode == 2:
            self.grid_renderer.render_point(window, clear=False, swap=False)
        if edge_render_mode == 5:
            self.edge_renderer.render_point(window, clear=False, swap=False)
        elif edge_render_mode == 4:
            self.edge_renderer.render_line(window, clear=False, swap=False)
        elif edge_render_mode == 3:
            self.edge_renderer.render_ellipsoid_transparent(window, clear=False, swap=False)
        elif edge_render_mode == 2:
            self.edge_renderer.render_transparent_sphere(window, clear=False, swap=False)
        elif edge_render_mode == 1:
            self.edge_renderer.render_sphere(window, clear=False, swap=False)
        if node_render_mode == 3:
            self.node_renderer.render_point(window, clear=False, swap=False)
        elif node_render_mode == 2:
            self.node_renderer.render_transparent(window, clear=False, swap=False)
        elif node_render_mode == 1:
            self.node_renderer.render_sphere(window, clear=False, swap=False)

    def delete(self):
        self.node_processor.delete()
        self.node_renderer.delete()
        self.edge_processor.delete()
        self.edge_renderer.delete()
        self.grid_processor.delete()
        self.grid_renderer.delete()
