import numpy as np
from typing import List, Dict

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
    def __init__(self, layer_nodes: List[int], layer_data: List[np.array] = None, layer_distance: float = 1.0,
                 layer_width: float = 1.0, sampling_rate: float = 10.0, importance_prune_threshold: float = 0.5,
                 bandwidth_reduction: float = 0.9):
        print("[%s] Prepare network processing for network of size: %s" % (LOG_SOURCE, layer_nodes))
        self.layer_nodes: List[int] = layer_nodes
        self.layer_distance: float = layer_distance
        self.layer_width: float = layer_width
        self.bandwidth_reduction = bandwidth_reduction

        self.network: NetworkModel = NetworkModel(self.layer_nodes, self.layer_width, self.layer_distance, layer_data,
                                                  importance_prune_threshold)
        self.sample_length: float = self.network.layer_width / sampling_rate
        self.grid_cell_size: float = self.sample_length / 3.0
        self.sample_radius: float = self.sample_length * 2.0

        self.grid: Grid = Grid(Vector3([self.grid_cell_size, self.grid_cell_size, self.grid_cell_size]),
                               self.network.bounding_volume)

        self.node_processor: NodeProcessor = NodeProcessor()
        self.node_processor.set_data(self.network)
        self.node_renderer: NodeRenderer = NodeRenderer(self.node_processor, self.grid)

        self.edge_processor: EdgeProcessor = EdgeProcessor(self.sample_length)
        self.edge_processor.set_data(self.network)
        self.edge_processor.init_sample_edge()
        self.edge_renderer: EdgeRenderer = EdgeRenderer(self.edge_processor, self.grid)

        self.grid_processor: GridProcessor = GridProcessor(self.grid, self.node_processor, self.edge_processor, 100.0,
                                                           self.network.average_node_distance,
                                                           self.network.average_edge_distance, self.bandwidth_reduction)
        self.grid_processor.calculate_position()
        self.grid_processor.calculate_edge_density()

        self.grid_renderer: GridRenderer = GridRenderer(self.grid_processor)

        self.action_finished: bool = False
        self.last_action_mode: int = 0

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
        self.grid_processor.reset(self.network.average_node_distance, self.network.average_edge_distance)

    def process(self, window: Window, action_mode: int, smoothing: bool = False):
        if self.last_action_mode is not action_mode:
            if action_mode == 0:
                print("[%s] Resample %i edges" % (LOG_SOURCE, len(self.edge_processor.edges)))
                self.edge_processor.sample_edges()
            else:
                self.action_finished = False
                self.grid_processor.reset()

        self.edge_processor.check_limits(window.cam.view)

        if action_mode is not 0 and not self.action_finished:
            if action_mode > 3:
                print("[%s] Resample %i edges" % (LOG_SOURCE, len(self.edge_processor.edges)))
                self.edge_processor.sample_edges()

            if action_mode == 1:
                print("[%s] Advect %i nodes, iteration %i" % (
                    LOG_SOURCE, len(self.node_processor.nodes), self.grid_processor.node_iteration))
                self.grid_processor.clear_buffer()
                self.grid_processor.calculate_node_density()
                if self.grid_processor.advection_direction < 0:
                    self.grid_processor.advection_direction = 1.0
                self.grid_processor.node_advect()
                if self.grid_processor.node_limit_reached:
                    self.action_finished = True
            elif action_mode == 2:
                print("[%s] Diverge %i nodes, iteration %i" % (
                    LOG_SOURCE, len(self.node_processor.nodes), self.grid_processor.node_iteration))
                self.grid_processor.clear_buffer()
                self.grid_processor.calculate_node_density()
                if self.grid_processor.advection_direction > 0:
                    self.grid_processor.advection_direction = -1.0
                self.grid_processor.node_advect()
                if self.grid_processor.node_limit_reached:
                    self.action_finished = True
            elif action_mode == 3:
                print("[%s] Randomize %i nodes" % (LOG_SOURCE, len(self.node_processor.nodes)))
                self.node_processor.node_noise(self.sample_length, 0.5)
            if action_mode == 4:
                print("[%s] Advect %i edges, iteration %i" % (
                    LOG_SOURCE, len(self.edge_processor.edges), self.grid_processor.edge_iteration))
                self.grid_processor.clear_buffer()
                self.grid_processor.calculate_edge_density()
                if self.grid_processor.advection_direction < 0:
                    self.grid_processor.advection_direction = 1.0
                self.grid_processor.sample_advect()
                if self.grid_processor.edge_limit_reached:
                    self.action_finished = True
            elif action_mode == 5:
                print("[%s] Diverge %i edges, iteration %i" % (
                    LOG_SOURCE, len(self.edge_processor.edges), self.grid_processor.edge_iteration))
                self.grid_processor.clear_buffer()
                self.grid_processor.calculate_edge_density()
                if self.grid_processor.edge_bandwidth > 0:
                    self.grid_processor.advection_direction = -1.0
                self.grid_processor.sample_advect()
                if self.grid_processor.edge_limit_reached:
                    self.action_finished = True
            elif action_mode == 6:
                print("[%s] Randomize %i edges" % (LOG_SOURCE, len(self.edge_processor.edges)))
                self.edge_processor.sample_noise(0.5)
                self.grid_processor.reset()

            if action_mode > 3:
                if smoothing:
                    print("[%s] Smooth %i edges" % (LOG_SOURCE, len(self.edge_processor.edges)))
                    for i in range(10):
                        self.edge_processor.sample_smooth()

        self.last_action_mode = action_mode

    def render(self, window: Window, edge_render_mode: int, grid_render_mode: int, node_render_mode: int,
               edge_render_options: Dict[str, float] = None, grid_render_options: Dict[str, float] = None,
               node_render_options: Dict[str, float] = None):

        clear_screen([1.0, 1.0, 1.0, 1.0])
        if window.gradient and grid_render_mode == 1:
            self.grid_renderer.render_cube(window, clear=False, swap=False, options=grid_render_options)
        elif window.gradient and grid_render_mode == 2:
            self.grid_renderer.render_point(window, clear=False, swap=False, options=grid_render_options)
        if edge_render_mode == 5:
            self.edge_renderer.render_point(window, clear=False, swap=False, options=edge_render_options)
        elif edge_render_mode == 4:
            self.edge_renderer.render_line(window, clear=False, swap=False, options=edge_render_options)
        elif edge_render_mode == 3:
            self.edge_renderer.render_ellipsoid_transparent(window, clear=False, swap=False,
                                                            options=edge_render_options)
        elif edge_render_mode == 2:
            self.edge_renderer.render_transparent_sphere(window, clear=False, swap=False, options=edge_render_options)
        elif edge_render_mode == 1:
            self.edge_renderer.render_sphere(window, clear=False, swap=False, options=edge_render_options)
        if node_render_mode == 3:
            self.node_renderer.render_point(window, clear=False, swap=False, options=node_render_options)
        elif node_render_mode == 2:
            self.node_renderer.render_transparent(window, clear=False, swap=False, options=node_render_options)
        elif node_render_mode == 1:
            self.node_renderer.render_sphere(window, clear=False, swap=False, options=node_render_options)

    def delete(self):
        self.node_processor.delete()
        self.node_renderer.delete()
        self.edge_processor.delete()
        self.edge_renderer.delete()
        self.grid_processor.delete()
        self.grid_renderer.delete()
