import time

import numpy as np
from typing import List, Dict

from pyrr import Vector3

from data.data_handler import ImportanceDataHandler, ProcessedNNHandler
from models.grid import Grid
from models.network import NetworkModel
from opengl_helper.render_utility import clear_screen
from processing.advection_process import AdvectionProgress
from processing.edge_processing import EdgeProcessor
from processing.grid_processing import GridProcessor
from processing.node_processing import NodeProcessor
from rendering.edge_rendering import EdgeRenderer
from rendering.grid_rendering import GridRenderer
from rendering.node_rendering import NodeRenderer
from utility.window import Window
from OpenGL.GL import *

LOG_SOURCE: str = "NETWORK_PROCESSING"


class NetworkProcessor:
    def __init__(self, layer_nodes: List[int],
                 importance_data: ImportanceDataHandler = None,
                 processed_nn: ProcessedNNHandler = None,
                 layer_distance: float = 1.0, layer_width: float = 1.0, sampling_rate: float = 10.0,
                 prune_percentage: float = 0.1, node_bandwidth_reduction: float = 0.98,
                 edge_bandwidth_reduction: float = 0.9, edge_importance_type: int = 0):
        print("[%s] Prepare network processing for network of size: %s" % (LOG_SOURCE, layer_nodes))
        self.layer_nodes: List[int] = layer_nodes
        self.layer_distance: float = layer_distance
        self.layer_width: float = layer_width

        print("[%s] Create network model..." % LOG_SOURCE)
        self.network: NetworkModel = NetworkModel(self.layer_nodes, self.layer_width, self.layer_distance,
                                                  importance_data, processed_nn, prune_percentage)
        self.sample_length: float = self.network.layer_width / sampling_rate
        self.grid_cell_size: float = self.sample_length / 3.0
        self.sample_radius: float = self.sample_length * 2.0

        self.node_advection_status: AdvectionProgress = AdvectionProgress(self.network.average_node_distance,
                                                                          node_bandwidth_reduction,
                                                                          self.grid_cell_size * 1.0)
        self.edge_advection_status: AdvectionProgress = AdvectionProgress(self.network.average_edge_distance,
                                                                          edge_bandwidth_reduction,
                                                                          self.grid_cell_size * 1.0)
        self.edge_importance_type: int = edge_importance_type

        print("[%s] Create grid..." % LOG_SOURCE)
        self.grid: Grid = Grid(Vector3([self.grid_cell_size, self.grid_cell_size, self.grid_cell_size]),
                               self.network.bounding_volume, self.layer_distance)

        print("[%s] Prepare node processing..." % LOG_SOURCE)
        self.node_processor: NodeProcessor = NodeProcessor()
        self.node_processor.set_data(self.network)
        self.node_renderer: NodeRenderer = NodeRenderer(self.node_processor, self.grid)

        print("[%s] Prepare edge processing..." % LOG_SOURCE)
        self.edge_processor: EdgeProcessor = EdgeProcessor(self.sample_length,
                                                           edge_importance_type=edge_importance_type)
        self.edge_processor.set_data(self.network)
        if not self.edge_processor.sampled:
            self.edge_processor.init_sample_edge()
        self.edge_renderer: EdgeRenderer = EdgeRenderer(self.edge_processor, self.grid)

        print("[%s] Prepare grid processing..." % LOG_SOURCE)
        self.grid_processor: GridProcessor = GridProcessor(self.grid, self.node_processor, self.edge_processor, 200.0)
        self.grid_processor.calculate_position()
        self.grid_renderer: GridRenderer = GridRenderer(self.grid_processor)

        self.action_finished: bool = False
        self.last_action_mode: int = 0

    def reset_edges(self):
        self.edge_processor.delete()
        self.edge_renderer.delete()

        self.node_processor.read_nodes_from_buffer()
        self.network.set_nodes(self.node_processor.nodes)
        self.edge_processor = EdgeProcessor(self.sample_length, edge_importance_type=self.edge_importance_type)
        self.edge_processor.set_data(self.network)
        self.edge_processor.init_sample_edge()
        self.edge_renderer = EdgeRenderer(self.edge_processor, self.grid)

        self.grid_processor.set_new_edge_processor(self.edge_processor)
        self.node_advection_status.reset()
        self.edge_advection_status.reset()

    def process(self, action_mode: int, smoothing: bool = False):
        if self.last_action_mode is not action_mode:
            if action_mode == 0:
                print("[%s] Resample %i edges" % (LOG_SOURCE, self.edge_processor.get_edge_count()))
                self.edge_processor.sample_edges()
                self.edge_processor.check_limits()
            else:
                self.action_finished = False
                self.node_advection_status.reset()
                self.edge_advection_status.reset()

        if action_mode is not 0 and not self.action_finished:
            if action_mode > 3:
                print("[%s] Resample %i edges" % (LOG_SOURCE, self.edge_processor.get_edge_count()))
                self.edge_processor.sample_edges()
                self.edge_processor.check_limits()

            if action_mode == 1:
                self.node_advection()
            elif action_mode == 2:
                self.node_advection(True)
            elif action_mode == 3:
                print("[%s] Randomize %i nodes" % (LOG_SOURCE, len(self.node_processor.nodes)))
                self.node_processor.node_noise(self.sample_length, 0.5)
            if action_mode == 4:
                self.edge_advection()
            elif action_mode == 5:
                self.edge_advection(True)
            elif action_mode == 6:
                print("[%s] Randomize %i edges" % (LOG_SOURCE, self.edge_processor.get_edge_count()))
                self.edge_processor.sample_noise(0.5)

            if action_mode > 3:
                if smoothing:
                    print("[%s] Smooth %i edges" % (LOG_SOURCE, self.edge_processor.get_edge_count()))
                    for i in range(7):
                        glFinish()
                        self.edge_processor.sample_smooth(self.edge_advection_status, True)
                        glFinish()
        else:
            self.edge_processor.check_limits(False)

        self.last_action_mode = action_mode
        glFinish()

    def node_advection(self, reverse: bool = False):
        print("[%s] Advect %i nodes, iteration %i" % (
            LOG_SOURCE, len(self.node_processor.nodes), self.node_advection_status.iteration))

        if reverse:
            self.node_advection_status.advection_direction = -1.0
        else:
            self.node_advection_status.advection_direction = 1.0

        self.grid_processor.clear_buffer()
        self.grid_processor.calculate_node_density(self.node_advection_status)
        self.grid_processor.node_advect(self.node_advection_status)

        self.node_advection_status.iterate()

        if self.node_advection_status.limit_reached:
            self.action_finished = True

    def edge_advection(self, reverse: bool = False):
        print("[%s] Advect %i edges, iteration %i" % (
            LOG_SOURCE, self.edge_processor.get_edge_count(), self.edge_advection_status.iteration))

        if reverse:
            self.edge_advection_status.advection_direction = -1.0
        else:
            self.edge_advection_status.advection_direction = 1.0

        for layer in range(len(self.network.layer) - 1):
            self.grid_processor.clear_buffer()
            self.grid_processor.calculate_edge_density(layer, self.edge_advection_status, True)
            self.grid_processor.sample_advect(layer, self.edge_advection_status, True)

        self.edge_advection_status.iterate()

        if self.edge_advection_status.limit_reached:
            self.action_finished = True

    def render(self, window: Window, edge_render_mode: int, grid_render_mode: int, node_render_mode: int,
               edge_render_options: Dict[str, float] = None, grid_render_options: Dict[str, float] = None,
               node_render_options: Dict[str, float] = None, show_class: int = 0):
        clear_screen([1.0, 1.0, 1.0, 1.0])
        if window.gradient and grid_render_mode == 1:
            self.grid_renderer.render_cube(window, clear=False, swap=False, options=grid_render_options)
        elif window.gradient and grid_render_mode == 2:
            self.grid_renderer.render_point(window, clear=False, swap=False, options=grid_render_options)
        if edge_render_mode == 5:
            self.edge_renderer.render_point(window, clear=False, swap=False, options=edge_render_options,
                                            show_class=show_class)
        elif edge_render_mode == 4:
            self.edge_renderer.render_line(window, clear=False, swap=False, options=edge_render_options,
                                           show_class=show_class)
        elif edge_render_mode == 3:
            self.edge_renderer.render_ellipsoid_transparent(window, clear=False, swap=False,
                                                            options=edge_render_options, show_class=show_class)
        elif edge_render_mode == 2:
            self.edge_renderer.render_transparent_sphere(window, clear=False, swap=False, options=edge_render_options,
                                                         show_class=show_class)
        elif edge_render_mode == 1:
            self.edge_renderer.render_sphere(window, clear=False, swap=False, options=edge_render_options,
                                             show_class=show_class)
        if node_render_mode == 3:
            self.node_renderer.render_point(window, clear=False, swap=False, options=node_render_options,
                                            show_class=show_class)
        elif node_render_mode == 2:
            self.node_renderer.render_transparent(window, clear=False, swap=False, options=node_render_options,
                                                  show_class=show_class)
        elif node_render_mode == 1:
            self.node_renderer.render_sphere(window, clear=False, swap=False, options=node_render_options,
                                             show_class=show_class)

    def save_model(self, file_path: str):
        layer_data: List[int] = self.network.layer
        print("Reading nodes from buffer...")
        node_data: List[float] = self.node_processor.read_nodes_from_buffer(raw=True)
        print("Reading edges from buffer...")
        edge_data: List[List[np.array]] = self.edge_processor.read_edges_from_all_buffer()
        print("Reading samples from buffer...")
        sample_data: List[List[np.array]] = self.edge_processor.read_samples_from_all_buffer()
        max_sample_points: int = self.edge_processor.max_sample_points
        print("Saving processed network data...")
        np.savez(file_path, (layer_data, node_data, edge_data, sample_data, max_sample_points))

    def delete(self):
        self.node_processor.delete()
        self.node_renderer.delete()
        self.edge_processor.delete()
        self.edge_renderer.delete()
        self.grid_processor.delete()
        self.grid_renderer.delete()

    def get_node_mid(self) -> Vector3:
        self.node_processor.read_nodes_from_buffer()
        self.network.set_nodes(self.node_processor.nodes)
        return self.network.get_node_mid()
