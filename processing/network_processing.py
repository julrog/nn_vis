import logging
from enum import IntEnum
from typing import List

import numpy as np
from OpenGL.GL import *
from progressbar import ProgressBar
from pyrr import Vector3

from data.data_handler import ImportanceDataHandler, ProcessedNNHandler
from models.grid import Grid
from models.network import NetworkModel
from opengl_helper.compute_shader_handler import ComputeShaderHandler
from opengl_helper.render_utility import clear_screen
from opengl_helper.shader_handler import RenderShaderHandler
from processing.advection_process import AdvectionProgress
from processing.edge_processing import EdgeProcessor
from processing.grid_processing import GridProcessor
from processing.node_processing import NodeProcessor
from processing.processing_config import ProcessingConfig
from rendering.edge_rendering import EdgeRenderer
from rendering.grid_rendering import GridRenderer
from rendering.node_rendering import NodeRenderer
from rendering.rendering_config import RenderingConfig
from utility.camera import Camera


class NetworkProcess(IntEnum):
    RESET = 0
    NODE_ADVECT = 1
    NODE_DIVERGE = 2
    NODE_RANDOM = 3
    EDGE_ADVECT = 4
    EDGE_DIVERGE = 5
    EDGE_RANDOM = 6


class NetworkProcessor:
    def __init__(self, layer_nodes: List[int],
                 processing_config: ProcessingConfig,
                 importance_data: ImportanceDataHandler = None,
                 processed_nn: ProcessedNNHandler = None):
        logging.info("Prepare network processing for network of size: %s" % layer_nodes)
        self.layer_nodes: List[int] = layer_nodes
        self.layer_distance: float = processing_config["layer_distance"]
        self.layer_width: float = processing_config["layer_width"]

        logging.info("Create network model...")
        self.network: NetworkModel = NetworkModel(self.layer_nodes, self.layer_width, self.layer_distance,
                                                  importance_data, processed_nn, processing_config["prune_percentage"])
        self.sample_length: float = self.network.layer_width / processing_config["sampling_rate"]
        self.grid_cell_size: float = self.sample_length / 3.0
        self.sample_radius: float = self.sample_length * 2.0

        RenderShaderHandler().set_classification_number(self.network.num_classes)
        ComputeShaderHandler().set_classification_number(self.network.num_classes)

        self.node_advection_status: AdvectionProgress = AdvectionProgress(self.network.average_node_distance,
                                                                          processing_config["node_bandwidth_reduction"],
                                                                          self.grid_cell_size * 2.0)
        self.edge_advection_status: AdvectionProgress = AdvectionProgress(self.network.average_edge_distance,
                                                                          processing_config["edge_bandwidth_reduction"],
                                                                          self.grid_cell_size * 2.0)
        self.edge_importance_type: int = processing_config["edge_importance_type"]

        logging.info("Create grid...")
        self.grid: Grid = Grid(Vector3([self.grid_cell_size, self.grid_cell_size, self.grid_cell_size]),
                               self.network.bounding_volume, self.layer_distance)

        logging.info("Prepare node processing...")
        self.node_processor: NodeProcessor = NodeProcessor(self.network)
        self.node_renderer: NodeRenderer = NodeRenderer(self.node_processor, self.grid)

        logging.info("Prepare edge processing...")
        self.edge_processor: EdgeProcessor = EdgeProcessor(self.sample_length,
                                                           edge_importance_type=self.edge_importance_type)
        self.edge_processor.set_data(self.network)
        if not self.edge_processor.sampled:
            self.edge_processor.init_sample_edge()
        self.edge_renderer: EdgeRenderer = EdgeRenderer(self.edge_processor, self.grid)

        logging.info("Prepare grid processing...")
        self.grid_processor: GridProcessor = GridProcessor(self.grid, self.node_processor, self.edge_processor, 10000.0)
        self.grid_processor.calculate_position()
        self.grid_renderer: GridRenderer = GridRenderer(self.grid_processor)

        self.action_finished: bool = False
        self.last_action_mode: NetworkProcess = NetworkProcess.RESET

        self.edge_smoothing: bool = processing_config["smoothing"]
        self.edge_smoothing_iterations: int = processing_config["smoothing_iterations"]
        self.bar: ProgressBar or None = None

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
        self.edge_processor.check_limits()

    def process(self, action_mode: NetworkProcess):
        if self.last_action_mode is not action_mode:
            if action_mode == NetworkProcess.RESET:
                logging.info("Resample %i edges" % self.edge_processor.get_edge_count())
                self.edge_processor.sample_edges()
                self.edge_processor.check_limits()
            else:
                self.action_finished = False
                self.node_advection_status.reset()
                self.edge_advection_status.reset()

        if action_mode is not NetworkProcess.RESET and not self.action_finished:
            if action_mode >= NetworkProcess.EDGE_ADVECT:
                self.edge_processor.sample_edges()
                self.edge_processor.check_limits()

            if action_mode == NetworkProcess.NODE_ADVECT:
                self.node_advection()
            elif action_mode == NetworkProcess.NODE_DIVERGE:
                self.node_advection(True)
            elif action_mode == NetworkProcess.NODE_RANDOM:
                self.node_processor.node_noise(self.sample_length, 0.5)
            if action_mode == NetworkProcess.EDGE_ADVECT:
                self.edge_advection()
            elif action_mode == NetworkProcess.EDGE_DIVERGE:
                self.edge_advection(True)
            elif action_mode == NetworkProcess.EDGE_RANDOM:
                self.edge_processor.sample_noise(3.0)

            if action_mode >= NetworkProcess.EDGE_ADVECT:
                if self.edge_smoothing:
                    for i in range(self.edge_smoothing_iterations):
                        glFinish()
                        self.edge_processor.sample_smooth(self.edge_advection_status, True)
                        glFinish()
        else:
            self.edge_processor.check_limits()

        self.last_action_mode = action_mode
        glFinish()

    def smooth_edges(self):
        glFinish()
        self.edge_processor.sample_smooth(self.edge_advection_status, True)
        glFinish()

    def node_advection(self, reverse: bool = False):
        if self.bar is None:
            logging.info("Advect %i nodes" % len(self.node_processor.nodes))
            self.bar = ProgressBar(max_value=self.node_advection_status.get_max_iterations())
            self.bar.start()

        if reverse:
            self.node_advection_status.advection_direction = -1.0
        else:
            self.node_advection_status.advection_direction = 1.0

        self.grid_processor.clear_buffer()
        self.grid_processor.calculate_node_density(self.node_advection_status)
        self.grid_processor.node_advect(self.node_advection_status)

        self.node_advection_status.iterate()
        self.bar.update(self.node_advection_status.iteration)

        if self.node_advection_status.limit_reached:
            self.reset_edges()
            self.action_finished = True
            self.bar.finish()
            self.bar = None

    def edge_advection(self, reverse: bool = False):
        if self.bar is None:
            logging.info("Advect %i edges" % self.edge_processor.get_edge_count())
            self.bar = ProgressBar(max_value=self.edge_advection_status.get_max_iterations())
            self.bar.start()

        if reverse:
            self.edge_advection_status.advection_direction = -1.0
        else:
            self.edge_advection_status.advection_direction = 1.0

        for layer in range(len(self.network.layer) - 1):
            self.grid_processor.clear_buffer()
            self.grid_processor.calculate_edge_density(layer, self.edge_advection_status, True)
            self.grid_processor.sample_advect(layer, self.edge_advection_status, True)

        self.edge_advection_status.iterate()
        self.bar.update(self.edge_advection_status.iteration)

        if self.edge_advection_status.limit_reached:
            self.action_finished = True
            self.bar.finish()
            self.bar = None

    def render(self, cam: Camera, config: RenderingConfig, show_class: int = 0):
        clear_screen([1.0, 1.0, 1.0, 1.0])
        if config["grid_render_mode"] == 1:
            self.grid_renderer.render("grid_cube", cam, config=config)
        elif config["grid_render_mode"] == 2:
            self.grid_renderer.render("grid_point", cam, config=config)
        if config["edge_render_mode"] == 5:
            self.edge_renderer.render("sample_point", cam, config=config, show_class=show_class)
        elif config["edge_render_mode"] == 4:
            self.edge_renderer.render("sample_line", cam, config=config, show_class=show_class)
        elif config["edge_render_mode"] == 3:
            self.edge_renderer.render("sample_ellipsoid_transparent", cam, config=config, show_class=show_class)
        elif config["edge_render_mode"] == 2:
            self.edge_renderer.render("sample_transparent_sphere", cam, config=config, show_class=show_class)
        elif config["edge_render_mode"] == 1:
            self.edge_renderer.render("sample_sphere", cam, config=config, show_class=show_class)
        if config["node_render_mode"] == 3:
            self.node_renderer.render("node_point", cam, config=config, show_class=show_class)
        elif config["node_render_mode"] == 2:
            self.node_renderer.render("node_transparent_sphere", cam, config=config, show_class=show_class)
        elif config["node_render_mode"] == 1:
            self.node_renderer.render("node_sphere", cam, config=config, show_class=show_class)

    def save_model(self, file_path: str):
        layer_data: List[int] = self.network.layer
        logging.info("Reading nodes from buffer...")
        node_data: List[float] = self.node_processor.read_nodes_from_buffer(raw=True)
        logging.info("Reading edges from buffer...")
        edge_data: List[List[np.array]] = self.edge_processor.read_edges_from_all_buffer()
        logging.info("Reading examples from buffer...")
        sample_data: List[List[np.array]] = self.edge_processor.read_samples_from_all_buffer()
        max_sample_points: int = self.edge_processor.max_sample_points
        logging.info("Saving processed network data...")
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
