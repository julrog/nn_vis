import logging
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from OpenGL.GL import glFinish

from definitions import ADDITIONAL_EDGE_BUFFER_DATA, pairwise
from models.edge import Edge
from models.network import NetworkModel
from opengl_helper.buffer import (BufferObject, SwappingBufferObject,
                                  get_buffer_settings)
from opengl_helper.compute_shader import ComputeShader
from opengl_helper.compute_shader_handler import ComputeShaderHandler
from opengl_helper.vertex_data_handler import VertexDataHandler
from processing.advection_process import AdvectionProgress
from utility.performance import track_time


class EdgeProcessor:
    def __init__(self, sample_length: float, max_edges_per_buffer: int = 1000, edge_importance_type: int = 0):
        shader_settings: Dict[str, str] = {
            'init_edge_sampler': 'edge/initial_edge_sample.comp',
            'edge_sampler': 'edge/edge_sample.comp',
            'edge_noise': 'edge/sample_noise.comp',
            'sample_smooth': 'edge/sample_smooth.comp',
            'edge_limits': 'edge/edge_limits.comp',
            'sample_copy': 'edge/sample_copy.comp',
        }
        for shader_name, path in shader_settings.items():
            ComputeShaderHandler().create(shader_name, path)

        self.max_edges_per_buffer: int = max_edges_per_buffer
        self.sample_buffer: List[List[SwappingBufferObject]] = []
        self.edge_buffer: List[List[BufferObject]] = []
        self.ssbo_handler: List[List[VertexDataHandler]] = []

        self.edge_count: int = 0
        self.layer_edge_count: List[int] = []
        self.layer_container_edge_count: List[List[int]] = []

        self.sampled: bool = False
        self.sample_length: float = sample_length
        self.point_count: int = 0
        self.nearest_view_z: int = -1000000
        self.farthest_view_z: int = 1000000
        self.max_sample_points: int = 0
        self.smooth_radius: float = 0.0
        self.edge_importance_type: int = edge_importance_type
        self.edge_min_importance: float = 0.0
        self.edge_max_importance: float = 1.0

    def set_data(self, network: NetworkModel):
        edges: List[List[List[Edge]]] = network.generate_filtered_edges(
            self.max_edges_per_buffer)
        self.edge_min_importance = network.edge_min_importance
        self.edge_max_importance = network.edge_max_importance

        self.layer_edge_count = [len(layer) for layer in edges]
        self.layer_container_edge_count = [
            [len(container) for container in layer] for layer in edges]
        self.edge_count = sum([sum(layer)
                              for layer in self.layer_container_edge_count])

        # calculate smoothing radius
        max_distance: float = network.generate_max_distance()
        self.smooth_radius = (
            int((max_distance * 2.0) / self.sample_length) * 8.0) / 100.0

        # read or calculate max sample point value for buffer objects
        if len(edges[0][0][0].sample_data) > 8:
            self.sampled = True
            self.max_sample_points = int(len(edges[0][0][0].sample_data) / 4)
        else:
            self.max_sample_points = int(
                (max_distance * 5.0) / self.sample_length) + 2

        if len(self.sample_buffer) > 0:
            self.delete()

        self.fill_buffer(edges, network.num_classes)

    def fill_buffer(self, edges: List[List[List[Edge]]], num_classes: int):
        for layer_data in edges:
            new_layer_sample_buffer: List[SwappingBufferObject] = []
            new_layer_edge_buffer: List[BufferObject] = []
            new_layer_ssbo_handler: List[VertexDataHandler] = []
            for edge_container in layer_data:
                new_sample_buffer: SwappingBufferObject = SwappingBufferObject(ssbo=True, object_size=4,
                                                                               render_data_size=[
                                                                                   4, 4],
                                                                               render_data_offset=[0, 4])

                object_size, render_data_offset, render_data_size = \
                    get_buffer_settings(
                        num_classes * 2, ADDITIONAL_EDGE_BUFFER_DATA)
                new_edge_buffer: BufferObject = BufferObject(ssbo=True, object_size=object_size,
                                                             render_data_size=render_data_size,
                                                             render_data_offset=render_data_offset)
                new_ssbo_handler: VertexDataHandler = VertexDataHandler(
                    [(new_sample_buffer, 0), (new_edge_buffer, 2)])

                initial_data: List[float] = []
                for edge in edge_container:
                    initial_data.extend(edge.sample_data)
                    if self.max_sample_points * 4 - len(edge.sample_data) > 0:
                        initial_data.extend(
                            [0] * (self.max_sample_points * 4 - len(edge.sample_data)))
                transfer_data = np.array(initial_data, dtype=np.float32)
                new_sample_buffer.load(transfer_data)
                new_sample_buffer.swap()
                new_sample_buffer.load(transfer_data)
                new_sample_buffer.swap()

                initial_data: List[float] = []
                for edge in edge_container:
                    initial_data.extend(edge.data)
                transfer_data = np.array(initial_data, dtype=np.float32)
                new_edge_buffer.load(transfer_data)

                new_layer_sample_buffer.append(new_sample_buffer)
                new_layer_edge_buffer.append(new_edge_buffer)
                new_layer_ssbo_handler.append(new_ssbo_handler)

            self.sample_buffer.append(new_layer_sample_buffer)
            self.edge_buffer.append(new_layer_edge_buffer)
            self.ssbo_handler.append(new_layer_ssbo_handler)

    def set_uniform(self, compute_shader: ComputeShader, uniforms: List[str]):
        uniform_data: List[Tuple[str, Any, Any]] = []
        if 'max_sample_points' in uniforms:
            uniform_data.append(
                ('max_sample_points', self.max_sample_points, 'int'))
        if 'sample_length' in uniforms:
            uniform_data.append(('sample_length', self.sample_length, 'float'))
        compute_shader.set_uniform_data(uniform_data)

    @track_time
    def resize_sample_storage(self, new_max_samples: int):
        logging.info('Resize buffer.')

        for i in range(len(self.sample_buffer)):
            for j in range(len(self.sample_buffer[i])):
                edge_sample_data = np.array(
                    self.read_samples_from_buffer(i, j), dtype=np.float32)
                edge_sample_data = edge_sample_data.reshape(
                    (self.get_edge_count(i, j), self.max_sample_points * 4))

                buffer_data = []
                for k in range(self.get_edge_count(i, j)):
                    edge_points: int = int(edge_sample_data[k][3])
                    buffer_data.extend(
                        edge_sample_data[k][None:(int(edge_points * 4))])
                    buffer_data.extend(
                        [0] * (new_max_samples * 4 - edge_points * 4))

                transfer_data = np.array(buffer_data, dtype=np.float32)

                self.sample_buffer[i][j].load(transfer_data)
                self.sample_buffer[i][j].swap()
                self.sample_buffer[i][j].load(transfer_data)
                self.sample_buffer[i][j].swap()

        self.max_sample_points = new_max_samples

    def run_compute(self, compute_shader: ComputeShader, compute_width_func: Callable, wait_for_compute: bool = False):
        for i in range(len(self.sample_buffer)):
            for j in range(len(self.sample_buffer[i])):
                self.ssbo_handler[i][j].set()
                compute_shader.compute(compute_width_func(i, j))
                self.sample_buffer[i][j].swap()
                compute_shader.barrier()
                if wait_for_compute:
                    glFinish()

    def copy(self):
        copy: ComputeShader = ComputeShaderHandler().get('sample_copy')
        self.set_uniform(copy, ['max_sample_points'])
        self.run_compute(copy, self.get_buffer_points)

    def set_edge_sample(self, compute_shader: ComputeShader, sample_length: float = None):
        if sample_length is not None:
            self.sample_length = sample_length
        self.set_uniform(compute_shader, [
                         'max_sample_points', 'sample_length'])
        self.run_compute(compute_shader, self.get_edge_count)
        self.copy()
        self.sampled = True

    @track_time
    def init_sample_edge(self, sample_length: float = None):
        init: ComputeShader = ComputeShaderHandler().get('init_edge_sampler')
        self.set_edge_sample(init, sample_length)

    @track_time
    def sample_edges(self, sample_length: float = None):
        sample: ComputeShader = ComputeShaderHandler().get('edge_sampler')
        self.set_edge_sample(sample, sample_length)

    @track_time
    def sample_noise(self, strength: float = 1.0, move_start_end: int = 0):
        noise: ComputeShader = ComputeShaderHandler().get('edge_noise')
        self.set_uniform(noise, ['max_sample_points', 'sample_length'])
        noise.set_uniform_data([
            ('noise_strength', strength, 'float'),
            ('move_start_end', move_start_end, 'int')
        ])
        self.run_compute(noise, self.get_edge_count)

    @track_time
    def sample_smooth(self, advection_status: AdvectionProgress, wait_for_compute: bool = False):
        smooth: ComputeShader = ComputeShaderHandler().get('sample_smooth')
        self.set_uniform(smooth, ['max_sample_points'])
        smooth.set_uniform_data(
            [('bandwidth_reduction', advection_status.get_bandwidth_reduction(), 'float')])
        self.run_compute(smooth, self.get_buffer_points, wait_for_compute)

    @track_time
    def check_limits(self, check_resize: bool = False):
        limit: ComputeShader = ComputeShaderHandler().get('edge_limits')
        self.set_uniform(limit, ['max_sample_points'])
        self.point_count = 0
        max_edge_samples: float = 0
        for i in range(len(self.edge_buffer)):
            for j in range(len(self.edge_buffer[i])):
                self.ssbo_handler[i][j].set()
                limit.compute(self.get_edge_count(i, j), barrier=True)
                if check_resize:
                    limits: List[int] = np.frombuffer(
                        self.edge_buffer[i][j].read(), dtype=np.float32)
                    for samples in pairwise(limits, 28):
                        self.point_count += samples
                        if samples > max_edge_samples:
                            max_edge_samples = samples
        if check_resize:
            if max_edge_samples * 1.1 >= (self.max_sample_points - 1):
                self.resize_sample_storage(int(max_edge_samples * 1.1))

    @track_time
    def read_edges_from_buffer(self, layer: int, container: int) -> np.array:
        return np.frombuffer(self.edge_buffer[layer][container].read(), dtype=np.float32)

    @track_time
    def read_samples_from_buffer(self, layer: int, container: int) -> np.array:
        return np.frombuffer(self.sample_buffer[layer][container].read(), dtype=np.float32)

    @track_time
    def read_edges_from_all_buffer(self) -> List[List[np.array]]:
        return [[np.frombuffer(buffer.read(), dtype=np.float32) for buffer in layer_buffer] for layer_buffer in
                self.edge_buffer]

    @track_time
    def read_samples_from_all_buffer(self) -> List[List[np.array]]:
        return [[np.frombuffer(buffer.read(), dtype=np.float32) for buffer in layer_buffer] for layer_buffer in
                self.sample_buffer]

    def get_buffer_points(self, layer: int, container: int) -> int:
        return int(self.sample_buffer[layer][container].size / 16.0)

    def get_all_buffer_points(self, layer: int, container: int) -> int:
        return int(self.sample_buffer[layer][container].size / 16.0)

    def get_edge_count(self, layer: int = None, container: int = None) -> int:
        if layer is None and container is None:
            return self.edge_count
        elif container is None:
            return self.layer_edge_count[layer]
        else:
            return self.layer_container_edge_count[layer][container]

    def delete(self):
        for layer_buffer in self.sample_buffer:
            for container_buffer in layer_buffer:
                container_buffer.delete()
        self.sample_buffer = []

        for layer_buffer in self.edge_buffer:
            for container_buffer in layer_buffer:
                container_buffer.delete()
        self.edge_buffer = []

        for layer_ssbo_handler in self.ssbo_handler:
            for container_ssbo_handler in layer_ssbo_handler:
                container_ssbo_handler.delete()
        self.ssbo_handler = []
