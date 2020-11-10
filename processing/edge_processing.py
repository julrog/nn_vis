from typing import List
import numpy as np
from definitions import pairwise
from models.edge import Edge
from models.network import NetworkModel
from opengl_helper.buffer import SwappingBufferObject, BufferObject
from opengl_helper.compute_shader import ComputeShader, ComputeShaderHandler
from processing.advection_process import AdvectionProgress
from utility.performance import track_time
from opengl_helper.render_utility import VertexDataHandler
from OpenGL.GL import *


LOG_SOURCE: str = "EDGE_PROCESSING"


class EdgeProcessor:
    def __init__(self, sample_length: float, max_edges_per_buffer: int = 1000, edge_importance_type: int = 0):
        self.init_compute_shader: ComputeShader = ComputeShaderHandler().create("init_edge_sampler",
                                                                                "edge/initial_edge_sample.comp")
        self.sample_compute_shader: ComputeShader = ComputeShaderHandler().create("edge_sampler",
                                                                                  "edge/edge_sample.comp")
        self.noise_compute_shader: ComputeShader = ComputeShaderHandler().create("edge_noise",
                                                                                 "edge/sample_noise.comp")
        self.smooth_compute_shader: ComputeShader = ComputeShaderHandler().create("sample_smooth",
                                                                                  "edge/sample_smooth.comp")
        self.limit_compute_shader: ComputeShader = ComputeShaderHandler().create("edge_limits",
                                                                                 "edge/edge_limits.comp")
        self.sample_copy_compute_shader: ComputeShader = ComputeShaderHandler().create("sample_copy",
                                                                                       "edge/sample_copy.comp")

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
        edges: List[List[List[Edge]]] = network.generate_filtered_edges(self.max_edges_per_buffer)
        self.edge_min_importance = network.edge_min_importance
        self.edge_max_importance = network.edge_max_importance

        self.layer_edge_count = [len(layer) for layer in edges]
        self.layer_container_edge_count = [[len(container) for container in layer] for layer in edges]
        self.edge_count = sum([sum(layer) for layer in self.layer_container_edge_count])

        # calculate smoothing radius
        max_distance: float = network.generate_max_distance()
        self.smooth_radius = (int((max_distance * 2.0) / self.sample_length) * 8.0) / 100.0

        # read or calculate max sample point value for buffer objects
        if len(edges[0][0][0].sample_data) > 8:
            self.sampled = True
            self.max_sample_points = int(len(edges[0][0][0].sample_data) / 4)
        else:
            self.max_sample_points = int((max_distance * 5.0) / self.sample_length) + 2

        # generate and load initial data for the buffer
        if len(self.sample_buffer) > 0:
            self.delete()
        for layer_data in edges:
            new_layer_sample_buffer: List[SwappingBufferObject] = []
            new_layer_edge_buffer: List[BufferObject] = []
            new_layer_ssbo_handler: List[VertexDataHandler] = []
            for edge_container in layer_data:
                new_sample_buffer: SwappingBufferObject = SwappingBufferObject(ssbo=True, object_size=4,
                                                                               render_data_size=[4, 4],
                                                                               render_data_offset=[0, 4])
                new_edge_buffer: BufferObject = BufferObject(ssbo=True, object_size=28,
                                                             render_data_size=[4, 4, 4, 4, 4, 4, 4],
                                                             render_data_offset=[0, 4, 8, 12, 16, 20, 24])
                new_ssbo_handler: VertexDataHandler = VertexDataHandler([(new_sample_buffer, 0), (new_edge_buffer, 2)])

                initial_data: List[float] = []
                for edge in edge_container:
                    initial_data.extend(edge.sample_data)
                    if self.max_sample_points * 4 - len(edge.sample_data) > 0:
                        initial_data.extend([0] * (self.max_sample_points * 4 - len(edge.sample_data)))
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

    @track_time
    def resize_sample_storage(self, new_max_samples: int):
        print("[%s] Resize buffer." % LOG_SOURCE)

        for i in range(len(self.sample_buffer)):
            for j in range(len(self.sample_buffer[i])):
                edge_sample_data = np.array(self.read_samples_from_buffer(i, j), dtype=np.float32)
                edge_sample_data = edge_sample_data.reshape((self.get_edge_count(i, j), self.max_sample_points * 4))

                buffer_data = []
                for k in range(self.get_edge_count(i, j)):
                    edge_points: int = int(edge_sample_data[k][3])
                    buffer_data.extend(edge_sample_data[k][None:(int(edge_points * 4))])
                    buffer_data.extend([0] * (new_max_samples * 4 - edge_points * 4))

                transfer_data = np.array(buffer_data, dtype=np.float32)

                self.sample_buffer[i][j].load(transfer_data)
                self.sample_buffer[i][j].swap()
                self.sample_buffer[i][j].load(transfer_data)
                self.sample_buffer[i][j].swap()

        self.max_sample_points = new_max_samples

    @track_time
    def init_sample_edge(self, sample_length: float = None):
        if sample_length is not None:
            self.sample_length = sample_length

        self.init_compute_shader.set_uniform_data([
            ('max_sample_points', self.max_sample_points, 'int'),
            ('sample_length', self.sample_length, 'float')
        ])

        for i in range(len(self.sample_buffer)):
            for j in range(len(self.sample_buffer[i])):
                self.ssbo_handler[i][j].set()
                self.init_compute_shader.compute(self.get_edge_count(i, j))
                self.sample_buffer[i][j].swap()
                self.init_compute_shader.barrier()
                self.ssbo_handler[i][j].set()
                self.sample_copy_compute_shader.compute(self.get_buffer_points(i, j))
                self.sample_buffer[i][j].swap()
                self.sample_copy_compute_shader.barrier()

        self.sampled = True

    @track_time
    def sample_edges(self, sample_length: float = None):
        if sample_length is not None:
            self.sample_length = sample_length

        self.sample_compute_shader.set_uniform_data([
            ('max_sample_points', self.max_sample_points, 'int'),
            ('sample_length', self.sample_length, 'float')
        ])

        for i in range(len(self.sample_buffer)):
            for j in range(len(self.sample_buffer[i])):
                self.ssbo_handler[i][j].set()
                self.sample_compute_shader.compute(self.get_edge_count(i, j))
                self.sample_buffer[i][j].swap()
                self.sample_compute_shader.barrier()
                self.ssbo_handler[i][j].set()
                self.sample_copy_compute_shader.compute(self.get_buffer_points(i, j))
                self.sample_buffer[i][j].swap()
                self.sample_copy_compute_shader.barrier()

        self.sampled = True

    @track_time
    def sample_noise(self, strength: float = 1.0, move_start_end: int = 0):
        self.noise_compute_shader.set_uniform_data([
            ('max_sample_points', self.max_sample_points, 'int'),
            ('sample_length', self.sample_length, 'float'),
            ('noise_strength', strength, 'float'),
            ('move_start_end', move_start_end, 'int')
        ])

        for i in range(len(self.sample_buffer)):
            for j in range(len(self.sample_buffer[i])):
                self.ssbo_handler[i][j].set()
                self.noise_compute_shader.compute(self.get_edge_count(i, j))
                self.sample_buffer[i][j].swap()
        self.noise_compute_shader.barrier()

    @track_time
    def sample_smooth(self, advection_status: AdvectionProgress, wait_for_compute: bool = False):
        self.smooth_compute_shader.set_uniform_data([
            ('max_sample_points', self.max_sample_points, 'int'),
            ('bandwidth_reduction', advection_status.get_bandwidth_reduction(), 'float')
        ])

        for i in range(len(self.sample_buffer)):
            for j in range(len(self.sample_buffer[i])):
                self.ssbo_handler[i][j].set()
                self.smooth_compute_shader.compute(self.get_buffer_points(i, j))
                self.sample_buffer[i][j].swap()
                if wait_for_compute:
                    glFinish()
        self.smooth_compute_shader.barrier()

    @track_time
    def check_limits(self, check_resize: bool = False):
        self.limit_compute_shader.set_uniform_data([('max_sample_points', self.max_sample_points, 'int')])
        self.point_count = 0
        max_edge_samples: float = 0
        for i in range(len(self.edge_buffer)):
            for j in range(len(self.edge_buffer[i])):
                self.ssbo_handler[i][j].set()
                self.limit_compute_shader.compute(self.get_edge_count(i, j), barrier=True)

                if check_resize:
                    limits: List[int] = np.frombuffer(self.edge_buffer[i][j].read(), dtype=np.float32)

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
