import math
from typing import List, Tuple
from pyrr import Matrix44
import numpy as np

from opengl_helper.buffer import SwappingBufferObject, BufferObject
from opengl_helper.compute_shader import ComputeShader, ComputeShaderHandler
from models import NetworkModel, Edge
from utility.performance import track_time
from opengl_helper.render_utility import VertexDataHandler

LOG_SOURCE: str = "PROCESSING"


class EdgeProcessor:
    def __init__(self, sample_length: float):
        self.sample_compute_shader: ComputeShader = ComputeShaderHandler().create("edge_sampler",
                                                                                  "edge_sample.comp")
        self.noise_compute_shader: ComputeShader = ComputeShaderHandler().create("edge_noise",
                                                                                 "edge_noise.comp")
        self.limit_compute_shader: ComputeShader = ComputeShaderHandler().create("edge_limits",
                                                                                 "edge_limits.comp")
        self.sample_buffer: SwappingBufferObject = SwappingBufferObject(True)
        self.limits_buffer: BufferObject = BufferObject(True)
        self.ssbo_handler: VertexDataHandler = VertexDataHandler([(self.sample_buffer, 0), (self.limits_buffer, 2)])

        self.edges: List[Edge] = []
        self.sampled: bool = False
        self.sample_length: float = sample_length

        self.point_count: int = 0
        self.nearest_view_z: int = -1000000
        self.farthest_view_z: int = 1000000
        self.max_sample_points: int = 0

    def set_data(self, network: NetworkModel):
        # generate edges
        self.edges = network.generate_edges()

        #  estimate a suitable sample size for buffer objects
        max_distance = network.generate_max_distance()
        self.max_sample_points = int((max_distance * 2.0) / self.sample_length)

        # generate and load initial data for the buffer
        initial_data: List[float] = []
        for edge in self.edges:
            initial_data.extend(edge.initial_data)
            initial_data.extend([0] * (self.max_sample_points * 4 - len(edge.initial_data)))
        transfer_data = np.array(initial_data, dtype=np.float32)
        self.sample_buffer.load(transfer_data)
        self.sample_buffer.swap()
        self.sample_buffer.load(transfer_data)

    @track_time
    def resize_sample_storage(self, new_max_samples: int):
        edge_sample_data = np.array(self.read_samples_from_sample_storage(raw=True, auto_resize_enabled=False),
                                    dtype=np.float32)
        edge_sample_data = edge_sample_data.reshape((len(self.edges), self.max_sample_points * 4))

        self.max_sample_points = new_max_samples

        buffer_data = []
        for i in range(len(self.edges)):
            edge_points: int = int(edge_sample_data[i][3])
            buffer_data.extend(edge_sample_data[i][None:(int(edge_points * 4))])
            buffer_data.extend([0] * (self.max_sample_points * 4 - edge_points * 4))

        transfer_data = np.array(buffer_data, dtype=np.float32)

        self.sample_buffer.load(transfer_data)
        self.sample_buffer.swap()
        self.sample_buffer.load(transfer_data)

    @track_time
    def sample_edges(self, sample_length: float = None):
        if sample_length is not None:
            self.sample_length = sample_length

        self.ssbo_handler.set()
        self.sample_compute_shader.set_uniform_data([('sample_length', self.sample_length, 'float')])
        self.sample_compute_shader.set_uniform_data([('max_sample_points', self.max_sample_points, 'int')])

        for i in range(math.ceil(len(self.edges) / self.sample_compute_shader.max_workgroup_size)):
            self.sample_compute_shader.set_uniform_data(
                [('work_group_offset', i * self.sample_compute_shader.max_workgroup_size, 'int')])

            if i == math.ceil(len(self.edges) / self.sample_compute_shader.max_workgroup_size) - 1:
                self.sample_compute_shader.compute(len(self.edges) % self.sample_compute_shader.max_workgroup_size)
            else:
                self.sample_compute_shader.compute(self.sample_compute_shader.max_workgroup_size)

        self.sample_buffer.swap()
        self.sampled = True

    @track_time
    def sample_noise(self, strength: float = 1.0):
        self.ssbo_handler.set()

        self.noise_compute_shader.set_uniform_data([('sample_length', self.sample_length, 'float')])
        self.noise_compute_shader.set_uniform_data([('noise_strength', strength, 'float')])
        self.noise_compute_shader.set_uniform_data([('max_sample_points', self.max_sample_points, 'int')])

        for i in range(math.ceil(len(self.edges) / self.noise_compute_shader.max_workgroup_size)):
            self.noise_compute_shader.set_uniform_data(
                [('work_group_offset', i * self.noise_compute_shader.max_workgroup_size, 'int')])

            if i == math.ceil(len(self.edges) / self.noise_compute_shader.max_workgroup_size) - 1:
                self.noise_compute_shader.compute(len(self.edges) % self.noise_compute_shader.max_workgroup_size)
            else:
                self.noise_compute_shader.compute(self.noise_compute_shader.max_workgroup_size)

        self.sample_buffer.swap()

    @track_time
    def read_samples_from_sample_storage(self, raw: bool = False, auto_resize_enabled: bool = True) -> List[float]:
        edge_sample_data = np.frombuffer(self.sample_buffer.read(), dtype=np.float32)
        if raw:
            return edge_sample_data

        edge_sample_data = edge_sample_data.reshape((len(self.edges), self.max_sample_points * 4))
        buffer_data = []
        max_edge_samples = 0
        self.point_count = 0
        for i in range(len(self.edges)):
            edge_points: int = int(edge_sample_data[i][3])
            buffer_data.extend(edge_sample_data[i][None:(int(edge_points * 4))])
            self.point_count += edge_points
            if edge_points > max_edge_samples:
                max_edge_samples = edge_points

        if auto_resize_enabled and max_edge_samples >= (self.max_sample_points - 5) * 0.8:
            self.resize_sample_storage(max_edge_samples * 2)

        return buffer_data

    @track_time
    def check_limits(self, view: Matrix44):
        self.limits_buffer.load(np.array([0, -1000000, 1000000, 0], dtype=int))

        self.ssbo_handler.set()

        self.limit_compute_shader.set_uniform_data([('view', view, 'mat4')])
        self.limit_compute_shader.set_uniform_data([('max_sample_points', self.max_sample_points, 'int')])

        for i in range(math.ceil(len(self.edges) / self.limit_compute_shader.max_workgroup_size)):
            self.limit_compute_shader.set_uniform_data(
                [('work_group_offset', i * self.limit_compute_shader.max_workgroup_size, 'int')])

            if i == math.ceil(len(self.edges) / self.limit_compute_shader.max_workgroup_size) - 1:
                self.limit_compute_shader.compute(len(self.edges) % self.limit_compute_shader.max_workgroup_size)
            else:
                self.limit_compute_shader.compute(self.limit_compute_shader.max_workgroup_size)

        limits: List[int] = np.frombuffer(self.limits_buffer.read(), dtype=int)
        self.point_count = limits[0]
        self.nearest_view_z = limits[1]
        self.farthest_view_z = limits[2]
        max_edge_samples = limits[3]
        if max_edge_samples >= (self.max_sample_points - 5) * 0.8:
            self.resize_sample_storage(int(max_edge_samples * 2))

    @track_time
    def get_near_far_from_view(self) -> Tuple[float, float]:
        return self.nearest_view_z / 1000.0, self.farthest_view_z / 1000.0

    @track_time
    def get_buffer_points(self) -> int:
        return int(self.sample_buffer.size / 16.0)
