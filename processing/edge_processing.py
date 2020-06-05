from typing import List
from pyrr import Matrix44
import numpy as np

from definitions import pairwise
from models.edge import Edge
from models.network import NetworkModel
from opengl_helper.buffer import SwappingBufferObject, BufferObject
from opengl_helper.compute_shader import ComputeShader, ComputeShaderHandler
from utility.performance import track_time
from opengl_helper.render_utility import VertexDataHandler

LOG_SOURCE: str = "EDGE_PROCESSING"


class EdgeProcessor:
    def __init__(self, sample_length: float):
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
        self.sample_buffer: SwappingBufferObject = SwappingBufferObject(ssbo=True, object_size=4,
                                                                        render_data_size=[4, 4],
                                                                        render_data_offset=[0, 4])
        self.edge_buffer: BufferObject = BufferObject(ssbo=True, object_size=28, render_data_size=[4, 4, 4, 4, 4, 4, 4],
                                                      render_data_offset=[0, 4, 8, 12, 16, 20, 24])
        self.ssbo_handler: VertexDataHandler = VertexDataHandler([(self.sample_buffer, 0), (self.edge_buffer, 2)])

        self.edges: List[Edge] = []
        self.sampled: bool = False
        self.sample_length: float = sample_length

        self.point_count: int = 0
        self.nearest_view_z: int = -1000000
        self.farthest_view_z: int = 1000000
        self.max_sample_points: int = 0
        self.smooth_radius: float = 0.0

    def set_data(self, network: NetworkModel):
        # generate edges
        self.edges = network.generate_edges()

        #  estimate a suitable sample size for buffer objects
        max_distance: float = network.generate_max_distance()
        self.max_sample_points = int((max_distance * 2.0) / self.sample_length) + 2
        self.smooth_radius = (self.max_sample_points * 8.0)/100.0

        # generate and load initial data for the buffer
        initial_data: List[float] = []
        for edge in self.edges:
            initial_data.extend(edge.initial_data)
            initial_data.extend([0] * (self.max_sample_points * 4 - len(edge.initial_data)))
        transfer_data = np.array(initial_data, dtype=np.float32)
        self.sample_buffer.load(transfer_data)
        self.sample_buffer.swap()
        self.sample_buffer.load(transfer_data)

        initial_data: List[float] = []
        for edge in self.edges:
            initial_data.extend(edge.data)
        transfer_data = np.array(initial_data, dtype=np.float32)
        self.edge_buffer.load(transfer_data)

    @track_time
    def resize_sample_storage(self, new_max_samples: int):
        print("[%s] Resize buffer." % LOG_SOURCE)
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
        self.sample_buffer.swap()

    @track_time
    def init_sample_edge(self, sample_length: float = None):
        if sample_length is not None:
            self.sample_length = sample_length

        self.ssbo_handler.set()
        self.init_compute_shader.set_uniform_data([
            ('max_sample_points', self.max_sample_points, 'int'),
            ('sample_length', self.sample_length, 'float')
        ])
        self.init_compute_shader.compute(len(self.edges))

        self.sample_buffer.swap()
        self.sampled = True

    @track_time
    def sample_edges(self, sample_length: float = None):
        if sample_length is not None:
            self.sample_length = sample_length

        self.ssbo_handler.set()
        self.sample_compute_shader.set_uniform_data([
            ('max_sample_points', self.max_sample_points, 'int'),
            ('sample_length', self.sample_length, 'float')
        ])
        self.sample_compute_shader.compute(len(self.edges))

        self.sample_buffer.swap()
        self.sampled = True

    @track_time
    def sample_noise(self, strength: float = 1.0, move_start_end: int = 0):
        self.ssbo_handler.set()

        self.noise_compute_shader.set_uniform_data([
            ('max_sample_points', self.max_sample_points, 'int'),
            ('sample_length', self.sample_length, 'float'),
            ('noise_strength', strength, 'float'),
            ('move_start_end', move_start_end, 'int')
        ])
        self.noise_compute_shader.compute(len(self.edges))

        self.sample_buffer.swap()

    @track_time
    def sample_smooth(self):
        self.ssbo_handler.set()

        self.smooth_compute_shader.set_uniform_data([
            ('max_sample_points', self.max_sample_points, 'int')
        ])
        self.smooth_compute_shader.compute(self.get_buffer_points())

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
    def check_limits(self, view: Matrix44, check_resize: bool = True):
        self.ssbo_handler.set()

        # self.limit_compute_shader.set_uniform_data([('view', view, 'mat4')])
        self.limit_compute_shader.set_uniform_data([('max_sample_points', self.max_sample_points, 'int')])
        self.limit_compute_shader.compute(len(self.edges))

        if check_resize:
            limits: List[int] = np.frombuffer(self.edge_buffer.read(), dtype=np.float32)

            self.point_count = 0
            max_edge_samples: float = 0
            for samples in pairwise(limits, 28):
                self.point_count += samples
                if samples > max_edge_samples:
                    max_edge_samples = samples

            if max_edge_samples >= (self.max_sample_points - 1) * 0.8:
                self.resize_sample_storage(int(max_edge_samples * 2))

    @track_time
    def get_buffer_points(self) -> int:
        return int(self.sample_buffer.size / 16.0)

    def delete(self):
        self.sample_buffer.delete()
        self.edge_buffer.delete()
        self.ssbo_handler.delete()
