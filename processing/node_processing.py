from typing import List
from pyrr import Vector3
import numpy as np

from models.network import NetworkModel
from models.node import Node
from opengl_helper.buffer import SwappingBufferObject
from opengl_helper.compute_shader import ComputeShader, ComputeShaderHandler
from utility.performance import track_time
from opengl_helper.render_utility import VertexDataHandler

LOG_SOURCE: str = "NODE_PROCESSING"


class NodeProcessor:
    def __init__(self):
        self.noise_compute_shader: ComputeShader = ComputeShaderHandler().create("node_noise", "node/node_noise.comp")
        self.node_buffer: SwappingBufferObject = SwappingBufferObject(ssbo=True, object_size=16,
                                                                      render_data_offset=[0, 14],
                                                                      render_data_size=[4, 1])
        self.ssbo_handler: VertexDataHandler = VertexDataHandler([(self.node_buffer, 0)])

        self.nodes: List[Node] = []

        self.point_count: int = 0
        self.nearest_view_z: int = -1000000
        self.farthest_view_z: int = 1000000
        self.max_sample_points: int = 0

    def set_data(self, network: NetworkModel):
        self.nodes = network.get_nodes()

        # generate and load initial data for the buffer
        initial_data: List[float] = []
        for node in self.nodes:
            initial_data.extend(node.data)
        transfer_data = np.array(initial_data, dtype=np.float32)
        self.node_buffer.load(transfer_data)
        self.node_buffer.swap()
        self.node_buffer.load(transfer_data)

    @track_time
    def node_noise(self, sample_length: float, strength: float = 1.0):
        self.ssbo_handler.set()

        self.noise_compute_shader.set_uniform_data([
            ('noise_strength', strength, 'float'),
            ('sample_length', sample_length, 'float')
        ])
        self.noise_compute_shader.compute(len(self.nodes))

        self.node_buffer.swap()

    @track_time
    def read_nodes_from_buffer(self, raw: bool = False) -> List[float]:
        buffer_data = np.frombuffer(self.node_buffer.read(), dtype=np.float32)
        if raw:
            return buffer_data

        node_data = buffer_data.reshape((len(self.nodes), self.node_buffer.object_size))
        node_count = len(self.nodes)
        for i in range(node_count):
            self.nodes[i].reset_position(Vector3([node_data[i][0], node_data[i][1], node_data[i][2]]))

        return buffer_data

    @track_time
    def get_buffer_points(self) -> int:
        return len(self.nodes)

    def delete(self):
        self.node_buffer.delete()
