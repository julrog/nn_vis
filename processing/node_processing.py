from typing import List

import numpy as np
from pyrr import Vector3

from definitions import ADDITIONAL_NODE_BUFFER_DATA
from models.network import NetworkModel
from models.node import Node
from opengl_helper.buffer import SwappingBufferObject, get_buffer_settings
from opengl_helper.compute_shader import ComputeShader
from opengl_helper.compute_shader_handler import ComputeShaderHandler
from opengl_helper.vertex_data_handler import VertexDataHandler
from utility.performance import track_time


class NodeProcessor:
    def __init__(self, network: NetworkModel):
        ComputeShaderHandler().create('node_noise', 'node/node_noise.comp')

        object_size, render_data_offset, render_data_size = \
            get_buffer_settings(network.num_classes,
                                ADDITIONAL_NODE_BUFFER_DATA)
        self.node_buffer: SwappingBufferObject = SwappingBufferObject(ssbo=True, object_size=object_size,
                                                                      render_data_offset=render_data_offset,
                                                                      render_data_size=render_data_size)
        self.ssbo_handler: VertexDataHandler = VertexDataHandler(
            [(self.node_buffer, 0)])

        self.nodes: List[Node] = network.get_nodes()

        self.point_count: int = 0
        self.nearest_view_z: int = -1000000
        self.farthest_view_z: int = 1000000
        self.max_sample_points: int = 0

        self.node_min_importance: float = network.node_min_importance
        self.node_max_importance: float = network.node_max_importance

        self.set_data()

    def set_data(self):
        initial_data: List[float] = []
        for node in self.nodes:
            initial_data.extend(node.data)
        transfer_data = np.array(initial_data, dtype=np.float32)
        self.node_buffer.load(transfer_data)
        self.node_buffer.swap()
        self.node_buffer.load(transfer_data)

    @track_time
    def node_noise(self, sample_length: float, strength: float = 1.0):
        noise: ComputeShader = ComputeShaderHandler().get('node_noise')
        noise.set_uniform_data(
            [('noise_strength', strength, 'float'), ('sample_length', sample_length, 'float')])
        self.ssbo_handler.set()
        noise.compute(len(self.nodes), barrier=True)
        self.node_buffer.swap()

    @track_time
    def read_nodes_from_buffer(self, raw: bool = False) -> List[float]:
        buffer_data = np.frombuffer(self.node_buffer.read(), dtype=np.float32)
        if raw:
            return buffer_data

        node_data = buffer_data.reshape(
            (len(self.nodes), self.node_buffer.object_size))
        node_count = len(self.nodes)
        for i in range(node_count):
            self.nodes[i].reset_position(
                Vector3([node_data[i][0], node_data[i][1], node_data[i][2]]))

        return buffer_data

    @track_time
    def get_buffer_points(self) -> int:
        return len(self.nodes)

    def delete(self):
        self.node_buffer.delete()
