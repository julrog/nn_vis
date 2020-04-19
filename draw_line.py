import math

from pyrr import Vector3

from edge import EdgeHandler, EdgeRenderer
from file import FileHandler
from performance import track_time
from window import WindowHandler
from OpenGL.GL import *

WIDTH, HEIGHT = 1920, 1080

window_handler = WindowHandler()
window = window_handler.create_window("Testing", WIDTH, HEIGHT, 0)
window.set_position(0, 0)
window.set_callbacks()
window.activate()

print(glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE))
nodes_layer_one = 18
nodes_layer_one_sqrt = math.ceil(math.sqrt(nodes_layer_one))
nodes_layer_two = 18
nodes_layer_two_sqrt = math.ceil(math.sqrt(nodes_layer_two))

layer_one = []
for i in range(nodes_layer_one):
    layer_one.append(((i % nodes_layer_one_sqrt) / nodes_layer_one_sqrt) * 2.0 - 1.0)
    layer_one.append(((math.floor(i / nodes_layer_one_sqrt)) / nodes_layer_one_sqrt) * 2.0 - 1.0)
    layer_one.append(-1.0)

layer_two = []
for i in range(nodes_layer_two):
    layer_two.append(((i % nodes_layer_two_sqrt) / nodes_layer_two_sqrt) * 2.0 - 1.0)
    layer_two.append(((math.floor(i / nodes_layer_two_sqrt)) / nodes_layer_two_sqrt) * 2.0 - 1.0)
    layer_two.append(1.0)

sample_length = (Vector3([-0.5, -0.3, 0.0]) - Vector3([0.5, -0.3, 0.0])).length / 50.0
edge_handler = EdgeHandler(sample_length, True)
edge_handler.set_data(layer_one, layer_two)
edge_handler.sample_edges()

edge_renderer = EdgeRenderer(edge_handler)


@track_time(track_recursive=False)
def frame():
    window_handler.update()

    edge_handler.sample_noise(0.66)
    edge_handler.sample_edges()

    edge_renderer.render_transparent(window)

    window.swap()


FileHandler().read_statistics()
while window.is_active():
    frame()
FileHandler().write_statistics()
window_handler.destroy()
