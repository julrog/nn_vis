import math

from pyrr import Vector3

from edge import EdgeHandler, EdgeRenderer
from file import FileHandler
from performance import track_time
from window import WindowHandler
from OpenGL.GL import *

WIDTH, HEIGHT = 1920, 1080

window_handler = WindowHandler()
window = window_handler.create_window("Testing", WIDTH, HEIGHT, 1)
window.set_position(0, 0)
window.set_callbacks()
window.activate()

print("OpenGL Version: %d.%d" % (glGetIntegerv(GL_MAJOR_VERSION), glGetIntegerv(GL_MINOR_VERSION)))
nodes_layer_one = 1000
nodes_layer_one_sqrt = math.ceil(math.sqrt(nodes_layer_one))
nodes_layer_two = 100
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

sample_length = (Vector3([-1.0, 0.0, 0.0]) - Vector3([1.0, 0.0, 0.0])).length / 20.0
edge_handler = EdgeHandler(sample_length)
edge_handler.set_data(layer_one, layer_two)
edge_handler.sample_edges()

edge_renderer = EdgeRenderer(edge_handler)

frame_count: int = 0
edge_handler.check_limits(window.cam.get_view_matrix())


@track_time(track_recursive=False)
def frame():
    global frame_count
    window_handler.update()

    edge_handler.sample_noise(0.5)
    edge_handler.sample_edges()
    if frame_count % 10 == 0:
        print("Rendering %d points from %d edges." % (edge_handler.get_buffer_points(), len(edge_handler.edges)))

    edge_renderer.render_point(window, swap=True)
    frame_count += 1


FileHandler().read_statistics()
while window.is_active():
    frame()
FileHandler().write_statistics()
window_handler.destroy()
