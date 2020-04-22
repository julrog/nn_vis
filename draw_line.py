import math

from pyrr import Vector3

from network_handler import EdgeHandler, EdgeRenderer
from file import FileHandler
from network_model import NetworkModel
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

network = NetworkModel([9, 9, 9], (Vector3([-1, -1, -11]), Vector3([1, 1, -2])))
sample_length = (network.bounding_range.z * 2.0) / 100.0
edge_handler = EdgeHandler(sample_length)
edge_handler.set_data(network)
edge_handler.sample_edges()

edge_renderer = EdgeRenderer(edge_handler)

frame_count: int = 0
edge_handler.check_limits(window.cam.get_view_matrix())


@track_time(track_recursive=False)
def frame():
    global frame_count
    window_handler.update()

    #edge_handler.sample_noise(0.66)
    #edge_handler.sample_edges()
    edge_handler.check_limits(window.cam.get_view_matrix())

    if frame_count % 10 == 0:
        print("Rendering %d points from %d edges." % (edge_handler.get_buffer_points(), len(edge_handler.edges)))

    edge_renderer.render_transparent(window, swap=True)
    frame_count += 1


FileHandler().read_statistics()
while window.is_active():
    frame()
FileHandler().write_statistics()
window_handler.destroy()
