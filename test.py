import numpy as np
from pyrr import Vector3

from models.grid import Grid
from models.network import NetworkModel
from opengl_helper.render_utility import clear_screen
from processing.edge_processing import EdgeProcessor
from processing.grid_processing import GridProcessor
from rendering.edge_rendering import EdgeRenderer

from rendering.grid_rendering import GridRenderer
from utility.file import FileHandler
from utility.performance import track_time
from utility.window import WindowHandler
from OpenGL.GL import *

WIDTH, HEIGHT = 1920, 1080

FileHandler().read_statistics()

window_handler = WindowHandler()
window = window_handler.create_window("Testing", WIDTH, HEIGHT, 1)
window.set_position(0, 0)
window.set_callbacks()
window.activate()

print("OpenGL Version: %d.%d" % (glGetIntegerv(GL_MAJOR_VERSION), glGetIntegerv(GL_MINOR_VERSION)))

network = NetworkModel([25, 9, 25], (Vector3([-1, -1, -7]), Vector3([1, 1, -3])))

sample_length = (network.bounding_range.z * 2.0) / 50.0
grid_cell_size = sample_length / 3.0
sample_radius = sample_length * 2.0

grid = Grid(Vector3([grid_cell_size, grid_cell_size, grid_cell_size]),
            (Vector3([-2, -2, -8]), Vector3([2, 2, -2])))

edge_handler = EdgeProcessor(sample_length)
edge_handler.set_data(network)
edge_handler.sample_edges()
edge_handler.check_limits(window.cam.view)
edge_renderer = EdgeRenderer(edge_handler, grid)

grid_processor = GridProcessor(grid, edge_handler, 10.0, sample_radius, 0.05)
grid_processor.calculate_position()
grid_processor.calculate_density()
grid_processor.calculate_gradient()

grid_renderer = GridRenderer(grid_processor)

frame_count: int = 0


@track_time(track_recursive=False)
def frame():
    global frame_count
    window_handler.update()

    edge_handler.check_limits(window.cam.view)
    if not window.freeze:
        if window.gradient:
            grid_processor.clear_buffer()
            grid_processor.calculate_density()
            grid_processor.calculate_gradient()
            grid_processor.sample_advect()

        # edge_handler.sample_noise(0.5)
        edge_handler.sample_edges()

    clear_screen([1.0, 1.0, 1.0, 1.0])
    if window.gradient:
        grid_renderer.render_cube(window, clear=False, swap=False)
    edge_renderer.render_transparent(window, clear=False, swap=False)
    window.swap()

    if frame_count % 10 == 0:
        print("Rendering %d points from %d edges." % (edge_handler.get_buffer_points(), len(edge_handler.edges)))
    frame_count += 1


while window.is_active():
    frame()
FileHandler().write_statistics()
window_handler.destroy()
