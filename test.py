import numpy as np
from pyrr import Vector3

from opengl_helper.render_utility import clear_screen
from processing.edge_processing import EdgeProcessor
from processing.grid_processing import GridProcessor
from rendering import EdgeRenderer, GridRenderer
from utility.file import FileHandler
from models import NetworkModel, Grid
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

network = NetworkModel([9, 4, 9], (Vector3([-1, -1, -11]), Vector3([1, 1, -2])))

sample_length = (network.bounding_range.z * 2.0) / 50.0
grid_cell_size = sample_length / 1.0
sample_radius = sample_length * 2.0

grid = Grid(Vector3([grid_cell_size, grid_cell_size, grid_cell_size]),
            (Vector3([-3, -3, -13]), Vector3([3, 3, 0])))

edge_handler = EdgeProcessor(sample_length)
edge_handler.set_data(network)
edge_handler.sample_edges()
edge_handler.check_limits(window.cam.view)
edge_renderer = EdgeRenderer(edge_handler, grid)

grid_processor = GridProcessor(grid, edge_handler, 20.0, sample_radius)
grid_processor.calculate_position()
grid_processor.calculate_density()
grid_processor.calculate_gradient()

grid_renderer = GridRenderer(grid_processor)

frame_count: int = 0


@track_time(track_recursive=False)
def frame():
    global frame_count
    window_handler.update()

    edge_handler.sample_noise(0.5)
    edge_handler.sample_edges()
    edge_handler.check_limits(window.cam.view)

    grid_processor.clear_buffer()
    grid_processor.calculate_density()
    grid_processor.calculate_gradient()

    clear_screen([1.0, 1.0, 1.0, 1.0])
    grid_renderer.render_cube(window, clear=False, swap=False)
    edge_renderer.render_transparent(window, clear=False, swap=True)

    if frame_count % 10 == 0:
        print("Rendering %d points from %d edges." % (edge_handler.get_buffer_points(), len(edge_handler.edges)))
    frame_count += 1


while window.is_active():
    frame()
FileHandler().write_statistics()
window_handler.destroy()
