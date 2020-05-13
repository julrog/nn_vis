import threading
import time

from gui.window import OptionGui
from processing.network_processing import NetworkProcessor
from utility.file import FileHandler
from utility.performance import track_time
from utility.window import WindowHandler
from OpenGL.GL import *

global options
options = OptionGui()


def compute_render(name: str):
    global options

    width, height = 1920, 1170

    FileHandler().read_statistics()

    window_handler = WindowHandler()
    window = window_handler.create_window("Testing", width, height, 1)
    window.set_position(0, 0)
    window.set_callbacks()
    window.activate()

    print("OpenGL Version: %d.%d" % (glGetIntegerv(GL_MAJOR_VERSION), glGetIntegerv(GL_MINOR_VERSION)))

    network_processor = None

    frame_count: int = 0

    @track_time(track_recursive=False)
    def frame():
        window_handler.update()

        if "trigger_network_sample" in options.settings and options.settings["trigger_network_sample"] > 0:
            network_processor.reset_edges()
            options.settings["trigger_network_sample"] = 0

        if network_processor is not None:
            network_processor.process(window, options.settings["action_state"])
            network_processor.render(window, options.settings["render_edge"], options.settings["render_grid"])

        if "sample_count" in options.settings:
            options.settings["sample_count"].set(network_processor.edge_processor.get_buffer_points())
        if "edge_count" in options.settings:
            options.settings["edge_count"].set(len(network_processor.edge_processor.edges))
        if "cell_count" in options.settings:
            options.settings["cell_count"].set(network_processor.grid_processor.grid.grid_cell_count_overall)

        window.swap()

    while options is None or (len(options.settings["current_layer_data"]) is 0 and not options.settings["Closed"]):
        window_handler.update()
        time.sleep(5)

    if not options.settings["Closed"]:
        print(options.settings["current_layer_data"])
        print("Start building network: " + str(options.settings["current_layer_data"]))
        network_processor = NetworkProcessor(options.settings["current_layer_data"])

        while window.is_active() and not options.settings["Closed"]:
            if network_processor.layer_data is not options.settings["current_layer_data"]:
                network_processor.delete()
                print("Rebuilding network: " + str(options.settings["current_layer_data"]))
                network_processor = NetworkProcessor(options.settings["current_layer_data"],
                                                     layer_distance=options.settings["layer_distance"],
                                                     node_size=options.settings["node_size"],
                                                     sampling_rate=options.settings["sampling_rate"])
            frame()
            frame_count += 1

        network_processor.delete()

    FileHandler().write_statistics()
    window_handler.destroy()


compute_render_thread: threading.Thread = threading.Thread(target=compute_render, args=(1,))
compute_render_thread.start()

options.start()
