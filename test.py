import threading
import time

from gui.window import OptionGui
from processing.network_processing import NetworkProcessor
from utility.file import FileHandler
from utility.performance import track_time
from utility.window import WindowHandler, Window
from OpenGL.GL import *

global options
options = OptionGui()


def compute_render(name: str):
    global options

    width, height = 1920, 1170

    FileHandler().read_statistics()

    window_handler: WindowHandler = WindowHandler()
    window: Window = window_handler.create_window("Testing", width, height, 1)
    window.set_position(0, 0)
    window.set_callbacks()
    window.activate()

    print("OpenGL Version: %d.%d" % (glGetIntegerv(GL_MAJOR_VERSION), glGetIntegerv(GL_MINOR_VERSION)))

    network_processor: NetworkProcessor or None = None

    frame_count: int = 0
    start_count: int = -1
    start_time: float = time.perf_counter()

    @track_time(track_recursive=False)
    def frame():
        window_handler.update()

        if "trigger_network_sample" in options.settings and options.settings["trigger_network_sample"] > 0:
            network_processor.reset_edges()
            options.settings["trigger_network_sample"] = 0

        if network_processor is not None:
            network_processor.process(window, options.settings["action_state"], options.settings["edge_smoothing"])
            network_processor.render(window, options.settings["render_Edge"], options.settings["render_Grid"],
                                     options.settings["render_Node"], options.settings["render_shader_setting_Edge"],
                                     options.settings["render_shader_setting_Grid"],
                                     options.settings["render_shader_setting_Node"])

        if "sample_count" in options.settings:
            options.settings["sample_count"].set(network_processor.edge_processor.get_buffer_points())
        if "edge_count" in options.settings:
            options.settings["edge_count"].set(len(network_processor.edge_processor.edges))
        if "cell_count" in options.settings:
            options.settings["cell_count"].set(network_processor.grid_processor.grid.grid_cell_count_overall)
        if "pruned_edges" in options.settings:
            options.settings["pruned_edges"].set(network_processor.network.pruned_edges)

        window.swap()

    while options is None or (len(options.settings["current_layer_data"]) is 0 and not options.settings["Closed"]):
        window_handler.update()
        time.sleep(5)

    if not options.settings["Closed"]:
        print(options.settings["current_layer_data"])
        print("Start building network: " + str(options.settings["current_layer_data"]))
        network_processor = NetworkProcessor(options.settings["current_layer_data"])

        while window.is_active() and not options.settings["Closed"]:
            if network_processor.layer_nodes is not options.settings["current_layer_data"]:
                network_processor.delete()
                print("Rebuilding network: " + str(options.settings["current_layer_data"]))
                network_processor = NetworkProcessor(options.settings["current_layer_data"],
                                                     layer_distance=options.settings["layer_distance"],
                                                     node_size=options.settings["node_size"],
                                                     sampling_rate=options.settings["sampling_rate"],
                                                     importance_prune_threshold=options.settings[
                                                         "importance_threshold"])
            if start_count < 0:
                start_count = frame_count
                start_time = time.perf_counter()
            frame()
            frame_count += 1
            if time.perf_counter() - start_time > 1.0:
                options.settings["fps"].set(
                    float("{:.2f}".format(float(frame_count - start_count) / (time.perf_counter() - start_time))))
                start_count = -1

        network_processor.delete()

    FileHandler().write_statistics()
    window_handler.destroy()


compute_render_thread: threading.Thread = threading.Thread(target=compute_render, args=(1,))
compute_render_thread.start()

options.start()
