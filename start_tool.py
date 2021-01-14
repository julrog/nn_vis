import threading
import time
from gui.ui_window import OptionGui
from opengl_helper.screenshot import create_screenshot
from processing.network_processing import NetworkProcessor
from utility.file import FileHandler
from utility.performance import track_time
from utility.window import WindowHandler, Window
from OpenGL.GL import *


global options
options = OptionGui()


def compute_render(name: str):
    global options

    width, height = 1920, 1200

    FileHandler().read_statistics()

    window_handler: WindowHandler = WindowHandler()
    window: Window = window_handler.create_window()
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
            network_processor.process(options.settings["action_state"])
            network_processor.render(window.cam, options.render_config, options.settings["show_class"])

        if "sample_count" in options.settings:
            options.settings["sample_count"].set(network_processor.edge_processor.point_count)
        if "edge_count" in options.settings:
            options.settings["edge_count"].set(network_processor.edge_processor.get_edge_count())
        if "cell_count" in options.settings:
            options.settings["cell_count"].set(network_processor.grid_processor.grid.grid_cell_count_overall)
        if "pruned_edges" in options.settings:
            options.settings["pruned_edges"].set(network_processor.network.pruned_edges)

        window.swap()

    while options is None or (len(options.settings["current_layer_data"]) is 0 and not options.settings["Closed"]):
        window_handler.update()
        time.sleep(5)

    if not options.settings["Closed"]:
        print("Start building network: " + str(options.settings["current_layer_data"]))
        network_processor = NetworkProcessor(options.settings["current_layer_data"],
                                             options.processing_config,
                                             importance_data=options.settings["importance_data"],
                                             processed_nn=options.settings["processed_nn"])
        window.cam.base = network_processor.get_node_mid()
        window.cam.set_position(1)

        while window.is_active() and not options.settings["Closed"]:
            if options.settings["update_model"]:
                options.settings["update_model"] = False
                network_processor.delete()
                print("Rebuilding network: " + str(options.settings["current_layer_data"]))
                network_processor = NetworkProcessor(options.settings["current_layer_data"],
                                                     options.processing_config,
                                                     importance_data=options.settings["importance_data"],
                                                     processed_nn=options.settings["processed_nn"])
                window.cam.base = network_processor.get_node_mid()
                window.cam.set_position(1)
            if start_count < 0:
                start_count = frame_count
                start_time = time.perf_counter()

            if window.screenshot_series > 0:
                window.cam.set_position(window.screenshot_series)
            frame()
            if window.screenshot_series > 0:
                if 'network_name' in options.settings.keys():
                    create_screenshot(width, height,
                                      options.settings['network_name'] + "_" + str(window.screenshot_series))
                else:
                    create_screenshot(width, height, "network_" + str(window.screenshot_series))
                window.screenshot_series -= 1
            else:
                if window.screenshot:
                    if 'network_name' in options.settings.keys():
                        create_screenshot(width, height, options.settings['network_name'])
                    else:
                        create_screenshot(width, height)
                    window.screenshot = False
                elif window.record:
                    window.frame_id += 1
                    if 'network_name' in options.settings.keys():
                        create_screenshot(width, height, options.settings['network_name'], frame_id=window.frame_id)
                    else:
                        create_screenshot(width, height, frame_id=window.frame_id)
            frame_count += 1
            if time.perf_counter() - start_time > 1.0:
                options.settings["fps"].set(
                    float("{:.2f}".format(float(frame_count - start_count) / (time.perf_counter() - start_time))))
                start_count = -1
            if "save_file" in options.settings.keys() and options.settings["save_file"]:
                network_processor.save_model(options.settings["save_processed_nn_path"])
                options.settings["save_file"] = False

        network_processor.delete()

    FileHandler().write_statistics()
    window_handler.destroy()
    options.destroy()


compute_render_thread: threading.Thread = threading.Thread(target=compute_render, args=(1,))
compute_render_thread.setDaemon(True)
compute_render_thread.start()

options.start()
