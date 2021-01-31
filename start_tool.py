import logging
import threading
import time

from OpenGL.GL import *

from gui.constants import StatisticLink
from gui.ui_window import OptionGui
from opengl_helper.screenshot import create_screenshot
from processing.network_processing import NetworkProcessor
from utility.file import FileHandler
from utility.log_handling import setup_logger
from utility.performance import track_time
from utility.types import CameraPose
from utility.window import WindowHandler, Window

global options_gui
options_gui = OptionGui()
setup_logger("tool")


def compute_render(some_name: str):
    global options_gui

    width, height = 1920, 1200

    FileHandler().read_statistics()

    window_handler: WindowHandler = WindowHandler()
    window: Window = window_handler.create_window()
    window.set_callbacks()
    window.activate()

    logging.info("OpenGL Version: %d.%d" % (glGetIntegerv(GL_MAJOR_VERSION), glGetIntegerv(GL_MINOR_VERSION)))

    network_processor: NetworkProcessor or None = None

    @track_time(track_recursive=False)
    def frame():
        window_handler.update()

        if "trigger_network_sample" in options_gui.settings and options_gui.settings["trigger_network_sample"] > 0:
            network_processor.reset_edges()
            options_gui.settings["trigger_network_sample"] = 0

        if network_processor is not None:
            network_processor.process(options_gui.settings["action_state"])
            network_processor.render(window.cam, options_gui.render_config, options_gui.settings["show_class"])

        if StatisticLink.SAMPLE_COUNT in options_gui.settings:
            options_gui.settings[StatisticLink.SAMPLE_COUNT].set(network_processor.edge_processor.point_count)
        if StatisticLink.EDGE_COUNT in options_gui.settings:
            options_gui.settings[StatisticLink.EDGE_COUNT].set(network_processor.edge_processor.get_edge_count())
        if StatisticLink.CELL_COUNT in options_gui.settings:
            options_gui.settings[StatisticLink.CELL_COUNT].set(
                network_processor.grid_processor.grid.grid_cell_count_overall)
        if StatisticLink.PRUNED_EDGES in options_gui.settings:
            options_gui.settings[StatisticLink.PRUNED_EDGES].set(network_processor.network.pruned_edges)

        window.swap()

    while options_gui is None or (
            len(options_gui.settings["current_layer_data"]) is 0 and not options_gui.settings["Closed"]):
        window_handler.update()
        time.sleep(5)

    if not options_gui.settings["Closed"]:
        print("Start building network: " + str(options_gui.settings["current_layer_data"]))
        network_processor = NetworkProcessor(options_gui.settings["current_layer_data"],
                                             options_gui.processing_config,
                                             importance_data=options_gui.settings["importance_data"],
                                             processed_nn=options_gui.settings["processed_nn"])
        window.cam.base = network_processor.get_node_mid()
        window.cam.set_position(CameraPose.LEFT)

        fps: float = 60
        frame_count: int = 0
        to_pause_time: float = 0
        last_frame_count: int = 0
        checked_frame_count: int = -1
        check_time: float = time.perf_counter()
        last_time: float = time.perf_counter()

        while window.is_active() and not options_gui.settings["Closed"]:
            if options_gui.settings["update_model"]:
                options_gui.settings["update_model"] = False
                network_processor.delete()
                print("Rebuilding network: " + str(options_gui.settings["current_layer_data"]))
                network_processor = NetworkProcessor(options_gui.settings["current_layer_data"],
                                                     options_gui.processing_config,
                                                     importance_data=options_gui.settings["importance_data"],
                                                     processed_nn=options_gui.settings["processed_nn"])
                window.cam.base = network_processor.get_node_mid()
                window.cam.set_position(CameraPose.LEFT)

            frame()
            if window.screenshot:
                if "network_name" in options_gui.settings.keys():
                    create_screenshot(width, height, options_gui.settings["network_name"])
                else:
                    create_screenshot(width, height)
                window.screenshot = False
            elif window.record:
                window.frame_id += 1
                if "network_name" in options_gui.settings.keys():
                    create_screenshot(width, height, options_gui.settings["network_name"], frame_id=window.frame_id)
                else:
                    create_screenshot(width, height, frame_id=window.frame_id)

            frame_count += 1
            if time.perf_counter() - check_time > 1.0:
                options_gui.settings[StatisticLink.FPS].set(float(
                    "{:.2f}".format(float(frame_count - checked_frame_count) / (time.perf_counter() - check_time))))
                checked_frame_count = frame_count
                check_time = time.perf_counter()
            if "save_file" in options_gui.settings.keys() and options_gui.settings["save_file"]:
                network_processor.save_model(options_gui.settings["save_processed_nn_path"])
                options_gui.settings["save_file"] = False

            current_time: float = time.perf_counter()
            elapsed_time: float = current_time - last_time
            if elapsed_time < 1.0 / fps:
                if elapsed_time > 0.001:
                    to_pause_time += (float(frame_count - last_frame_count) / fps) - elapsed_time
                    last_frame_count = frame_count
                    last_time = current_time

                if to_pause_time > 0.005:
                    time.sleep(to_pause_time)
                    paused_for: float = time.perf_counter() - current_time
                    to_pause_time -= paused_for
                    last_time += paused_for
            else:
                last_frame_count = frame_count
                last_time = current_time
                to_pause_time = 0 if to_pause_time < 0 else to_pause_time - (elapsed_time - 1.0 / fps)

        network_processor.delete()

    FileHandler().write_statistics()
    window_handler.destroy()
    options_gui.destroy()


compute_render_thread: threading.Thread = threading.Thread(target=compute_render, args=(1,))
compute_render_thread.setDaemon(True)
compute_render_thread.start()

options_gui.start()
