from data.data_handler import ImportanceDataHandler
from definitions import DATA_PATH
from processing.network_processing import NetworkProcessor
from OpenGL.GL import *

from utility.window import WindowHandler, Window


def process_loop(processor: NetworkProcessor):
    processor.process(1, True)
    while not processor.action_finished:
        processor.process(1, True)
    processor.reset_edges()
    processor.process(4, True)
    while not processor.action_finished:
        processor.process(4, True)


def process_network(network_name: str, importance_type: str, prune_rate: float = 0.9, edge_importance_type: int = 0):
    window_handler: WindowHandler = WindowHandler()
    window: Window = window_handler.create_window("Testing", 1, 1, 1)
    window.set_position(0, 0)
    window.set_callbacks()
    window.activate()

    print("OpenGL Version: %d.%d" % (glGetIntegerv(GL_MAJOR_VERSION), glGetIntegerv(GL_MINOR_VERSION)))

    importance_data_path: str = DATA_PATH + "model/%s/%s_importance_data.npz" % (network_name, importance_type)

    importance_data: ImportanceDataHandler = ImportanceDataHandler(importance_data_path)

    network_processor: NetworkProcessor = NetworkProcessor(importance_data.layer_data,
                                                           importance_data=importance_data,
                                                           processed_nn=None,
                                                           layer_distance=1.0,
                                                           layer_width=1.0,
                                                           sampling_rate=10.0,
                                                           prune_percentage=prune_rate,
                                                           node_bandwidth_reduction=0.98,
                                                           edge_bandwidth_reduction=0.9,
                                                           edge_importance_type=edge_importance_type)

    process_loop(network_processor)

    network_processor.save_model(
        DATA_PATH + "model/%s/%s_processed_eit%i.npz" % (network_name, importance_type, edge_importance_type))

    network_processor.delete()
    window_handler.destroy()
