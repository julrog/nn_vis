import logging
import ntpath
import os
import threading
import time
import zipfile
from argparse import ArgumentParser
from typing import List, Optional, Tuple

import wget
from OpenGL.GL import GL_MAJOR_VERSION, GL_MINOR_VERSION, glGetIntegerv

from data.data_handler import ProcessedNNHandler
from definitions import DATA_PATH
from gui.constants import StatisticLink
from gui.ui_window import OptionGui
from processing.network_processing import NetworkProcessor
from utility.file import FileHandler
from utility.log_handling import setup_logger
from utility.performance import track_time
from vr.vr_handler import VRHandler

RENDER_MODES: List[Tuple[int, int]] = [(3, 2), (4, 1), (1, 1), (2, 2)]


def download_and_unzip_sample() -> str:
    output_directory = DATA_PATH
    filename = wget.download(
        'https://drive.google.com/uc?export=download&id=1EpsubJhHH4shqzDhsBB0SHsBjWgWa03S', out=output_directory)
    zip_filepath = os.path.join(output_directory, filename)
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(DATA_PATH)
    return os.path.join(DATA_PATH, 'sample_model.pro.npz')


def open_processed_network(option_gui: OptionGui, filename: str) -> None:
    data_loader: ProcessedNNHandler = ProcessedNNHandler(filename)
    option_gui.processing_config['prune_percentage'] = 0.9
    option_gui.processing_setting.set()
    option_gui.settings['network_name'] = ntpath.basename(
        filename) + '_processed'
    option_gui.update_layer(data_loader.layer_data, processed_nn=data_loader)


def compute_render(_: str) -> None:
    global options_gui
    global use_sample

    if use_sample:
        global sample_filepath
        logging.info('Loading sample model...')
        open_processed_network(options_gui, sample_filepath)

    FileHandler().read_statistics()

    vr_handler: VRHandler = VRHandler()

    logging.info(
        'OpenGL Version: %d.%d'
        % (glGetIntegerv(GL_MAJOR_VERSION), glGetIntegerv(GL_MINOR_VERSION))
    )

    network_processor: Optional[NetworkProcessor] = None

    @track_time(track_recursive=False)
    def frame() -> None:
        if network_processor is not None:
            if (
                'trigger_network_sample' in options_gui.settings
                and options_gui.settings['trigger_network_sample'] > 0
            ):
                network_processor.reset_edges()
                options_gui.settings['trigger_network_sample'] = 0
            network_processor.process(options_gui.settings['action_state'])

            if vr_handler.update():
                if vr_handler.input_handler.rotate_class:
                    options_gui.settings['show_class'] = (
                        options_gui.settings['show_class'] + 1
                    ) % (network_processor.layer_nodes[-1] + 2)
                if vr_handler.input_handler.rotate_render:
                    vr_handler.input_handler.current_render_mode = (
                        vr_handler.input_handler.current_render_mode + 1
                    ) % len(RENDER_MODES)
                    options_gui.render_config['edge_render_mode'] = RENDER_MODES[
                        vr_handler.input_handler.current_render_mode
                    ][0]
                    options_gui.render_config['node_render_mode'] = RENDER_MODES[
                        vr_handler.input_handler.current_render_mode
                    ][1]
                for cam, target in zip(vr_handler.context.cam, vr_handler.targets):
                    cam.generate_view()
                    target.frame_buffer.bind()
                    network_processor.render(
                        cam,
                        options_gui.render_config,
                        options_gui.settings['show_class'],
                    )
                    vr_handler.submit_target_texture(target)

            if StatisticLink.SAMPLE_COUNT in options_gui.settings:
                options_gui.settings[StatisticLink.SAMPLE_COUNT].set(
                    network_processor.edge_processor.point_count
                )
            if StatisticLink.EDGE_COUNT in options_gui.settings:
                options_gui.settings[StatisticLink.EDGE_COUNT].set(
                    network_processor.edge_processor.get_edge_count()
                )
            if StatisticLink.CELL_COUNT in options_gui.settings:
                options_gui.settings[StatisticLink.CELL_COUNT].set(
                    network_processor.grid_processor.grid.grid_cell_count_overall
                )
            if StatisticLink.PRUNED_EDGES in options_gui.settings:
                options_gui.settings[StatisticLink.PRUNED_EDGES].set(
                    network_processor.network.pruned_edges
                )

        vr_handler.context.swap()

    while options_gui is None or (
        len(options_gui.settings['current_layer_data']) == 0
        and not options_gui.settings['Closed']
    ):
        vr_handler.update()
        time.sleep(5)

    if not options_gui.settings['Closed']:
        print(
            'Start building network: ' +
            str(options_gui.settings['current_layer_data'])
        )
        options_gui.settings['update_model'] = False
        network_processor = NetworkProcessor(
            options_gui.settings['current_layer_data'],
            options_gui.processing_config,
            importance_data=options_gui.settings['importance_data'],
            processed_nn=options_gui.settings['processed_nn'],
        )
        vr_handler.context.cam[0].base = network_processor.get_node_mid()
        vr_handler.context.cam[1].base = network_processor.get_node_mid()

        fps: float = 120
        frame_count: int = 0
        to_pause_time: float = 0
        last_frame_count: int = 0
        checked_frame_count: int = -1
        check_time: float = time.perf_counter()
        last_time: float = time.perf_counter()

        while vr_handler.context.is_active() and not options_gui.settings['Closed']:
            if options_gui.settings['update_model']:
                options_gui.settings['update_model'] = False
                network_processor.delete()
                print(
                    'Rebuilding network: '
                    + str(options_gui.settings['current_layer_data'])
                )
                network_processor = NetworkProcessor(
                    options_gui.settings['current_layer_data'],
                    options_gui.processing_config,
                    importance_data=options_gui.settings['importance_data'],
                    processed_nn=options_gui.settings['processed_nn'],
                )
                vr_handler.context.cam[0].base = network_processor.get_node_mid(
                )
                vr_handler.context.cam[1].base = network_processor.get_node_mid(
                )

            frame()

            frame_count += 1
            if time.perf_counter() - check_time > 1.0:
                options_gui.settings[StatisticLink.FPS].set(
                    float(
                        '{:.2f}'.format(
                            float(frame_count - checked_frame_count)
                            / (time.perf_counter() - check_time)
                        )
                    )
                )
                checked_frame_count = frame_count
                check_time = time.perf_counter()
            if (
                'save_file' in options_gui.settings.keys()
                and options_gui.settings['save_file']
            ):
                network_processor.save_model(
                    options_gui.settings['save_processed_nn_path']
                )
                options_gui.settings['save_file'] = False

            current_time: float = time.perf_counter()
            elapsed_time: float = current_time - last_time
            if elapsed_time < 1.0 / fps:
                if elapsed_time > 0.001:
                    to_pause_time += (
                        float(frame_count - last_frame_count) / fps
                    ) - elapsed_time
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
                to_pause_time = (
                    0
                    if to_pause_time < 0
                    else to_pause_time - (elapsed_time - 1.0 / fps)
                )

        network_processor.delete()

    FileHandler().write_statistics()
    vr_handler.destroy()
    options_gui.destroy()


def parse_args() -> bool:
    parser = ArgumentParser(prog='Start nn_vis tool for VR')
    parser.add_argument('--demo', action='store_true',
                        help='Download sample of a processed model and render it with 90% pruned edges instead of generating a random model.')
    args = parser.parse_args()
    return args.demo


if __name__ == '__main__':
    global options_gui
    options_gui = OptionGui()

    global sample_filepath
    sample_filepath = 'sample_model.npz'

    global use_sample
    use_sample = parse_args()

    setup_logger('tool_vr')

    if use_sample:
        expected_sample_path = os.path.join(DATA_PATH, sample_filepath)
        if not os.path.exists(expected_sample_path):
            logging.info(
                f'Downloading sample model to "{expected_sample_path}". This might take a minute ...')
            sample_filepath = download_and_unzip_sample()
        else:
            logging.info(
                f'Using sample model at "{expected_sample_path}"')
            sample_filepath = expected_sample_path
    compute_render_thread: threading.Thread = threading.Thread(
        target=compute_render, args=(1,))
    compute_render_thread.setDaemon(True)
    compute_render_thread.start()

    options_gui.start()
