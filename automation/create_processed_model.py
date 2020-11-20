from typing import List
from pyrr import Vector3
from data.data_handler import ImportanceDataHandler
from definitions import DATA_PATH
from opengl_helper.frame_buffer import FrameBufferObject
from opengl_helper.screenshot import create_screenshot
from processing.network_processing import NetworkProcessor
from OpenGL.GL import *

from processing.processing_config import ProcessingConfig
from rendering.rendering_config import RenderingConfig
from utility.camera import Camera
from utility.window import WindowHandler, Window


def generate_images(cam: Camera, screenshot_name: str, frame_buffer: FrameBufferObject, processor: NetworkProcessor,
                    cam_pos_list: List[int], show_class_list: List[int]):
    for show_class in show_class_list:
        for cam_pos in cam_pos_list:
            cam.set_position(cam_pos)
            cam.update()
            render(cam, processor, show_class)
            create_screenshot(frame_buffer.width, frame_buffer.height,
                              screenshot_name + "_class_" + str(show_class) + "cam_" + str(cam_pos),
                              frame_buffer=frame_buffer)


def process_loop(processor: NetworkProcessor):
    processor.process(1, True)
    while not processor.action_finished:
        processor.process(1, True)
    processor.reset_edges()
    processor.process(4, True)
    while not processor.action_finished:
        processor.process(4, True)
    processor.edge_processor.sample_edges()
    processor.edge_processor.check_limits()


def viewed_process_loop(processor: NetworkProcessor, screenshot_name: str, width: int, height: int,
                        frame_buffer: FrameBufferObject, show_nodes: bool = False, show_edge: bool = False,
                        show_smoothing: bool = False, rotate_cam: bool = False):
    show_classes: List[int] = [0, 1, 2]
    cam_poses: List[int] = [1, 5]
    processor.process(0, True)

    cam: Camera = Camera(width, height, Vector3([0.0, 0.0, 0.0]), rotation_speed=0.25 * 2.0)
    glViewport(0, 0, width, height)
    cam.set_size(width, height)
    cam.update_base(processor.get_node_mid())

    generate_images(cam, screenshot_name, frame_buffer, processor, cam_poses, show_classes)
    if show_nodes:
        generate_images(cam, screenshot_name, frame_buffer, processor, cam_poses, show_classes)

    while not processor.action_finished:
        processor.process(1, True)
        if show_nodes:
            cam.update_base(processor.get_node_mid())
            generate_images(cam, screenshot_name, frame_buffer, processor, cam_poses, show_classes)

    processor.reset_edges()
    generate_images(cam, screenshot_name, frame_buffer, processor, cam_poses, show_classes)

    cam.rotate_around_base = rotate_cam
    if show_smoothing and show_edge:
        processor.process(4, False)
        generate_images(cam, screenshot_name, frame_buffer, processor, cam_poses, show_classes)
        for i in range(7):
            processor.smooth_edges()
            generate_images(cam, screenshot_name, frame_buffer, processor, cam_poses, show_classes)

    else:
        processor.process(4, True)
        if show_edge:
            generate_images(cam, screenshot_name, frame_buffer, processor, cam_poses, show_classes)

    while not processor.action_finished:
        if show_smoothing and show_edge:
            processor.process(4, False)
            generate_images(cam, screenshot_name, frame_buffer, processor, cam_poses, show_classes)
            for i in range(7):
                processor.smooth_edges()
                generate_images(cam, screenshot_name, frame_buffer, processor, cam_poses, show_classes)
        else:
            processor.process(4, True)
            if show_edge:
                generate_images(cam, screenshot_name, frame_buffer, processor, cam_poses, show_classes)

    processor.edge_processor.sample_edges()
    processor.edge_processor.check_limits()
    generate_images(cam, screenshot_name, frame_buffer, processor, cam_poses, show_classes)


def render(cam: Camera, processor: NetworkProcessor, show_class: int = 0):
    config: RenderingConfig = RenderingConfig()
    processor.render(cam, config, show_class)


def process_network(network_name: str, importance_type: str, screenshot_mode: int = 0, show_smoothing: bool = False):
    width: int = 1600
    height: int = 900

    window_handler: WindowHandler = WindowHandler()
    window: Window = window_handler.create_window("Testing", 1, 1, 1)
    window.set_position(0, 0)
    window.set_callbacks()
    window.activate()

    frame_buffer: FrameBufferObject = FrameBufferObject(width, height) if screenshot_mode > 0 else None
    print("OpenGL Version: %d.%d" % (glGetIntegerv(GL_MAJOR_VERSION), glGetIntegerv(GL_MINOR_VERSION)))

    importance_data_path: str = DATA_PATH + "model/%s/%s_importance_data.npz" % (network_name, importance_type)

    importance_data: ImportanceDataHandler = ImportanceDataHandler(importance_data_path)

    config: ProcessingConfig = ProcessingConfig()
    network_processor: NetworkProcessor = NetworkProcessor(importance_data.layer_data,
                                                           config,
                                                           importance_data=importance_data,
                                                           processed_nn=None,
                                                           frame_buffer=frame_buffer)

    if screenshot_mode < 2:
        network_processor.reset_edges()
        process_loop(network_processor)
    else:
        frame_buffer.bind()
        network_processor.reset_edges()
        viewed_process_loop(network_processor, "processing_network",  width, height, frame_buffer, show_smoothing,
                            rotate_cam=True if screenshot_mode > 5 else False)

    network_processor.save_model(
        DATA_PATH + "model/%s/processed.npz" % network_name)

    if screenshot_mode > 0:
        if frame_buffer is not None:
            frame_buffer.bind()
            cam: Camera = Camera(width, height, Vector3([0.0, 0.0, 0.0]), rotation_speed=0.25 * 2.0)
            cam.base = network_processor.get_node_mid()
            glViewport(0, 0, width, height)
            cam.set_size(width, height)
            cam.set_position(0)
            render(cam, network_processor, 0)
            create_screenshot(width, height, "network" % importance_type, frame_buffer=frame_buffer)

    if frame_buffer is not None:
        frame_buffer.delete()
    network_processor.delete()
    window_handler.destroy()
