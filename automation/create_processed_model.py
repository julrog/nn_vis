from typing import Dict, List

from pyrr import Vector3

from data.data_handler import ImportanceDataHandler
from definitions import DATA_PATH
from opengl_helper.frame_buffer import FrameBufferObject
from opengl_helper.screenshot import create_screenshot
from processing.network_processing import NetworkProcessor
from OpenGL.GL import *

from utility.camera import Camera
from utility.window import WindowHandler, Window


def generate_images(cam: Camera, screenshot_name: str, frame_buffer: FrameBufferObject, processor: NetworkProcessor,
                    cam_pos_list: List[int], show_class_list: List[int], show_edge: bool, node_phong: bool = False):
    for show_class in show_class_list:
        for cam_pos in cam_pos_list:
            cam.set_position(cam_pos)
            cam.update()
            render(cam, processor, show_class, show_edge=show_edge, node_phong=node_phong)
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

    generate_images(cam, screenshot_name, frame_buffer, processor, cam_poses, show_classes, True, False)
    if show_nodes:
        generate_images(cam, screenshot_name, frame_buffer, processor, cam_poses, show_classes, False, True)

    while not processor.action_finished:
        processor.process(1, True)
        if show_nodes:
            cam.update_base(processor.get_node_mid())
            generate_images(cam, screenshot_name, frame_buffer, processor, cam_poses, show_classes, False, True)

    processor.reset_edges()
    generate_images(cam, screenshot_name, frame_buffer, processor, cam_poses, show_classes, True, False)

    cam.rotate_around_base = rotate_cam
    if show_smoothing and show_edge:
        processor.process(4, False)
        generate_images(cam, screenshot_name, frame_buffer, processor, cam_poses, show_classes, True, False)
        for i in range(7):
            processor.smooth_edges()
            generate_images(cam, screenshot_name, frame_buffer, processor, cam_poses, show_classes, True, False)

    else:
        processor.process(4, True)
        if show_edge:
            generate_images(cam, screenshot_name, frame_buffer, processor, cam_poses, show_classes, True, False)

    while not processor.action_finished:
        if show_smoothing and show_edge:
            processor.process(4, False)
            generate_images(cam, screenshot_name, frame_buffer, processor, cam_poses, show_classes, True, False)
            for i in range(7):
                processor.smooth_edges()
                generate_images(cam, screenshot_name, frame_buffer, processor, cam_poses, show_classes, True, False)
        else:
            processor.process(4, True)
            if show_edge:
                generate_images(cam, screenshot_name, frame_buffer, processor, cam_poses, show_classes, True, False)

    generate_images(cam, screenshot_name, frame_buffer, processor, cam_poses, show_classes, True, False)


def render(cam: Camera, processor: NetworkProcessor, show_class: int = 0, show_edge: bool = True,
           node_phong: bool = False):
    edge_options: Dict[str, any] = {"Size": 0.2, "Opacity": 0.0, "Importance Opacity": 1.1, "Depth Opacity": 0.0,
                                    "Density Exponent": 0.5, "Importance Threshold": 0.01}
    grid_options: Dict[str, any] = {}
    node_options: Dict[str, any] = {"Size": 0.05, "Opacity": 0.2, "Importance Opacity": 1.0, "Depth Opacity": 0.0,
                                    "Density Exponent": 0.5, "Importance Threshold": 0.01}

    if show_class > 1:
        node_options: Dict[str, any] = {"Size": 0.05, "Opacity": 0.0, "Importance Opacity": 1.1, "Depth Opacity": 0.0,
                                        "Density Exponent": 0.5, "Importance Threshold": 0.01}

    edge_render_mode: int = 3 if show_edge else 0
    node_render_mode: int = 1 if node_phong else 2
    processor.render(cam, edge_render_mode, 0, node_render_mode, edge_options, grid_options, node_options, show_class)


def process_network(network_name: str, importance_type: str, prune_rate: float = 0.9, edge_importance_type: int = 0,
                    screenshot_mode: int = 0, show_nodes: bool = False, show_edge: bool = False,
                    show_smoothing: bool = False):
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

    network_processor: NetworkProcessor = NetworkProcessor(importance_data.layer_data,
                                                           importance_data=importance_data,
                                                           processed_nn=None,
                                                           layer_distance=1.0,
                                                           layer_width=1.0,
                                                           sampling_rate=10.0,
                                                           prune_percentage=prune_rate,
                                                           node_bandwidth_reduction=0.98,
                                                           edge_bandwidth_reduction=0.9,
                                                           edge_importance_type=edge_importance_type,
                                                           frame_buffer=frame_buffer)

    if screenshot_mode < 2:
        network_processor.reset_edges()
        process_loop(network_processor)
    else:
        frame_buffer.bind()
        network_processor.reset_edges()
        viewed_process_loop(network_processor, "processing_network_%s_eti%i" % (importance_type, edge_importance_type),
                            width, height, frame_buffer, show_nodes, show_edge, show_smoothing,
                            rotate_cam=True if screenshot_mode > 5 else False)

    network_processor.save_model(
        DATA_PATH + "model/%s/%s_processed_eit%i.npz" % (network_name, importance_type, edge_importance_type))

    if screenshot_mode > 0:
        if frame_buffer is not None:
            frame_buffer.bind()
            cam: Camera = Camera(width, height, Vector3([0.0, 0.0, 0.0]), rotation_speed=0.25 * 2.0)
            cam.base = network_processor.get_node_mid()
            glViewport(0, 0, width, height)
            cam.set_size(width, height)
            cam.set_position(0)
            render(cam, network_processor, 0)
            create_screenshot(width, height, "network_%s_eti%i" % (importance_type, edge_importance_type),
                              frame_buffer=frame_buffer)

    if frame_buffer is not None:
        frame_buffer.delete()
    network_processor.delete()
    window_handler.destroy()
