from typing import List
from pyrr import Vector3

from automation.automation_config import AutomationConfig
from data.data_handler import ImportanceDataHandler
from definitions import DATA_PATH
from opengl_helper.frame_buffer import FrameBufferObject
from opengl_helper.screenshot import create_screenshot
from processing.network_processing import NetworkProcessor, NetworkProcess
from OpenGL.GL import *

from processing.processing_config import ProcessingConfig
from rendering.rendering_config import RenderingConfig
from utility.camera import Camera
from utility.types import ProcessRenderMode
from utility.window import WindowHandler, Window


def generate_images(cam: Camera, screenshot_name: str, frame_buffer: FrameBufferObject, processor: NetworkProcessor,
                    automation_config: AutomationConfig):
    for show_class in automation_config["class_list"]:
        for cam_pos in automation_config["camera_pose_list"]:
            cam.set_position(cam_pos)
            cam.update()
            render(cam, processor, show_class)
            create_screenshot(frame_buffer.width, frame_buffer.height,
                              screenshot_name + "_class_" + str(show_class) + "_cam_" + str(cam_pos),
                              frame_buffer=frame_buffer)


def process_loop(processor: NetworkProcessor):
    processor.process(NetworkProcess.NODE_ADVECT)
    while not processor.action_finished:
        processor.process(NetworkProcess.NODE_ADVECT)
    processor.reset_edges()
    processor.process(NetworkProcess.EDGE_ADVECT)
    while not processor.action_finished:
        processor.process(NetworkProcess.EDGE_ADVECT)
    processor.edge_processor.sample_edges()
    processor.edge_processor.check_limits()


def viewed_process_loop(processor: NetworkProcessor, screenshot_name: str, frame_buffer: FrameBufferObject,
                        automation_config: AutomationConfig):
    processor.process(NetworkProcess.NODE_ADVECT)

    cam: Camera = Camera(automation_config["screenshot_width"],
                         automation_config["screenshot_height"],
                         Vector3([0.0, 0.0, 0.0]),
                         rotation_speed=automation_config["camera_rotation_speed"])
    glViewport(0, 0, automation_config["screenshot_width"], automation_config["screenshot_height"])
    cam.set_size(automation_config["screenshot_width"], automation_config["screenshot_height"])
    cam.update_base(processor.get_node_mid())

    generate_images(cam, screenshot_name, frame_buffer, processor, automation_config)
    if automation_config["screenshot_mode"] & ProcessRenderMode.NODE_ITERATIONS:
        generate_images(cam, screenshot_name, frame_buffer, processor, automation_config)

    while not processor.action_finished:
        processor.process(NetworkProcess.NODE_ADVECT)
        if automation_config["screenshot_mode"] & ProcessRenderMode.NODE_ITERATIONS:
            cam.update_base(processor.get_node_mid())
            generate_images(cam, screenshot_name, frame_buffer, processor, automation_config)

    processor.reset_edges()
    generate_images(cam, screenshot_name, frame_buffer, processor, automation_config)

    cam.rotate_around_base = automation_config["camera_rotation"]
    if automation_config["screenshot_mode"] & (ProcessRenderMode.SMOOTHING | ProcessRenderMode.EDGE_ITERATIONS):
        edge_smoothing: bool = processor.edge_smoothing
        if edge_smoothing:
            processor.edge_smoothing = False
            processor.process(NetworkProcess.EDGE_ADVECT)
            processor.edge_smoothing = edge_smoothing

            generate_images(cam, screenshot_name, frame_buffer, processor, automation_config)
            for i in range(processor.edge_smoothing_iterations):
                processor.smooth_edges()
                generate_images(cam, screenshot_name, frame_buffer, processor, automation_config)

    else:
        processor.process(NetworkProcess.EDGE_ADVECT)
        if automation_config["screenshot_mode"] & ProcessRenderMode.EDGE_ITERATIONS:
            generate_images(cam, screenshot_name, frame_buffer, processor, automation_config)

    while not processor.action_finished:
        if automation_config["screenshot_mode"] & (ProcessRenderMode.SMOOTHING | ProcessRenderMode.EDGE_ITERATIONS):
            edge_smoothing: bool = processor.edge_smoothing
            if edge_smoothing:
                processor.edge_smoothing = False
                processor.process(NetworkProcess.EDGE_ADVECT)
                processor.edge_smoothing = edge_smoothing

                generate_images(cam, screenshot_name, frame_buffer, processor, automation_config)
                for i in range(processor.edge_smoothing_iterations):
                    processor.smooth_edges()
                    generate_images(cam, screenshot_name, frame_buffer, processor, automation_config)

        else:
            processor.process(NetworkProcess.EDGE_ADVECT)
            if automation_config["screenshot_mode"] & ProcessRenderMode.EDGE_ITERATIONS:
                generate_images(cam, screenshot_name, frame_buffer, processor, automation_config)

    processor.edge_processor.sample_edges()
    processor.edge_processor.check_limits()
    generate_images(cam, screenshot_name, frame_buffer, processor, automation_config)


def render(cam: Camera, processor: NetworkProcessor, show_class: int = 0):
    config: RenderingConfig = RenderingConfig()
    processor.render(cam, config, show_class)


def process_network(network_name: str, importance_type: str, automation_config: AutomationConfig):
    window_handler: WindowHandler = WindowHandler()
    window: Window = window_handler.create_window("Testing", 1, 1, 1)
    window.set_position(0, 0)
    window.set_callbacks()
    window.activate()

    frame_buffer: FrameBufferObject or None = None
    if automation_config["screenshot_mode"]:
        frame_buffer = FrameBufferObject(automation_config["screenshot_width"], automation_config["screenshot_height"])

    print("OpenGL Version: %d.%d" % (glGetIntegerv(GL_MAJOR_VERSION), glGetIntegerv(GL_MINOR_VERSION)))

    importance_data_path: str = DATA_PATH + "model/%s/%s_importance_data.npz" % (network_name, importance_type)

    importance_data: ImportanceDataHandler = ImportanceDataHandler(importance_data_path)

    config: ProcessingConfig = ProcessingConfig()
    network_processor: NetworkProcessor = NetworkProcessor(importance_data.layer_data,
                                                           config,
                                                           importance_data=importance_data,
                                                           processed_nn=None,
                                                           frame_buffer=frame_buffer)

    network_processor.reset_edges()
    if not automation_config["screenshot_mode"] or automation_config["screenshot_mode"] is ProcessRenderMode.FINAL:
        process_loop(network_processor)
    else:
        frame_buffer.bind()
        viewed_process_loop(network_processor,
                            "processing_network",
                            frame_buffer,
                            automation_config)

    network_processor.save_model(
        DATA_PATH + "model/%s/processed.npz" % network_name)

    if automation_config["screenshot_mode"]:
        if frame_buffer is not None:
            frame_buffer.bind()
            cam: Camera = Camera(automation_config["screenshot_width"], automation_config["screenshot_height"],
                                 Vector3([0.0, 0.0, 0.0]), rotation_speed=0.25 * 2.0)
            cam.base = network_processor.get_node_mid()
            glViewport(0, 0, automation_config["screenshot_width"], automation_config["screenshot_height"])
            cam.set_size(automation_config["screenshot_width"], automation_config["screenshot_height"])
            cam.set_position(automation_config["camera_pose_final"])
            render(cam, network_processor, 0)
            create_screenshot(automation_config["screenshot_width"], automation_config["screenshot_height"],
                              "network_%s" % importance_type, frame_buffer=frame_buffer)

    if frame_buffer is not None:
        frame_buffer.delete()
    network_processor.delete()
    window_handler.destroy()
