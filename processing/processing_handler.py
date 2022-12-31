import os.path
from typing import Optional

from OpenGL.GL import glViewport
from pyrr import Vector3

from data.data_handler import ImportanceDataHandler
from definitions import DATA_PATH, ProcessRenderMode
from opengl_helper.frame_buffer import FrameBufferObject
from opengl_helper.screenshot import create_screenshot
from processing.network_processing import NetworkProcess, NetworkProcessor
from processing.processing_config import ProcessingConfig
from rendering.rendering_config import RenderingConfig
from utility.camera import Camera
from utility.recording_config import RecordingConfig
from utility.window import Window, WindowHandler


class ProcessingHandler:
    def __init__(self, network_name: str, importance_data_name: str) -> None:
        self.network_name: str = network_name
        self.importance_data_name: str = importance_data_name
        window: Window = WindowHandler().create_window(hidden=True)
        window.set_callbacks()
        window.activate()

        importance_data_path: str = DATA_PATH + 'model/%s/%s.imp.npz' % (self.network_name,
                                                                         self.importance_data_name)

        if not os.path.exists(importance_data_path):
            raise Exception("Importance data '%s' for model '%s' is not yet created." % (self.network_name,
                                                                                         self.importance_data_name))

        importance_data: ImportanceDataHandler = ImportanceDataHandler(
            importance_data_path)

        config: ProcessingConfig = ProcessingConfig()
        self.processor: NetworkProcessor = NetworkProcessor(importance_data.layer_data,
                                                            config,
                                                            importance_data=importance_data,
                                                            processed_nn=None)

    def process_loop(self) -> None:
        self.processor.process(NetworkProcess.NODE_ADVECT)
        while not self.processor.action_finished:
            self.processor.process(NetworkProcess.NODE_ADVECT)
        self.processor.reset_edges()
        self.processor.process(NetworkProcess.EDGE_ADVECT)
        while not self.processor.action_finished:
            self.processor.process(NetworkProcess.EDGE_ADVECT)
        self.processor.edge_processor.sample_edges()
        self.processor.edge_processor.check_limits()

    def process(self) -> None:
        self.processor.reset_edges()
        self.process_loop()

        self.processor.save_model(DATA_PATH + 'model/%s/%s.pro.npz' % (self.network_name,
                                                                       self.importance_data_name))
        self.clean_up()

    def clean_up(self) -> None:
        self.processor.delete()
        WindowHandler().destroy()


class RecordingProcessingHandler(ProcessingHandler):
    def __init__(self, network_name: str, importance_data_name: str, recording_config: RecordingConfig) -> None:
        super().__init__(network_name, importance_data_name)
        self.screenshot_name: str = 'processed_network'
        self.recording_config = recording_config
        self.frame_buffer: Optional[FrameBufferObject] = None
        if self.recording_config['screenshot_mode']:
            self.frame_buffer = FrameBufferObject(self.recording_config['screenshot_width'],
                                                  self.recording_config['screenshot_height'])
        self.cam: Camera = self.setup_cam()

    def setup_cam(self) -> Camera:
        cam: Camera = Camera(self.recording_config['screenshot_width'],
                             self.recording_config['screenshot_height'],
                             Vector3([0.0, 0.0, 0.0]),
                             rotation_speed=self.recording_config['camera_rotation_speed'])
        glViewport(0, 0, self.recording_config['screenshot_width'],
                   self.recording_config['screenshot_height'])
        cam.set_size(self.recording_config['screenshot_width'],
                     self.recording_config['screenshot_height'])
        cam.update_base(self.processor.get_node_mid())
        return cam

    def generate_images(self) -> None:
        if self.frame_buffer is None:
            raise Exception('No Framebuffer set for generating images')
        for show_class in self.recording_config['class_list']:
            for cam_pos in self.recording_config['camera_pose_list']:
                self.cam.set_position(cam_pos)
                self.cam.update()
                self.render(show_class)
                create_screenshot(self.frame_buffer.width, self.frame_buffer.height,
                                  self.screenshot_name + '_class_' +
                                  str(show_class) + '_cam_' + str(cam_pos),
                                  frame_buffer=self.frame_buffer)

    def viewed_node_process(self) -> None:
        self.processor.process(NetworkProcess.NODE_ADVECT)
        if self.recording_config['screenshot_mode'] & ProcessRenderMode.NODE_ITERATIONS:
            self.cam.update_base(self.processor.get_node_mid())
            self.generate_images()

    def viewed_edge_process(self) -> None:
        if self.recording_config['screenshot_mode'] & (ProcessRenderMode.SMOOTHING | ProcessRenderMode.EDGE_ITERATIONS):
            edge_smoothing: bool = self.processor.edge_smoothing
            if edge_smoothing:
                self.processor.edge_smoothing = False
                self.processor.process(NetworkProcess.EDGE_ADVECT)
                self.processor.edge_smoothing = edge_smoothing

                self.generate_images()
                for i in range(self.processor.edge_smoothing_iterations):
                    self.processor.smooth_edges()
                    self.generate_images()
        else:
            self.processor.process(NetworkProcess.EDGE_ADVECT)
            if self.recording_config['screenshot_mode'] & ProcessRenderMode.EDGE_ITERATIONS:
                self.generate_images()

    def viewed_process_loop(self) -> None:
        self.viewed_node_process()
        while not self.processor.action_finished:
            self.viewed_node_process()

        self.processor.reset_edges()
        self.generate_images()

        self.cam.rotate_around_base = self.recording_config['camera_rotation']
        self.viewed_edge_process()
        while not self.processor.action_finished:
            self.viewed_edge_process()

        self.processor.edge_processor.sample_edges()
        self.processor.edge_processor.check_limits()
        self.generate_images()

    def render(self, show_class: int = 0) -> None:
        config: RenderingConfig = RenderingConfig()
        self.processor.render(self.cam, config, show_class)

    def process(self) -> None:
        self.processor.reset_edges()
        if not self.recording_config['screenshot_mode'] or self.recording_config[
                'screenshot_mode'] is ProcessRenderMode.FINAL:
            self.process_loop()
        else:
            if self.frame_buffer is not None:
                self.frame_buffer.bind()
            self.viewed_process_loop()

        self.processor.save_model(DATA_PATH + 'model/%s/%s.pro.npz' %
                                  (self.network_name, self.importance_data_name))

        if self.recording_config['screenshot_mode']:
            if self.frame_buffer is not None:
                self.frame_buffer.bind()
                self.cam.set_position(
                    self.recording_config['camera_pose_final'])
                self.render(0)
                create_screenshot(self.recording_config['screenshot_width'], self.recording_config['screenshot_height'],
                                  'network', frame_buffer=self.frame_buffer)

        if self.frame_buffer is not None:
            self.frame_buffer.delete()
        self.clean_up()
