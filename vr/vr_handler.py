from typing import List, Union

import numpy as np
import openvr

from vr.input_handler import VRInputHandler
from vr.render_context import VROpenGLContext
from vr.render_target import VRRenderTarget


class VRHandler:
    def __init__(
        self,
    ) -> None:
        self.vr_system: openvr.IVRSystem = openvr.init(
            openvr.VRApplication_Scene)
        self.vr_compositor: openvr.IVRCompositor = openvr.VRCompositor()
        poses_t = openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount
        self.poses: List[openvr.TrackedDevicePose_t] = poses_t()
        self.head_to_world: Union[np.ndarray, None] = None
        w, h = self.vr_system.getRecommendedRenderTargetSize()
        self.window_width: int = w
        self.window_height: int = h
        self.context: VROpenGLContext = VROpenGLContext(
            self.window_width, self.window_height
        )
        self.context.activate()

        self.targets: List[VRRenderTarget] = []
        for id, vr_eye_id in enumerate([openvr.Eye_Left, openvr.Eye_Right]):
            projection = self.vr_system.getProjectionMatrix(
                vr_eye_id, 0.1, 500.0)
            eye_to_head = self.vr_system.getEyeToHeadTransform(vr_eye_id)
            self.context.update_camera_matrices(id, projection, eye_to_head)
            self.targets.append(VRRenderTarget(vr_eye_id, id, w, h))

        self.input_handler: VRInputHandler = VRInputHandler()

    def update(self) -> bool:
        self.context.update()
        event = openvr.VREvent_t()
        has_events = True
        while has_events:
            has_events = self.vr_system.pollNextEvent(event)
            if event.eventType == openvr.VREvent_TrackedDeviceDeactivated:
                print(f'Device {event.trackedDeviceIndex} detached')
            elif event.eventType == openvr.VREvent_TrackedDeviceUpdated:
                print(f'Device {event.trackedDeviceIndex} updated')

        self.input_handler.update()

        for cam in self.context.cam:
            if self.input_handler.origin_update:
                cam.set_position(self.input_handler.origin)
            cam.apply_input(
                self.input_handler.scaling,
                self.input_handler.rotation,
                self.input_handler.grabbed != 0,
                self.input_handler.reset,
            )

        self.poses, _ = self.vr_compositor.waitGetPoses(self.poses, None)

        head_pose = self.poses[openvr.k_unTrackedDeviceIndex_Hmd]

        if not head_pose.bPoseIsValid:
            return False
        else:
            self.head_to_world = head_pose.mDeviceToAbsoluteTracking.m
        for cam in self.context.cam:
            cam.update_head(self.head_to_world)

        return True

    def submit_target_texture(self, target: VRRenderTarget) -> None:
        self.vr_compositor.submit(target.vr_eye_id, target.vr_texture)

    def destroy(self) -> None:
        self.context.destroy()
