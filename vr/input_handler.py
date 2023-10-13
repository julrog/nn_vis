import os
from typing import List, Optional, Tuple

import numpy as np
import openvr

from definitions import BASE_PATH


class HandInput:
    def __init__(self, side: str = 'right') -> None:
        self.side: str = side
        self.action_hand_source: int = openvr.VRInput().getInputSourceHandle(
            f'/user/hand/{self.side}'
        )
        self.action_pose: int = openvr.VRInput().getActionHandle(
            f'/actions/demo/in/hand_{self.side}'
        )
        self.action_haptic: int = openvr.VRInput().getActionHandle(
            f'/actions/demo/out/haptic_{self.side}'
        )
        self.action_grip: int = openvr.VRInput().getActionHandle(
            f'/actions/demo/in/grip_{self.side}'
        )


class VRInputHandler:
    def __init__(self) -> None:
        control_manifest_path: str = os.path.join(
            BASE_PATH, 'vr', 'actions.json')
        openvr.VRInput().setActionManifestPath(control_manifest_path)

        self.action_set_demo: int = openvr.VRInput().getActionSetHandle(
            '/actions/demo'
        )
        self.action_rotate_model: int = openvr.VRInput().getActionHandle(
            '/actions/demo/in/rotate_model'
        )
        self.action_shrink: int = openvr.VRInput().getActionHandle(
            '/actions/demo/in/shrink'
        )
        self.action_grow: int = openvr.VRInput().getActionHandle(
            '/actions/demo/in/grow'
        )
        self.action_reset: int = openvr.VRInput().getActionHandle(
            '/actions/demo/in/reset'
        )
        self.action_rotate_class: int = openvr.VRInput().getActionHandle(
            '/actions/demo/in/rotate_class_mode'
        )
        self.action_rotate_render: int = openvr.VRInput().getActionHandle(
            '/actions/demo/in/rotate_render_mode'
        )

        self.hands: List[HandInput] = [HandInput('left'), HandInput('right')]

        self.scaling: float = 0.0
        self.grabbed: int = 0
        self.origin: Optional[np.ndarray] = None
        self.origin_update: bool = False
        self.rotation: List[float] = [0, 0]
        self.reset: bool = False
        self.rotate_class: bool = False
        self.rotate_render: bool = False
        self.current_render_mode: int = 0

    def get_release_action(self, action: int) -> bool:
        action_data = openvr.VRInput().getDigitalActionData(
            action, openvr.k_ulInvalidInputValueHandle
        )
        return action_data.bActive and action_data.bChanged and not action_data.bState

    def get_pressed_action(self, action: int) -> bool:
        action_data: openvr.InputDigitalActionData_t = (
            openvr.VRInput().getDigitalActionData(
                action, openvr.k_ulInvalidInputValueHandle
            )
        )
        return action_data.bActive and action_data.bState

    def update_scaling_action(self) -> None:
        shrink_release: bool = self.get_release_action(self.action_shrink)
        grow_release: bool = self.get_release_action(self.action_grow)

        if shrink_release and not grow_release:
            self.scaling = 0.8
        elif not shrink_release and grow_release:
            self.scaling = 1.25
        else:
            self.scaling = 1.0

    def update_grabbing_action(self) -> None:
        left_grip_pressed: bool = self.get_pressed_action(
            self.hands[0].action_grip)
        right_grip_pressed: bool = self.get_pressed_action(
            self.hands[1].action_grip)

        if left_grip_pressed:
            self.grabbed = -1
        elif right_grip_pressed:
            self.grabbed = 1
        else:
            self.grabbed = 0

    def get_pose(self, action: int) -> Tuple[bool, np.ndarray]:
        pose_data: openvr.InputPoseActionData_t = (
            openvr.VRInput().getPoseActionDataForNextFrame(
                action,
                openvr.TrackingUniverseStanding,
                openvr.k_ulInvalidInputValueHandle,
            )
        )
        if not pose_data.bActive or not pose_data.pose.bPoseIsValid:
            return False, pose_data.pose.mDeviceToAbsoluteTracking.m
        else:
            return True, pose_data.pose.mDeviceToAbsoluteTracking.m

    def update_hand_poses(self) -> None:
        self.origin_update = False
        for id, hand in enumerate(self.hands):
            pose_update, pose = self.get_pose(hand.action_pose)
            if id * 2 - 1 == self.grabbed:
                if pose_update:
                    self.origin = pose
                self.origin_update = pose_update

    def update(self) -> None:
        action_sets: List[openvr.VRActiveActionSet_t] = (
            openvr.VRActiveActionSet_t * 1
        )()
        action_set: openvr.VRActiveActionSet_t = action_sets[0]
        action_set.ulActionSet = self.action_set_demo
        openvr.VRInput().updateActionState(action_sets)

        self.update_scaling_action()
        self.update_grabbing_action()
        self.update_hand_poses()

        self.reset = self.get_release_action(self.action_reset)
        self.rotate_class = self.get_release_action(self.action_rotate_class)
        self.rotate_render = self.get_release_action(self.action_rotate_render)

        analog_data: openvr.InputAnalogActionData_t = (
            openvr.VRInput().getAnalogActionData(
                self.action_rotate_model, openvr.k_ulInvalidInputValueHandle
            )
        )
        self.rotation = [analog_data.x, analog_data.y]
