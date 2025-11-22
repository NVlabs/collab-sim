# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#


"""Main VR teleoperation class for robot control."""

# Standard library
import os
import time
from abc import ABC, abstractmethod
from typing import List, Callable, Tuple, Optional

# Third-party
import numpy as np
import transforms3d as t3d
from scipy.spatial.transform import Rotation as R

# Isaac Sim / Omniverse
import carb
from isaacsim.core.utils.stage import add_reference_to_stage
from omni import usd
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.cortex.cortex_object import CortexObject
from omni.kit.xr.core import XRCore
from pxr import Gf, UsdGeom, Sdf

# Local
from collab_sim import collab_teleop_utils

# Module-level constants
EXT_DIR = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__))))
DATA_DIR = os.path.join(EXT_DIR, "data")

ft = collab_teleop_utils.FramesTransforms()

# VR controller rotation constants (degrees)
VR_ROTATION_Z = 90
VR_ROTATION_Y = -90

# VR headset teleport constants
VR_HEADSET_ROTATION_DEGREES = -90
VR_HEADSET_OFFSET = Gf.Vec3d(-0.1, 0.3, 0.2)


class VRTeleop(ABC):

    def __init__(self, world):
        """Initialize the VR teleoperation class.

        Args:
            world: Isaac Sim World instance
        """
        print("VRTeleop initialization ==============")
        self.world = world
        self.on_robot_reset_callback = None

    def is_vr_initialized(self):
        """Check if VR is currently running and enabled.

        Returns:
            bool: True if VR is enabled, False otherwise
        """
        return XRCore.get_singleton().is_xr_enabled()

    def set_up_vr_devices_with_active_vr(self):
        """Initialize VR devices and controllers (requires VR already enabled)."""
        self.profile = XRCore.get_singleton().get_current_profile()
        self.left_controller_device = XRCore.get_singleton().get_input_device("/user/hand/left")
        self.right_controller_device = XRCore.get_singleton().get_input_device("/user/hand/right")

    def set_up_vr_teleop_frames(self, robot, eegoalprim):
        """Set up VR teleoperation frames and targets.

        Creates visual prims that follow VR controllers and assigns the
        end-effector goal prim teleop by the primary vr controller.

        Args:
            robot: Robot instance (e.g., Franka)
            eegoalprim: End-effector goal prim for teleoperation
        """
        self.robot = robot

        self.CONTROL_RIGHT = 0
        self.gripper_opened = True

        # Create VR controller-following frames at origin
        self.target_prim_vrleft = self.create_target_prim(prim_path="/World/VRteleop/vrleft")
        self.target_prim_vrright = self.create_target_prim(prim_path="/World/VRteleop/vrright")

        self.world_to_eegoal_start = ft.identity_transform()
        self.world_to_eegoal_new = ft.identity_transform()
        self.world_to_vr_start = ft.identity_transform()
        self.world_to_vr_new = ft.identity_transform()
        self.delta_vr_T = ft.identity_transform()

        # assign the prim that vr will be updating as vr goal frame
        self.vr_target_motion_controller = eegoalprim 
        # and keep it's first pose for resets:
        p,q = eegoalprim.get_world_pose()
        self.vr_target_motion_controller_init_p = p
        self.vr_target_motion_controller_init_quat = q


    def reset_target_frame(self):
        """
        Resets positions of the teleop target prim to the initial recorded positions
        Used for env resets
        """
        self.vr_target_set_pose(self.vr_target_motion_controller_init_p, self.vr_target_motion_controller_init_quat)

    def set_robot_reset_callback(self, callback: Callable):
        """Set a callback function to be called after VR controller reset.

        The callback is called after world.reset() to move the robot back to
        its desired position (e.g., on the table instead of at origin).

        Args:
            callback: Function to call after reset (no arguments)
        """
        self.on_robot_reset_callback = callback
        

    def create_target_prim(self, prim_path="/World/", scale=0.4):
        """ 
        Creates an visual frame at the origin
        """
        usd_path = os.path.join(DATA_DIR, "axis.usda")
        target_prim = add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        target_prim.CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
        xformable = XFormPrim(str(target_prim.GetPath()), "motion_controller_target")  
        xformable.set_local_scale(scale*np.ones(3))
        return CortexObject(xformable) #rm todo
    

    def setpose_vrcontrollersfollower_frames(self, controller_pilot = 1):
        """ 
        Updates the pose of the visual frames that follow the vr controllers poses
        set_world_pose of vr-controllers-following frame-prims
        """

        # Get pose of VR controllers (as transforms)
        mat_leftx, mat_rightx = self.get_vr_controllers_poses() # in world frame

        # Left:
        np_p, np_quat_wxyz, np_quat_xyzW = self.get_np_p_q_from_usd_transform(mat_leftx)
        rf_wxyz = self.transform_usdcontrollerframe_to_eegoalframe(np_quat_xyzW)
        # set_world_pose of left-controller-following frame:
        self.target_prim_vrleft.set_world_pose(np_p, rf_wxyz)
        self.left_np_p = np_p
        self.left_rf_wxyz = rf_wxyz

        # Right:
        np_p, np_quat_wxyz, np_quat_xyzW = self.get_np_p_q_from_usd_transform(mat_rightx)
        rf_wxyz = self.transform_usdcontrollerframe_to_eegoalframe(np_quat_xyzW)
        # set_world_pose of right-controller-following frame:
        self.target_prim_vrright.set_world_pose(np_p, rf_wxyz)
        self.right_np_p = np_p
        self.right_rf_wxyz = rf_wxyz


    def transform_usdcontrollerframe_to_eegoalframe(self, np_quat_xyzW):
        """Transform VR controller frame to end-effector goal frame.

        Applies rotation transformations to align VR controller orientation
        with robot end-effector frame convention.

        Args:
            np_quat_xyzW: Quaternion in xyzW format

        Returns:
            np.array: Quaternion in wxyz format
        """
        # Check for zero norm quaternion (VR not connected)
        quat_norm = np.linalg.norm(np_quat_xyzW)
        if quat_norm < 1e-6:
            # Return identity quaternion if VR controller not available
            print("[WARN] VR controller quaternion is zero - VR may not be connected")
            return np.array([1.0, 0.0, 0.0, 0.0])  # wxyz identity

        # 'X' intrinsic
        r1 = R.from_euler('Z', VR_ROTATION_Z, degrees=True)
        r2 = R.from_quat([np_quat_xyzW])
        r3 = r2 * r1 #sequential 
        r4 = R.from_euler('Y', VR_ROTATION_Y, degrees=True)
        r5 = r3 * r4
        rf = r5
        rf_xyzW = rf.as_quat()
        rf_xyzW_ = rf_xyzW[0]
        rf_wxyz = np.array([rf_xyzW_[3], rf_xyzW_[0], rf_xyzW_[1], rf_xyzW_[2]])
        return rf_wxyz

    def vr_target_set_pose(self, p, q):
        self.motion_target_p = p
        self.motion_target_q = q
        self.vr_target_motion_controller.set_world_pose(self.motion_target_p, self.motion_target_q)

    def vr_target_set_position(self, p):
        current_p, current_q = self.vr_target_motion_controller.get_world_pose()
        self.motion_target_p = p
        self.motion_target_q = current_q
        self.vr_target_motion_controller.set_world_pose(self.motion_target_p, self.motion_target_q)      

    def vr_target_set_orientation(self, q):    
        current_p, current_q = self.vr_target_motion_controller.get_world_pose()
        self.motion_target_p = current_p
        self.motion_target_q = q
        self.vr_target_motion_controller.set_world_pose(current_p, q)    
    
        
    def get_vr_controllers_poses (self):
        mat_left  = self.left_controller_device.get_virtual_world_pose() 
        mat_right = self.right_controller_device.get_virtual_world_pose() 
        # print(f"###Pose left controller: {mat_left}")
        # print(f"###Pose right controller: {mat_right}")
        return (mat_left, mat_right)


    def teleop_action_on_button_press(self):
        print("Down")

        # current pose of VR right controller (at button press):
        vr_p, vr_quat = self.target_prim_vrright.get_world_pose()
        self.world_to_vr_start = ft.transform_from_pq(p=vr_p, quat=vr_quat)

        # current pose of EE goal:
        current_p, current_q = self.vr_target_motion_controller.get_world_pose()
        self.world_to_eegoal_start = ft.transform_from_pq(p=current_p, quat=current_q) #ee_goal pose


    
    def teleop_action_while_button_pressed(self):
        # while pressed:
        vr_p, vr_quat = self.target_prim_vrright.get_world_pose()
        self.world_to_vr_new = ft.transform_from_pq(p=vr_p, quat=vr_quat)

        delta_vr_t_in_world, delta_vr_rot_in_start_vr_frame = ft.delta_transform(self.world_to_vr_start, self.world_to_vr_new)

        eegoal_rot_start_in_world = ft.rotation_from_transform(self.world_to_eegoal_start)
        eegoal_rot_new_in_world = ft.concatenate_transforms(delta_vr_rot_in_start_vr_frame, eegoal_rot_start_in_world)
        
        self.world_to_eegoal_new = ft.transform_from_pq( ft.position_from_transform(self.world_to_eegoal_start)+delta_vr_t_in_world,
                                                            t3d.quaternions.mat2quat(eegoal_rot_new_in_world)   )
        
        # Set position of VR goal frame - to be read on main sim loop:
        self.vr_target_motion_controller.set_world_pose(ft.position_from_transform(self.world_to_eegoal_new), ft.quat_from_transform(self.world_to_eegoal_new)) 



    def teleop_action_on_button_release(self):
        print("Release")
        self.CONTROL_RIGHT = 0


    def gripper_action_on_button_press(self):
        print("Gripper action ====================<<o>>==================")
        if self.gripper_opened:
            self.robot.gripper.close() 
        else:
            self.robot.gripper.open() 
        self.gripper_opened = not self.gripper_opened    


    def reset_action_on_button_press(self):
        """
        Default callback for trigger button on secondary controller
        Action called each step that button is pressed
        Enabled if run init_vr_buttons()
        This action resets the environment to start a new episode
        """
        # action_primary_down called each step that button is pressed
        # if ev.type == XRGestureEventType.begin:
        # press BEGIN condition

        print("RESET TELEOP FRAME")
        # there is a segfault issue here if the headset or controllers were not moving
        # need to be active (colored) in steamvr
        # self.world.stop()
        self.world.reset()  # (soft=False)
        self.teleport_headset_to_start()
        self.reset_target_frame()
        # Call robot reset callback to move robot back to desired position
        if self.on_robot_reset_callback is not None:
            self.on_robot_reset_callback()


    def init_vr_leftcont_buttons_righthanded_teleop_default(self):
        """
        - Single arm teleop default
        - Creates buttons/action mapping to functions
        - For right-handed defaults teleop
        - **Left hand: trigger = reset env, side button = not used**
        """

        # left controller:
        self.left_trigger_button_manager = VRButtonManager(input_device_path = "/user/hand/left", button_name = "trigger", gesture_name = "click",
                                                           on_press=self.reset_action_on_button_press, 
                                                           on_while_pressed=None, 
                                                           on_release=None)
        self.left_squeeze_button_manager = VRButtonManager(input_device_path = "/user/hand/left", button_name = "squeeze", gesture_name = "click",
                                                           on_press=None, on_while_pressed=None, on_release=None)
        self.left_trigger_button_manager.print_all_buttons_this_device()
        

    def init_vr_rightcont_buttons_righthanded_teleop_default (self):
        """
        - Single arm teleop default
        - Creates buttons/action mapping to functions
        - For right-handed defaults teleop
        - **Right hand: trigger = teleop, side button = gripper open/close**
        """
        self.right_trigger_button_manager = VRButtonManager(input_device_path = "/user/hand/right", button_name = "trigger", gesture_name = "click",
                                                    on_press=self.teleop_action_on_button_press, 
                                                    on_while_pressed=self.teleop_action_while_button_pressed,
                                                    on_release=self.teleop_action_on_button_release)
        self.right_squeeze_button_manager = VRButtonManager(input_device_path = "/user/hand/right", button_name = "squeeze", gesture_name = "click",
                                                            on_press=self.gripper_action_on_button_press,
                                                            on_while_pressed=None, 
                                                            on_release=None)
        self.right_trigger_button_manager.print_all_buttons_this_device()

    def init_vr_thumbstick_reset(self, controller="left"):
        """
        Initialize thumbstick click as reset button.
        Useful for dual-arm teleop where both triggers are used for teleop.

        Args:
            controller: "left" or "right" - which controller's thumbstick to use
        """
        input_device_path = f"/user/hand/{controller}"
        self.thumbstick_reset_button_manager = VRButtonManager(
            input_device_path=input_device_path,
            button_name="thumbstick",
            gesture_name="click",
            on_press=self.reset_action_on_button_press,
            on_while_pressed=None,
            on_release=None
        )
        print(f"Thumbstick reset initialized on {controller} controller")

    def init_vr_squeeze_long_press_reset(self, controller="left", long_press_seconds=5.0):
        """
        Initialize squeeze button with long-press reset functionality.
        Short press (< long_press_seconds) = gripper toggle
        Long press (>= long_press_seconds) = environment reset

        Args:
            controller: "left" or "right" - which controller's squeeze button to use
            long_press_seconds: Duration in seconds to trigger reset (default 5.0)
        """
        self._squeeze_press_start_time = None
        self._squeeze_reset_triggered = False
        self._long_press_seconds = long_press_seconds

        def on_squeeze_press():
            self._squeeze_press_start_time = time.time()
            self._squeeze_reset_triggered = False
            print(f"Squeeze pressed - hold for {self._long_press_seconds}s to reset")

        def on_squeeze_while_pressed():
            if self._squeeze_press_start_time is None:
                return
            elapsed = time.time() - self._squeeze_press_start_time
            if elapsed >= self._long_press_seconds and not self._squeeze_reset_triggered:
                print(f"Long press detected ({elapsed:.1f}s) - triggering reset!")
                self._squeeze_reset_triggered = True
                self.reset_action_on_button_press()

        def on_squeeze_release():
            if self._squeeze_press_start_time is None:
                return
            elapsed = time.time() - self._squeeze_press_start_time
            if not self._squeeze_reset_triggered:
                # Short press - toggle gripper
                print(f"Short press ({elapsed:.1f}s) - toggling gripper")
                self.gripper_action_on_button_press()
            self._squeeze_press_start_time = None

        input_device_path = f"/user/hand/{controller}"
        self.squeeze_long_press_button_manager = VRButtonManager(
            input_device_path=input_device_path,
            button_name="squeeze",
            gesture_name="click",
            on_press=on_squeeze_press,
            on_while_pressed=on_squeeze_while_pressed,
            on_release=on_squeeze_release
        )
        print(f"Squeeze long-press reset initialized on {controller} controller ({long_press_seconds}s hold = reset)")



    def teleport_headset_to_pose(self, pose_matrix4d):
        """ 
        Teleports the origin of the VR HMD
        Usefull for instant telport and env resets poses
        """
        self.profile.teleport(pose_matrix4d)


    def teleport_headset_to_start(self):
        """
        Teleports the VR HMD to a transform with respect to the robot origin
        Useful for env resets
        todo: pass pose as argument
        """
        # Check if profile exists (only set up on primary VR controller)
        if not hasattr(self, 'profile') or self.profile is None:
            print("[INFO] teleport_headset_to_start: No VR profile, skipping teleport")
            return

        # Try common robot prim paths
        robot_prim = None
        for robot_path in ['/World/robot', '/World/robot1', '/World/robot2']:
            prim = self.world.stage.GetPrimAtPath(robot_path)
            if prim.IsValid():
                robot_prim = prim
                break

        if robot_prim is None:
            print("[WARN] teleport_headset_to_start: No robot prim found, skipping teleport")
            return

        xform = UsdGeom.Xform(robot_prim)
        mat1 = xform.ComputeLocalToWorldTransform(float("NaN"))
        rotation_matrix = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), VR_HEADSET_ROTATION_DEGREES))
        mat = rotation_matrix * mat1
        mat = mat * Gf.Matrix4d().SetTranslate(VR_HEADSET_OFFSET)
        self.teleport_headset_to_pose(mat)


    def get_matrix4d(self, x, y, z, rotation_matrix):
        """ 
        Transforms utils
        """
        pose_matrix = Gf.Matrix4d(1.0)
        translation = Gf.Vec3d(x, y, z)
        pose_matrix.SetTranslate(translation)
        pose_matrix.SetRotate(rotation_matrix)
        return pose_matrix    


    def get_np_p_q_from_usd_transform(self, T):
        """ 
        Transforms utils
        """
        translation: Gf.Vec3d = T.ExtractTranslation()
        quat: Gf.Rotation = T.ExtractRotationQuat().GetNormalized()
        scale: Gf.Vec3d = Gf.Vec3d(*(v.GetLength() for v in T.ExtractRotationMatrix()))

        qw = quat.GetReal()
        qxyz = quat.GetImaginary()

        np_quat_Wxyz = np.array([qw, qxyz[0], qxyz[1], qxyz[2]])
        np_quat_xyzW = np.array([qxyz[0], qxyz[1], qxyz[2], qw])
        np_p = np.array(translation)

        return np_p, np_quat_Wxyz, np_quat_xyzW
    

class VRButtonManager():

    def __init__(self, input_device_path, button_name, gesture_name, on_press=None, on_while_pressed=None, on_release=None):
        
        self.on_press = on_press or self.default_on_press
        self.on_while_pressed = on_while_pressed or self.default_while_pressed
        self.on_release = on_release or self.default_on_release

        self.prev_state = 0
        self.input_device_path = input_device_path
        self.button_name = button_name
        self.gesture_name = gesture_name
        self.input_device = XRCore.get_singleton().get_input_device(input_device_path)
        

    def default_on_press(self):
        print("Button pressed")

    def default_while_pressed(self):
        print("Button is being held down")

    def default_on_release(self):
        print("Button released")

    def read_button_state(self):
        current_state = self.input_device.get_input_gesture_value(self.button_name, self.gesture_name)
        return current_state

    def update(self):
        """Update button state and trigger appropriate callbacks.

        Queries the button state and calls on_press, on_while_pressed, or
        on_release based on state transitions. "click" gestures follow
        0->1->1->0 logic, while "x" and "y" gestures always trigger on_press.
        """
        current_state = self.read_button_state()
        print(f"Button state: {current_state}")

        if self.gesture_name == "click":
            state_transition = (self.prev_state, current_state)
            if state_transition == (0, 1):
                self.on_press()
            elif state_transition == (1, 1):
                self.on_while_pressed()
            elif state_transition == (1, 0):
                self.on_release()

            # Update for next iteration
            self.prev_state = current_state

        elif self.gesture_name in ("x", "y"):
            self.on_press()  # Always call on_press for joystick



    def print_all_buttons_this_device(self):
        button_names = self.input_device.get_input_names()
        if len(button_names) > 0:
            for buttonnameXRToken in button_names:
                gestures = self.input_device.get_input_gesture_names(buttonnameXRToken)
                for gestureXRToken in gestures:
                    value = self.input_device.get_input_gesture_value(buttonnameXRToken, gestureXRToken)
                    print(f"Button {buttonnameXRToken}, Gesture {gestureXRToken}: {value} ")

