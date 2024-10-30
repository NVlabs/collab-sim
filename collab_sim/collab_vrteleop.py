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


"""
Main VR Class
"""

# External:
import transforms3d as t3d
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Callable, Tuple
import os
from typing import Optional
############################################################
# XR:
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.kit.xr.profile.vr")
from omni.kit.xr.core import XRUtils
from omni import usd
import carb
carb.settings.get_settings().set("/xr/profile/vr/enabled", True) 
# buttons:
from omni.kit.xr.core import XRCore
from omni.kit.xr.core import XRGestureEventType
############################################################
# Isaac Sim:
from omni import usd
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.cortex.cortex_object import CortexObject
from scipy.spatial.transform import Rotation as R
from pxr import Gf, UsdGeom, Sdf
from omni.isaac.core.utils.prims import add_reference_to_stage
############################################################
# collab-sim data dir 
EXT_DIR = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__))))
DATA_DIR = os.path.join(EXT_DIR, "data")
############################################################
# collab-sim:
from collab_sim import collab_teleop_utils
ft = collab_teleop_utils.FramesTransforms()
############################################################


class VRTeleop(ABC):

    def __init__(self, world):
        """ 
        Initializes the VR class, get current profile
        """
        print ("VRTeleop initialization ==============")
        self.world = world     
        self.profile = XRCore.get_singleton().get_current_profile()


    def set_up_vr_teleop(self, robot, eegoalprim):
        """ 
        Sets default teleop workflow:
        1-creates visual prims to follow the vr controllers
        2-assigns a prim to be teleop by the primary vr controller
        """
        self.robot = robot #e.g. my_franka

        self.CONTROL_RIGHT = 0
        self.gripper_opened = True
        
        # create vr controller-following frames at origin
        self.target_prim_vrleft  = self.create_target_prim(prim_path="/World/VRteleop/vrleft") 
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
        self.left_np_p=np_p
        self.left_rf_wxyz=rf_wxyz
        
        # Right:
        np_p, np_quat_wxyz, np_quat_xyzW = self.get_np_p_q_from_usd_transform(mat_rightx)
        rf_wxyz = self.transform_usdcontrollerframe_to_eegoalframe(np_quat_xyzW)
        # set_world_pose of right-controller-following frame:
        self.target_prim_vrright.set_world_pose(np_p, rf_wxyz)
        self.right_np_p=np_p
        self.right_rf_wxyz=rf_wxyz


    def transform_usdcontrollerframe_to_eegoalframe(self, np_quat_xyzW):
        # 'X' intrinsic
        r1 = R.from_euler('Z', 90, degrees=True) #90 degrees rotation around Z
        r2 = R.from_quat([np_quat_xyzW]) #r2 is input quat (world frame)
        r3 = r2*r1 # sequential
        r4 = R.from_euler('Y', -90, degrees=True) #rot -90 around Y
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

    def get_device_prim_path(self, device_name: str) -> Optional[str]:
        """ 
        Gets the prim path for xr devices, e.g. vr controllers
        """
        xr_gui_manager = self.profile.get_xr_gui_manager()
        xr_gui_layer = xr_gui_manager.get_xr_usd_layer("xr_gui")

        device = self.profile.get_device(device_name)
        if not device:
            return None

        prim_path = xr_gui_layer.ensure_device_prim_path(device)
        if prim_path.startswith("/_xr_gui/noxr/"):
            return None

        local_stage = usd.get_context().get_stage()
        if not local_stage.GetPrimAtPath(prim_path):
            return None

        return prim_path    


    def get_vr_controllers_poses (self):
        # print("cd get_vr_controllers_poses -------+++++++++++++++++")
        left_controller_prim_path = self.get_device_prim_path("xrcontroller0")
        right_controller_prim_path = self.get_device_prim_path("xrcontroller1")
        mat_left = XRUtils.get_singleton().get_world_transform_matrix(left_controller_prim_path).RemoveScaleShear()
        mat_right = XRUtils.get_singleton().get_world_transform_matrix(right_controller_prim_path).RemoveScaleShear()
        # print (mat_left)
        # print (mat_right)
        return (mat_left, mat_right)


    def action_primary_select_teleopdefault (self, ev):
        """
        Default callback for triger button on primary controller
        Action called each step that button is pressed
        Enabled if run init_vr_buttons()
        This action sets the position of the end-effector goal for teleop
        (This pose is to be read in the main sim loop to command the teleop)
        """

        if ev.type == XRGestureEventType.begin:
            # press BEGIN condition
            print("Down")

            # current pose of VR right controller (at button press):
            vr_p, vr_quat = self.target_prim_vrright.get_world_pose()
            self.world_to_vr_start = ft.transform_from_pq(p=vr_p, quat=vr_quat)

            # current pose of EE goal:
            current_p, current_q = self.vr_target_motion_controller.get_world_pose()
            self.world_to_eegoal_start = ft.transform_from_pq(p=current_p, quat=current_q) #ee_goal pose

        elif ev.type == XRGestureEventType.end:
            print("Release")
            self.CONTROL_RIGHT = 0

        else:
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


    def action_primary_grab_teleopdefault (self, ev):
        """
        Default callback for side buttons on primary controller
        Action called each step that button is pressed
        Enabled if run init_vr_buttons()
        This action opens/closes the gripper
        """
        if ev.type == XRGestureEventType.begin:
            # press BEGIN condition
            if self.gripper_opened:
                self.robot.gripper.close() 
            else:
                self.robot.gripper.open() 
            self.gripper_opened = not self.gripper_opened    
        elif ev.type == XRGestureEventType.end:
            print("Release1")     


    def action_secondary_select_teleopdefault (self, ev):
        """
        Default callback for triger button on primary controller
        Action called each step that button is pressed
        Enabled if run init_vr_buttons()
        This action resets the enviroment to start a new episode
        """
        # action_primary_down called each step that button is pressed
        if ev.type == XRGestureEventType.begin:
            # press BEGIN condition
            print("RESET TELEOP FRAME")
            # there is a segfault issue here if the headset or controllers were not moving
            # need to be active (colored) in steamvr   
            # self.world.stop()
            self.world.reset() #(soft=False)
            self.teleport_headset_to_start()
        elif ev.type == XRGestureEventType.end:
            print("action_secondary_select_teleopdefault Release")            


    # @abstractmethod
    def press1(self, ev):
        """
        Example placeholder for custom button actions
        Action called each step that button is pressed
        This method must be implemented in the subclass
        """
        pass


    # @abstractmethod
    def press2(self, ev):
        """
        Example placeholder for custom button actions
        Action called each step that button is pressed
        This method must be implemented in the subclass
        """
        pass


    def init_vr_buttons_general(self, action_callback_pairs: List[Tuple[str, Callable]]):
        """ 
        -Creates the callbacks for vr controller buttons,
        -given a general mapping in action_callback_pairs
        """
        if XRCore.get_singleton().is_xr_enabled():
            self.profile = XRCore.get_singleton().get_current_profile()
            profile = self.profile
            print ("Profile: {profile}")

            avatar = profile.get_avatar()
            print ("Avatar: {avatar}")

            layers = avatar.get_tool_layers()
            interaction_layer = layers[0]
            
            for action_name, callback_name in action_callback_pairs:
                gestures = interaction_layer.get_gestures_for_action(action_name)
                custom_gesture = gestures[0]
                interaction_layer.bind_gesture(custom_gesture, callback_name)
        else:
            print("Tried to initialize buttons but VR is not running")


    def init_vr_buttons(self):
        """ 
        -Creates mapping from buttons to callbacks
        -This is the default mapping, but the user might create 
        a custom mapping and call init_vr_buttons_general
        -Primary and secondary controllers are the 2 vr controllers
        -select = trigger
        -grab = side buttons
        """
        # vr buttons callbacks:
        action_callback_pairs = [
            ("primary_controller:sel:select",   self.action_primary_select_teleopdefault),
            ("primary_controller:grab",         self.action_primary_grab_teleopdefault),
            ("secondary_controller:sel:select", self.action_secondary_select_teleopdefault) # active for resets, remove for dual arm 
        ]
        self.init_vr_buttons_general(action_callback_pairs)

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
        xform = UsdGeom.Xform(self.world.stage.GetPrimAtPath('/World/robot'))
        mat1 = xform.ComputeLocalToWorldTransform(float("NaN"))
        rotation_matrix = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), -90))
        mat = rotation_matrix * mat1
        offset = vec = Gf.Vec3d(-0.1, 0.3, 0.6) 
        mat = mat * Gf.Matrix4d().SetTranslate(offset)
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