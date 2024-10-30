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


# isaac sim 2023.1.1
# cuda 11.8, python 3.10

    # collab-sim must be visible from python: 
    # export PYTHONPATH=$PYTHONPATH:"<...>/collab-sim"
    # useconda
    # usecuda118
    # conda activate vrcurobo_202311
    # useisaac23_1_1_rc8
    # vr: python franka_mpc_example.py --runvr --log_data --relevant_objects_str Cube P3 P4
    # non-vr: python franka_mpc_example.py --use_keyboard --log_data --relevant_objects_str P3 P4

import argparse
############################################################
# Config:
# config to be used if ran with no arguments, easy for debugging:
debug_run_vr = False 
debug_print_degub = True
debug_use_keyboard = False 
debug_log_data = False
debug_relevant_objects_str = ["Cube", "P3", "P4", "table"]
load_scene_usd = False 
# load_tabletop_scene = False
# or overide args if provided:
parser = argparse.ArgumentParser()
parser.add_argument("--run_vr", action='store_true', default=debug_run_vr, help="Enable VR mode (default: debug)")
parser.add_argument("--print_debug", action='store_true', default=debug_print_degub, help="Enable debug printing (default: debug)")
parser.add_argument("--use_keyboard", action='store_true', default=debug_use_keyboard, help="Enable keyboard open/close Franka gripper (default: debug)")
parser.add_argument("--log_data", action='store_true', default=debug_log_data, help="Enable data logging (default: debug)")
parser.add_argument(
    "--relevant_objects_str", 
    nargs='+', 
    default=debug_relevant_objects_str, 
    help="str for prim names to save in addition to scene registry (robot) (default: ['Cube', 'P3', 'P4'])"
)
args = parser.parse_args()
############################################################
if args.print_debug:
    print (args)
    # input("Press any key to continue...")
############################################################
# external:
import transforms3d as t3d
import torch
import numpy as np
import os
import time
import math
# curobo:
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
tensor_args = TensorDeviceType()
############################################################
# Isaac Sim:
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
# import carb
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from typing import Optional
import omni.appwindow  # Contains handle to keyboard
import time
############################################################
EXT_DIR = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__))))
DATA_DIR = os.path.join(EXT_DIR, "data")
############################################################
# collab-sim:
from collab_sim import collab_robot_controller
from collab_sim import collab_teleop_utils
############################################################ 
COLLAB_DIR = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..'))
DEMOS_VR_DIR = os.path.join(COLLAB_DIR, "data/demosVR") #dir for saving demo data for reply
DEMOSUSD_DIR = os.path.join(COLLAB_DIR, "data/demosUSD") #
SCENEUSD_DIR = os.path.join(COLLAB_DIR, "data/sceneUSD") #example USD envs
DATA_DIR = os.path.join(COLLAB_DIR, "collab_sim/data") # axis usd
############################################################

   
############################################################
############################################################
#used for commanding robot to this initial pose as soon as sim starts:
start_js = {
    "panda_joint1":0.15896346,
    "panda_joint2":-0.07731238,
    "panda_joint3":0.03789043,
    "panda_joint4": -2.2444482,
    "panda_joint5": 0.00352372,
    "panda_joint6":  2.167158,
    "panda_joint7": -2.1614377,
    "panda_finger_joint1": 0.0,
    "panda_finger_joint2": 0.0,
    }
#not same as first ee goal depending on p3x

############################################################
# poses
p_large_scene = [-5.5, 0.0, 0.0]
p_table = [0.0, 0.0, 0.0]
robot_origin_p_in_world_1 = np.array(p_table) + np.array([-0.27, 0.41, 0.82])
robot_origin_p_in_world_2 = np.array(p_table) + np.array([-0.25, -0.45, 0.82]) # + robot_origin_p_in_world_1
# robot_origin_quat_in_world_1 = [math.cos(-3.14 / 8), 0, 0, math.sin(-3.14 / 8)]
robot_origin_quat_in_world_1 = [math.cos(-3.14 / 6), 0, 0, math.sin(-3.14 / 6)] #[1, 0, 0, 0]
robot_origin_quat_in_world_2 = [math.cos(3.14 / 16), 0, 0, math.sin(3.14 / 16)]
p_cubes_min = robot_origin_p_in_world_1 + np.array([0.05, -0.30, 0.03])
p_cubes_max = robot_origin_p_in_world_1 + np.array([0.35, 0.30, 0.03])
cube_xyz_ranges = [
    [p_cubes_min[0], p_cubes_max[0], p_cubes_min[1], p_cubes_max[1], p_cubes_min[2], p_cubes_max[2] ],
    [p_cubes_min[0], p_cubes_max[0], p_cubes_min[1], p_cubes_max[1], p_cubes_min[2], p_cubes_max[2] ],
    [p_cubes_min[0], p_cubes_max[0], p_cubes_min[1], p_cubes_max[1], p_cubes_min[2], p_cubes_max[2] ],
    [p_cubes_min[0], p_cubes_max[0], p_cubes_min[1], p_cubes_max[1], p_cubes_min[2], p_cubes_max[2] ]
]
p_worker = np.array(p_table) + np.array([3.64, -6.50, 0.0]) 

# ROBOT ORIGIN
############################################################
# #robot 1 origin:
# robot_origin_p_in_world_1 = np.array([0.0, 0.0, 0.0])
# robot_origin_quat_in_world_1 = euler_angles_to_quat([0.0, 0.0, 0.0], degrees=True)
# #robot 2 origin:
# robot_origin_p_in_world_2 = robot_origin_p_in_world_1 + np.array([0.0, -0.5, 0.0])
# robot_origin_quat_in_world_2 = euler_angles_to_quat([0.0, 0.0, 0.0], degrees=True)
############################################################


# START EE GOAL:
############################################################
############################################################
#init ee goal 1 (in robot base), used for init mpc solver:
start_ee_goal_p_in_robotbase_1 = np.array([0.3, 0.1, 0.4])
start_ee_goal_quat_in_robotbase_1= euler_angles_to_quat([0, -180, -40], degrees=True)
############################################################
#init ee goal 2 (in robot base), used for init mpc solver:
start_ee_goal_p_in_robotbase_2 = np.array([0.3, 0.1, 0.4])
start_ee_goal_quat_in_robotbase_2= euler_angles_to_quat([0, -180, -20], degrees=True)
############################################################

# Curobo IK config:
config_file = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
urdf_file = config_file["kinematics"]["urdf_path"] 
base_link = config_file["kinematics"]["base_link"]
ee_link = config_file["kinematics"]["ee_link"]
robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)
# robot_cfg = RobotConfig.from_dict(config_file, tensor_args)

############################################################
my_world = World(stage_units_in_meters=1.0)
collab_isaacsim = collab_teleop_utils.IsaacSimUtils(world=my_world)
ft = collab_teleop_utils.FramesTransforms()
############################################################

collab_isaacsim.add_lights()
collab_isaacsim.reset_world_set_default_prim()




######################################
# Add Franka robot:
my_franka_1, my_controller_1, articulation_controller_1 = collab_isaacsim.load_franka(world_to_robotbase=ft.transform_from_pq(robot_origin_p_in_world_1, robot_origin_quat_in_world_1), 
                                                                                franka_name='Franka1',
                                                                                prim_path='/World/robot1')

collab_franka1 = collab_robot_controller.activerobot (  my_franka_1,                             
                                                        articulation_controller_1,
                                                        robot_origin_p_in_world_1, 
                                                        robot_origin_quat_in_world_1, 
                                                        start_ee_goal_p_in_robotbase_1, 
                                                        start_ee_goal_quat_in_robotbase_1)
collab_franka1.init_curobo_manager(robot_cfg)


my_franka_2, my_controller_2, articulation_controller_2 = collab_isaacsim.load_franka(world_to_robotbase=ft.transform_from_pq(robot_origin_p_in_world_2, robot_origin_quat_in_world_2), 
                                                                                franka_name='Franka2',
                                                                                prim_path='/World/robot2')

collab_franka2 = collab_robot_controller.activerobot (  my_franka_2,                             
                                                        articulation_controller_2,
                                                        robot_origin_p_in_world_2, 
                                                        robot_origin_quat_in_world_2, 
                                                        start_ee_goal_p_in_robotbase_2, 
                                                        start_ee_goal_quat_in_robotbase_2)
collab_franka2.init_curobo_manager(robot_cfg)


if load_scene_usd: # dev only
    # kitchencornerstatic_prim, kitchencornerstatic_xform_prim = collab_isaacsim.load_scene_usd_kitchencabinetreplaced()
    k_path = os.path.join(SCENEUSD_DIR, 'Collected_warehouse/warehouse.usd')
    k_prim_path = '/World/collab_largescene'
    k_prim_name = "collab_largescene"
    pos =  np.array(p_large_scene)
    quat = [math.cos(3.14 / 2), 0, 0, math.sin(3.14 / 2)] #np.array([1, 0, 0, 0]) 
    k_prim, k_xform_prim = collab_isaacsim.load_usd(k_path, k_prim_path, k_prim_name, pos, quat, scale=1)
    worker_usd = os.path.join(SCENEUSD_DIR, 'Collected_full_warehouse_worker_and_anim_cameras/worker.usd')
    w_prim, w_xform_prim = collab_isaacsim.load_usd(worker_usd, '/World/collab_worker', "collab_worker", p_worker, [1,0,0,0], scale=1)
else:
    collab_isaacsim.load_default_plane()

collab_isaacsim.load_workshop_table(p_table)

    

############################################################
cube_xyz_ranges = [
    [0.17, 0.17, 0.28, 0.28, 0.86, 0.86],
    [0.08, 0.08, 0.15, 0.15, 0.86, 0.86],
    [0.22, 0.22, -0.09, -0.09, 0.86, 0.86],
    [0.14, 0.14, -0.17, -0.17, 0.86, 0.86]
]
collab_isaacsim.load_random_cubes(cube_xyz_ranges)

############################################################
if args.use_keyboard:
    # configured to open and close the gripper with up/down arrows
    keyboard = collab_teleop_utils.KeyboardTeleopDebug(my_franka_1)

############################################################
# Create prim for teleop goal management:
############

# Viz frame for curobo ee goal in panda_hand frame, it will move to correct pose
p4x = collab_isaacsim.create_visual_frame("P4", "/World/DebugFrames/P4", 
                                         position=ft.position_from_transform(collab_franka1.world_to_starteegoal) , orientation=ft.quat_from_transform(collab_franka1.world_to_starteegoal) )

# Create prim frame for teleop target:
p3x = collab_isaacsim.create_visual_frame("P3", "/World/DebugFrames/P3", 
                                        position=ft.position_from_transform(collab_franka1.world_to_starteegoal) , orientation=ft.quat_from_transform(collab_franka1.world_to_starteegoal) )    

# Create prim frame for teleop target:
p4x_franka2 = collab_isaacsim.create_visual_frame("P4_franka2", "/World/DebugFrames/P4_franka2", 
                                         position=ft.position_from_transform(collab_franka2.world_to_starteegoal) , orientation=ft.quat_from_transform(collab_franka2.world_to_starteegoal) )

p3x_franka2 = collab_isaacsim.create_visual_frame("P3_franka2", "/World/DebugFrames/P3_franka2", 
                                        position=ft.position_from_transform(collab_franka2.world_to_starteegoal) , orientation=ft.quat_from_transform(collab_franka2.world_to_starteegoal) )    


if args.run_vr:
    from collab_sim import collab_vrteleop #Import VR module, enables VR 
    from omni.kit.xr.core import XRGestureEventType
    class Franka2VRController (collab_vrteleop.VRTeleop):

        def init_vr_dual_arm (self):
            pass
        def press1(self, ev):
            # print("press1 action: reset cube and sketch")
            # action_primary_down called each step that button is pressed
            if ev.type == XRGestureEventType.begin:
                # press BEGIN condition
                print("Down")
            elif ev.type == XRGestureEventType.end:
                print("Release")
            else:
                # while pressed:
                print ("pressed")
            
        def press2(self, ev):
            # print("press2 action")
            pass
        
        def press3x(self, ev):
            # print("press1 action")
            pass

        def action_secondary_select_teleop (self, ev):
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

                # current pose of VR LEFT controller (at button press):
                vr_p, vr_quat = self.target_prim_vrleft.get_world_pose()
                self.world_to_vr_start = ft.transform_from_pq(p=vr_p, quat=vr_quat)

                # current pose of EE goal:
                current_p, current_q = self.vr_target_motion_controller.get_world_pose()
                self.world_to_eegoal_start = ft.transform_from_pq(p=current_p, quat=current_q) #ee_goal pose

            elif ev.type == XRGestureEventType.end:
                print("Release")
                # self.CONTROL_RIGHT = 0

            else:
                # while pressed:
                vr_p, vr_quat = self.target_prim_vrleft.get_world_pose()
                self.world_to_vr_new = ft.transform_from_pq(p=vr_p, quat=vr_quat)

                delta_vr_t_in_world, delta_vr_rot_in_start_vr_frame = ft.delta_transform(self.world_to_vr_start, self.world_to_vr_new)

                eegoal_rot_start_in_world = ft.rotation_from_transform(self.world_to_eegoal_start)
                eegoal_rot_new_in_world = ft.concatenate_transforms(delta_vr_rot_in_start_vr_frame, eegoal_rot_start_in_world)
                
                self.world_to_eegoal_new = ft.transform_from_pq( ft.position_from_transform(self.world_to_eegoal_start)+delta_vr_t_in_world,
                                                                t3d.quaternions.mat2quat(eegoal_rot_new_in_world)   )
                
                # Set position of VR goal frame - to be read on main sim loop:
                self.vr_target_motion_controller.set_world_pose(ft.position_from_transform(self.world_to_eegoal_new), ft.quat_from_transform(self.world_to_eegoal_new)) 

        def action_secondary_grab_teleopdefault (self, ev):
            """
            Default callback for side buttons on primary controller
            Action called each step that button is pressed
            Enabled if run init_vr_buttons()
            This action opens/closes the gripper
            """
            if ev.type == XRGestureEventType.begin:
                # press BEGIN condition
                print("press secondary select")
                if self.gripper_opened:
                    self.robot.gripper.close() 
                else:
                    self.robot.gripper.open() 
                self.gripper_opened = not self.gripper_opened    
            elif ev.type == XRGestureEventType.end:
                print("Release secondary select")     
    

    vr_world_1 = collab_vrteleop.VRTeleop(world=my_world)
    vr_world_1.set_up_vr_teleop(robot=my_franka_1, eegoalprim=p3x) #Assign prim for vr teleop
    
    # imports and utils:
    vr_world_2 = Franka2VRController(world=my_world)
    vr_world_2.set_up_vr_teleop(robot=my_franka_2, eegoalprim=p3x_franka2) #Assign prim for vr teleop
    # # vr buttons callbacks:
    action_callback_pairs = [
        ("secondary_controller:sel:select", vr_world_2.action_secondary_select_teleop),
        ("secondary_controller:grab",       vr_world_2.action_secondary_grab_teleopdefault)
    ]


    
############################################################

collab_isaacsim.reset_world()
collab_franka1.franka_setup_gains_reset() #needed after reset - my_world.reset() initializes articulation_view, and resets changes to gains
collab_isaacsim.set_solver_TGS()

if args.log_data:
    sim_data_log = collab_teleop_utils.SimDataLog(args.relevant_objects_str, my_world)
    sim_data_log.save_world_usd(my_world)

if args.run_vr:
    vr_world_1.init_vr_buttons()
    vr_world_2.init_vr_buttons_general(action_callback_pairs)

def main():
    # send robot to initial joint configuration at sim start:
    collab_franka1.reset_robot_states_to_pose(start_js)
    collab_franka2.reset_robot_states_to_pose(start_js)
    collab_isaacsim.step_physics_and_render(100)

    ###########################################
    # #Initialize mpc buffer, which needs an ee_goal_pose and current state
    eegoal_Pose_in_robotbase_1 = Pose(position=tensor_args.to_device(start_ee_goal_p_in_robotbase_1), quaternion=tensor_args.to_device(start_ee_goal_quat_in_robotbase_1))
    collab_franka1.curobomanager.initialize_mpc_buffer(eegoal_Pose_in_robotbase_1)

    eegoal_Pose_in_robotbase_2 = Pose(position=tensor_args.to_device(start_ee_goal_p_in_robotbase_2), quaternion=tensor_args.to_device(start_ee_goal_quat_in_robotbase_2))
    collab_franka2.curobomanager.initialize_mpc_buffer(eegoal_Pose_in_robotbase_2)

    run_sim_to_first_ee_goal = True
    collab_isaacsim.step_render(1000) # time for user to adjust gui


    ########################################################################################################################
    #######################  SIM LOOP ######################################################################################

    while simulation_app.is_running():
        # start_iteration_time = time.time()
        collab_isaacsim.step_physics_and_render(1) #steps physics if my_world.is_playing, renders either play/stop to keep gui interactive

        if my_world.is_stopped(): #reset after stopping sim on the gui
            print ("my_world.reset()")
            my_world.reset()

        if my_world.current_time_step_index == 2: #==2 after a world.reset
            if args.log_data: # and data_dict: #save after world.reset (from vr controller button callback)
                sim_data_log.proccess_and_save_data(my_world.get_physics_dt())
            collab_franka1.franka_setup_gains_reset()
            collab_isaacsim.set_solver_TGS
            # reset USD:
            collab_isaacsim.step_physics_and_render(1)
            collab_isaacsim.step_render(20) 
            collab_isaacsim.reshuffle_cubes(cube_xyz_ranges)
            # vr_world_1.teleport_headset_to_start()
            run_sim_to_first_ee_goal = True # script will continue to compute first franka joint angles for initial p3x world_to_starteegoal and step sim to it

        if run_sim_to_first_ee_goal: #step sim to get robot from init USD state to the initial pose (p3x ee goal)
            p3x.set_world_pose(position=ft.position_from_transform(collab_franka1.world_to_starteegoal) , orientation=ft.quat_from_transform(collab_franka1.world_to_starteegoal))
            p3x_franka2.set_world_pose(position=ft.position_from_transform(collab_franka2.world_to_starteegoal) , orientation=ft.quat_from_transform(collab_franka2.world_to_starteegoal))
            collab_isaacsim.step_physics_and_render(10)
            
            world_to_eegoal_1 = ft.transform_from_pq(p=p3x.get_world_pose()[0], quat=p3x.get_world_pose()[1]) #ee_goal pose
            eegoal_Pose_in_robotbase_1 = collab_franka1.update_solver_ee_goal_from_teleop_widget(world_to_eegoal_1, collab_franka1.robotbase_to_world, p4x)
            
            world_to_eegoal_2 = ft.transform_from_pq(p=p3x_franka2.get_world_pose()[0], quat=p3x_franka2.get_world_pose()[1]) #ee_goal pose
            eegoal_Pose_in_robotbase_2 = collab_franka2.update_solver_ee_goal_from_teleop_widget(world_to_eegoal_2, collab_franka2.robotbase_to_world, p4x_franka2)
            #Get Franka on default position - not included in data saving:
            joint_commands_usd_franka1 = collab_franka1.curobomanager.compute_ik(eegoal_Pose_in_robotbase_1)
            joint_commands_usd_franka2 = collab_franka2.curobomanager.compute_ik(eegoal_Pose_in_robotbase_2)

            collab_franka1.reset_robot_states_to_pose(joint_commands_usd_franka1)
            collab_franka2.reset_robot_states_to_pose(joint_commands_usd_franka2)
            collab_isaacsim.step_physics_and_render(100)
            ###########################################
            # #Initialize mpc buffer, which needs an ee_goal_pose and current state
            eegoal_Pose_in_robotbase_1 = Pose(position=tensor_args.to_device(start_ee_goal_p_in_robotbase_1), quaternion=tensor_args.to_device(start_ee_goal_quat_in_robotbase_1))
            eegoal_Pose_in_robotbase_2 = Pose(position=tensor_args.to_device(start_ee_goal_p_in_robotbase_2), quaternion=tensor_args.to_device(start_ee_goal_quat_in_robotbase_2))
            
            collab_franka1.curobomanager.initialize_mpc_buffer(eegoal_Pose_in_robotbase_1)
            collab_franka2.curobomanager.initialize_mpc_buffer(eegoal_Pose_in_robotbase_2)
            run_sim_to_first_ee_goal = False 
            continue

        ######################################################################################
        # Compute world_to_eegoal:

        if args.run_vr:
            # Update pose of the frame-prims following the vr controllers:
            vr_world_1.setpose_vrcontrollersfollower_frames()
        # Update ee goal (p3x has been updated by vr_world_1 internally, or by manual teleop)
        world_to_eegoal_1 = ft.transform_from_pq(p=p3x.get_world_pose()[0], quat=p3x.get_world_pose()[1]) #ee_goal pose
        world_to_eegoal_2 = ft.transform_from_pq(p=p3x_franka2.get_world_pose()[0], quat=p3x_franka2.get_world_pose()[1]) #ee_goal pose
        
        ######################################################################################
        # Compute eegoal_Pose_in_robotbase for planner solver:
        eegoal_Pose_in_robotbase_1 = collab_franka1.update_solver_ee_goal_from_teleop_widget(world_to_eegoal_1, collab_franka1.robotbase_to_world, p4x)
        eegoal_Pose_in_robotbase_2 = collab_franka1.update_solver_ee_goal_from_teleop_widget(world_to_eegoal_2, collab_franka2.robotbase_to_world, p4x_franka2)
        
        ######################################################################################
        joint_commands_usd_franka1 = []
        joint_commands_usd_franka2 = []

        collab_franka1.curobomanager.step_MPC (collab_franka1.world_to_robotbase, eegoal_Pose_in_robotbase_1)
        collab_franka2.curobomanager.step_MPC (collab_franka2.world_to_robotbase, eegoal_Pose_in_robotbase_2)

        if not args.run_vr:
            collab_isaacsim.draw_points(collab_franka1.curobomanager.mpc_solver.get_visual_rollouts(), collab_franka1.world_to_robotbase)
            collab_isaacsim.draw_points(collab_franka2.curobomanager.mpc_solver.get_visual_rollouts(), collab_franka2.world_to_robotbase)

        joint_commands_usd_franka1.append(collab_franka1.curobomanager.mpc_result.js_action.position.cpu().numpy()) #only one command
        joint_commands_usd_franka2.append(collab_franka2.curobomanager.mpc_result.js_action.position.cpu().numpy()) #only one command

        # for waypoint in joint_commands_usd_franka1: #expect one waypoint for ik or MPC, multiple for motion_gen
        articulation_action_cu_1 = ArticulationAction(joint_positions=joint_commands_usd_franka1[0])
        articulation_action_cu_1.joint_indices = [0, 1, 2, 3, 4, 5, 6]
        articulation_action_cu_2 = ArticulationAction(joint_positions=joint_commands_usd_franka2[0])
        articulation_action_cu_2.joint_indices = [0, 1, 2, 3, 4, 5, 6]

        articulation_controller_1.apply_action(articulation_action_cu_1) # command the robot
        articulation_controller_2.apply_action(articulation_action_cu_2) # command the robot

        if args.log_data:
            sim_data_log.append_states_this_sim_step()  

           
############################################################

if __name__ == "__main__":
    main()
    simulation_app.close()


# curobo:
# claudiap/jointlimitPhysXfix Change joint limit in continuous to revolute joint conversion - PhysX supports -2Pi,2Pi