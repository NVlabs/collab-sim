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

"""Single Franka robot MPC teleoperation example.

Pre-requisites:
- SteamVR connection to VR headset running (for VR mode)
- collab-sim in PYTHONPATH
- Conda environment activated: conda activate collab-sim

Usage:
    VR mode:
        python franka_mpc_example.py --run_vr --log_data --relevant_objects_str Cube P3 P4
            --enable omni.kit.xr.profile.vr --enable isaacsim.xr.openxr

    Non-VR mode:
        python franka_mpc_example.py --use_keyboard --log_data --relevant_objects_str P3 P4
"""

import argparse
# Configuration (used when run with no arguments, for debugging)
debug_run_vr = False
debug_print_debug = True
debug_use_keyboard = True 
debug_log_data = False
debug_relevant_objects_str = ["Cube", "P3", "P4"]
load_scene_usd = False  # Only for development

# Parse command-line arguments (override debug config if provided)
parser = argparse.ArgumentParser()
parser.add_argument("--run_vr", action='store_true', default=debug_run_vr, help="Enable VR mode (default: debug)")
parser.add_argument("--print_debug", action='store_true', default=debug_print_debug, help="Enable debug printing (default: debug)")
parser.add_argument("--use_keyboard", action='store_true', default=debug_use_keyboard, help="Enable keyboard open/close Franka gripper (default: debug)")
parser.add_argument("--log_data", action='store_true', default=debug_log_data, help="Enable data logging (default: debug)")
parser.add_argument(
    "--relevant_objects_str",
    nargs='+',
    default=debug_relevant_objects_str,
    help="Object name patterns to save (default: ['Cube', 'P3', 'P4'])"
)

args, unknown_args = parser.parse_known_args()

if args.print_debug:
    print(args)

# External dependencies
import math
import os
import time

import numpy as np
import torch

# CuRobo
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml

tensor_args = TensorDeviceType()

# Isaac Sim
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.robot.manipulators.examples.franka import KinematicsSolver
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat

# Local
from collab_sim import collab_robot_controller, collab_teleop_utils

# Directory paths
COLLAB_DIR = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..'))
DEMOS_VR_DIR = os.path.join(COLLAB_DIR, "data/demosVR")
DEMOSUSD_DIR = os.path.join(COLLAB_DIR, "data/demosUSD")
SCENEUSD_DIR = os.path.join(COLLAB_DIR, "data/sceneUSD")
DATA_DIR = os.path.join(COLLAB_DIR, "collab_sim/data")

# Simulation step constants
INITIAL_RENDER_STEPS = 1000  # Time for user to adjust GUI
RESET_RENDER_STEPS = 20
ROBOT_SETTLE_STEPS = 100

# Initial joint configuration for robot startup
start_js = {
    "panda_joint1": 0.15896346,
    "panda_joint2": -0.07731238,
    "panda_joint3": 0.03789043,
    "panda_joint4": -2.2444482,
    "panda_joint5": 0.00352372,
    "panda_joint6": 2.167158,
    "panda_joint7": -2.1614377,
    "panda_finger_joint1": 0.0,
    "panda_finger_joint2": 0.0,
}

# Scene poses and ranges
p_large_scene = [-5.5, 0.0, 0.0]
p_table = [0.0, 0.0, 0.0]
robot_origin_p_in_world = np.array(p_table) + np.array([-0.25, 0.0, 0.82])
p_cubes_min = robot_origin_p_in_world + np.array([0.30, -0.44, 0.02])
p_cubes_max = robot_origin_p_in_world + np.array([0.60, 0.44, 0.02])
cube_xyz_ranges = [
    [p_cubes_min[0], p_cubes_max[0], p_cubes_min[1], p_cubes_max[1], p_cubes_min[2], p_cubes_max[2] ],
    [p_cubes_min[0], p_cubes_max[0], p_cubes_min[1], p_cubes_max[1], p_cubes_min[2], p_cubes_max[2] ],
    [p_cubes_min[0], p_cubes_max[0], p_cubes_min[1], p_cubes_max[1], p_cubes_min[2], p_cubes_max[2] ],
    [p_cubes_min[0], p_cubes_max[0], p_cubes_min[1], p_cubes_max[1], p_cubes_min[2], p_cubes_max[2] ]
]
p_worker = np.array(p_table) + np.array([3.64, -6.50, 0.0])

# Robot origin and initial EE goal
robot_origin_quat_in_world = euler_angles_to_quat([0.0, 0.0, 0.0], degrees=True)
start_ee_goal_p_in_robotbase = np.array([0.3, 0.1, 0.4])
start_ee_goal_quat_in_robotbase = euler_angles_to_quat([0, -180, 0], degrees=True)

# CuRobo IK configuration
config_file = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
urdf_file = config_file["kinematics"]["urdf_path"] 
base_link = config_file["kinematics"]["base_link"]
ee_link = config_file["kinematics"]["ee_link"]
robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)

# Initialize simulation
my_world = World(stage_units_in_meters=1.0)
collab_isaacsim = collab_teleop_utils.IsaacSimUtils(world=my_world)
ft = collab_teleop_utils.FramesTransforms()

collab_isaacsim.add_lights()
collab_isaacsim.reset_world_set_default_prim()

# Add Franka robot
my_franka, my_controller, articulation_controller = collab_isaacsim.load_franka(
    world_to_robotbase=ft.transform_from_pq(robot_origin_p_in_world, robot_origin_quat_in_world),
    franka_name='Franka',
    prim_path='/World/robot'
)

collab_franka1 = collab_robot_controller.ActiveRobot(my_franka,
                                                      articulation_controller,
                                                      robot_origin_p_in_world,
                                                      robot_origin_quat_in_world,
                                                      start_ee_goal_p_in_robotbase,
                                                      start_ee_goal_quat_in_robotbase)
collab_franka1.init_curobo_manager(robot_cfg)

# Load workshop table
collab_isaacsim.load_workshop_table(p_table)

if load_scene_usd:  # Dev-only
    k_path = os.path.join(SCENEUSD_DIR, 'Collected_warehouse/warehouse.usd')
    k_prim_path = '/World/collab_largescene'
    k_prim_name = "collab_largescene"
    pos = np.array(p_large_scene)
    quat = [math.cos(3.14 / 2), 0, 0, math.sin(3.14 / 2)]
    k_prim, k_xform_prim = collab_isaacsim.load_usd(k_path, k_prim_path, k_prim_name, pos, quat, scale=1)

    worker_usd = os.path.join(SCENEUSD_DIR, 'Collected_full_warehouse_worker_and_anim_cameras/worker.usd')
    w_prim, w_xform_prim = collab_isaacsim.load_usd(worker_usd, '/World/collab_worker', "collab_worker", p_worker, [1, 0, 0, 0], scale=1)
else:
    collab_isaacsim.load_default_plane()

collab_isaacsim.load_random_cubes(cube_xyz_ranges)

# Setup keyboard control if requested
if args.use_keyboard:
    keyboard = collab_teleop_utils.KeyboardTeleopDebug(my_franka)

# Create prims for teleop goal management

p4x = collab_isaacsim.create_visual_frame(
    "P4", "/World/DebugFrames/P4",
    position=ft.position_from_transform(collab_franka1.world_to_starteegoal),
    orientation=ft.quat_from_transform(collab_franka1.world_to_starteegoal)
)

p3x = collab_isaacsim.create_visual_frame(
    "P3", "/World/DebugFrames/P3",
    position=ft.position_from_transform(collab_franka1.world_to_starteegoal),
    orientation=ft.quat_from_transform(collab_franka1.world_to_starteegoal)
)    

if args.run_vr:
    from collab_sim import collab_vrteleop
    VRworld = collab_vrteleop.VRTeleop(world=my_world)
    while not VRworld.is_vr_initialized():
        print("Waiting for VR to start...")
        my_world.step(render=True)
    VRworld.set_up_vr_devices_with_active_vr()
    VRworld.set_up_vr_teleop_frames(robot=my_franka, eegoalprim=p3x)
    VRworld.init_vr_leftcont_buttons_righthanded_teleop_default()
    VRworld.init_vr_rightcont_buttons_righthanded_teleop_default()

    # Set callback to move robot back to table position after VR controller reset
    def on_vr_robot_reset():
        collab_isaacsim.move_robot_to_root_transform(
            my_franka,
            world_to_robotbase=ft.transform_from_pq(robot_origin_p_in_world, robot_origin_quat_in_world)
        )
    VRworld.set_robot_reset_callback(on_vr_robot_reset)

# Reset world and configure

collab_isaacsim.reset_world()
collab_isaacsim.move_robot_to_root_transform(
    my_franka,
    world_to_robotbase=ft.transform_from_pq(robot_origin_p_in_world, robot_origin_quat_in_world)
)

collab_isaacsim.set_solver_TGS()

if args.log_data:
    sim_data_log = collab_teleop_utils.SimDataLog(args.relevant_objects_str, my_world)
    sim_data_log.save_world_usd(my_world)


def main():
    """Main simulation loop."""
    collab_franka1.reset_robot_states_to_pose(start_js)
    collab_isaacsim.step_physics_and_render(ROBOT_SETTLE_STEPS)

    # Initialize MPC buffer
    eegoal_Pose_in_robotbase = Pose(
        position=tensor_args.to_device(start_ee_goal_p_in_robotbase),
        quaternion=tensor_args.to_device(start_ee_goal_quat_in_robotbase)
    )
    collab_franka1.curobo_manager.initialize_mpc_buffer(eegoal_Pose_in_robotbase)
    run_sim_to_first_ee_goal = True
    collab_isaacsim.step_render(INITIAL_RENDER_STEPS)

    # Main simulation loop
    while simulation_app.is_running():
        collab_isaacsim.step_physics_and_render(1) #steps physics if my_world.is_playing, renders either play/stop to keep gui interactive

        if my_world.is_stopped(): #reset after stopping sim on the gui
            print("my_world.reset()")
            collab_isaacsim.reset_world()
            collab_isaacsim.move_robot_to_root_transform(
                my_franka,
                world_to_robotbase=ft.transform_from_pq(robot_origin_p_in_world, robot_origin_quat_in_world)
            )

        if my_world.current_time_step_index == 2: #==2 after a world.reset
            if args.log_data: # and data_dict: #save after world.reset (from vr controller button callback)
                sim_data_log.proccess_and_save_data(my_world.get_physics_dt())

            collab_isaacsim.set_solver_TGS()
            collab_isaacsim.step_physics_and_render(1)
            collab_isaacsim.step_render(RESET_RENDER_STEPS)
            collab_isaacsim.reshuffle_cubes(cube_xyz_ranges)
            # VRworld.teleport_headset_to_start()
            run_sim_to_first_ee_goal = True # script will continue to compute first franka joint angles for initial p3x world_to_starteegoal and step sim to it

        if run_sim_to_first_ee_goal: #step sim to get robot from init USD state to the initial pose (p3x ee goal)
            p3x.set_world_pose(
                position=ft.position_from_transform(collab_franka1.world_to_starteegoal),
                orientation=ft.quat_from_transform(collab_franka1.world_to_starteegoal)
            )
            collab_isaacsim.step_physics_and_render(10)
            world_to_eegoal = ft.transform_from_pq(p=p3x.get_world_pose()[0], quat=p3x.get_world_pose()[1])
            eegoal_Pose_in_robotbase = collab_franka1.update_solver_ee_goal_from_teleop_widget(
                world_to_eegoal, collab_franka1.robotbase_to_world, p4x
            )
            #Get Franka on default position - not included in data saving:
            joint_commands_usd = collab_franka1.curobo_manager.compute_ik(eegoal_Pose_in_robotbase)
            collab_franka1.reset_robot_states_to_pose(joint_commands_usd)
            collab_isaacsim.step_physics_and_render(ROBOT_SETTLE_STEPS)

            eegoal_Pose_in_robotbase = Pose(
                position=tensor_args.to_device(start_ee_goal_p_in_robotbase),
                quaternion=tensor_args.to_device(start_ee_goal_quat_in_robotbase)
            )
            collab_franka1.curobo_manager.initialize_mpc_buffer(eegoal_Pose_in_robotbase)
            run_sim_to_first_ee_goal = False
            continue

        ######################################################################################
        # Compute world_to_eegoal:

        if args.run_vr and VRworld.is_vr_initialized():
            # Update pose of the frame-prims following the vr controllers:
            # print("updating vr controller follower frames")
            VRworld.setpose_vrcontrollersfollower_frames()
            VRworld.left_trigger_button_manager.update()
            VRworld.left_squeeze_button_manager.update()
            VRworld.right_trigger_button_manager.update()
            VRworld.right_squeeze_button_manager.update()
        # Update ee goal (p3x has been updated by VRworld internally, or by manual teleop)
        world_to_eegoal = ft.transform_from_pq(p=p3x.get_world_pose()[0], quat=p3x.get_world_pose()[1])
        
        # Compute eegoal_Pose_in_robotbase for planner solver:
        eegoal_Pose_in_robotbase = collab_franka1.update_solver_ee_goal_from_teleop_widget(
            world_to_eegoal, collab_franka1.robotbase_to_world, p4x
        )

        # Execute MPC step and command robot
        joint_commands_usd = []

        try:
            collab_franka1.curobo_manager.step_MPC(collab_franka1.world_to_robotbase, eegoal_Pose_in_robotbase)

            if not args.run_vr:
                collab_isaacsim.draw_points(
                    collab_franka1.curobo_manager.mpc_solver.get_visual_rollouts(),
                    collab_franka1.world_to_robotbase
                )

            joint_commands_usd.append(collab_franka1.curobo_manager.mpc_result.js_action.position.cpu().numpy())

            for waypoint in joint_commands_usd: #expect one waypoint for ik or MPC, multiple for motion_gen
                articulation_action_cu = ArticulationAction(joint_positions=waypoint)
                articulation_action_cu.joint_indices = [0, 1, 2, 3, 4, 5, 6]
                # command the robot:
                articulation_controller.apply_action(articulation_action_cu)

                if args.log_data:
                    sim_data_log.append_states_this_sim_step()

        except Exception as e:
            print(f"step_MPC and robot command failed with error: {e}. Retrying...")



if __name__ == "__main__":
    main()
    simulation_app.close()