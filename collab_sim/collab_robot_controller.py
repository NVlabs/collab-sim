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



# Standard library
import os

# Third-party
import numpy as np
import torch

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig

# Isaac Sim
from isaacsim.core.utils.types import ArticulationAction

# Local
from collab_sim import collab_teleop_utils

tensor_args = TensorDeviceType()
ft = collab_teleop_utils.FramesTransforms()

# Constants for Franka robot configuration
FRANKA_DOF = 7
FRANKA_JOINT_GAINS = np.array([100000000, 6000000., 10000000, 600000., 25000., 15000., 50000., 6000., 6000.])
FRANKA_MAX_EFFORTS = np.array([100000, 52.199997, 100000, 52.199997, 7.2, 7.2, 7.2, 50., 50])
FRANKA_VELOCITY_ITERATIONS = 4
FRANKA_POSITION_ITERATIONS = 124
FRANKA_EE_TO_HAND_OFFSET = np.array([0.0, 0.0, -0.1])

# IK solver configuration
IK_ROTATION_THRESHOLD = 0.05
IK_POSITION_THRESHOLD = 0.005
IK_NUM_SEEDS = 20

# MPC solver configuration
MPC_STEP_DT = 0.03
MPC_SHIFT_STEPS = 1
MPC_MAX_ATTEMPTS = 2


class ActiveRobot():

    def __init__(self,
                 robot,
                 articulation_controller,
                 robot_origin_p_in_world,
                 robot_origin_quat_in_world,
                 start_ee_goal_p_in_robotbase,
                 start_ee_goal_quat_in_robotbase) -> None:
        self.robot = robot
        self.articulation_controller = articulation_controller
        self.robot_origin_p_in_world = robot_origin_p_in_world
        self.robot_origin_quat_in_world = robot_origin_quat_in_world
        self.start_ee_goal_p_in_robotbase = start_ee_goal_p_in_robotbase
        self.start_ee_goal_quat_in_robotbase = start_ee_goal_quat_in_robotbase

        # Compute transform chains
        self.robotbase_to_starteegoal = ft.transform_from_pq(
            self.start_ee_goal_p_in_robotbase, self.start_ee_goal_quat_in_robotbase
        )
        self.world_to_robotbase = ft.transform_from_pq(
            self.robot_origin_p_in_world, self.robot_origin_quat_in_world
        )
        self.robotbase_to_world = ft.invert_homogeneous_transform(self.world_to_robotbase)
        self.world_to_starteegoal = ft.concatenate_transforms(
            self.world_to_robotbase, self.robotbase_to_starteegoal
        )


    def franka_setup_gains_reset(self):
        """Configure Franka robot gains and solver parameters."""
        self.articulation_controller.set_gains(kps=FRANKA_JOINT_GAINS)
        self.articulation_controller.set_max_efforts(values=FRANKA_MAX_EFFORTS)
        self.robot.set_solver_velocity_iteration_count(FRANKA_VELOCITY_ITERATIONS)
        self.robot.set_solver_position_iteration_count(FRANKA_POSITION_ITERATIONS)


    def init_curobo_manager(self, robot_cfg):
        self.robot_cfg = robot_cfg
        self.curobo_manager = CuroboManager(self.robot_cfg)
        self.curobo_manager.set_robot(self.robot)


    def reset_robot_states_to_pose(self, base_pose):
        if isinstance(base_pose, dict):
            joint_commands_usd = list(base_pose.values())[0:7]
        else:
            joint_commands_usd = base_pose
        articulation_action = ArticulationAction(joint_positions=joint_commands_usd)
        articulation_action.joint_indices = [0, 1, 2, 3, 4, 5, 6]
        self.articulation_controller.apply_action(articulation_action)
        self.robot.gripper.open()

    def update_solver_ee_goal_from_teleop_widget(self, world_to_eegoal, robotbase_to_world, ee_planner_frame):
        """
        Takes the current ee goal (teleop wiget) in robotbase_to_eegoal
        Transform to planning frame
        Using eegoal_to_pandahand, specific to Franka
        Updates the pose of the prim used as planning frame
        """
        # world_to_eegoal is the teleop widget frame 
        robotbase_to_eegoal = ft.concatenate_transforms(robotbase_to_world, world_to_eegoal)
        eegoal_to_pandahand = ft.transform_from_pq(
            p=FRANKA_EE_TO_HAND_OFFSET,
            quat=np.array([0.0, 0.0, 0.0, 1.0])
        )
        robotbase_to_pandahand = ft.concatenate_transforms(robotbase_to_eegoal, eegoal_to_pandahand)

        solver_ee_goal_p = ft.position_from_transform(robotbase_to_pandahand)
        solver_ee_goal_quat = ft.quat_from_transform(robotbase_to_pandahand)

        # Update visualization frame
        world_to_pandahand = ft.concatenate_transforms(world_to_eegoal, eegoal_to_pandahand)
        ee_planner_frame.set_world_pose(
            ft.position_from_transform(world_to_pandahand),
            ft.quat_from_transform(world_to_pandahand)
        )

        eegoal_Pose_in_robotbase = Pose(
            position=tensor_args.to_device(solver_ee_goal_p),
            quaternion=tensor_args.to_device(solver_ee_goal_quat)
        )
        return eegoal_Pose_in_robotbase


class CuroboManager():

    def __init__(self, robot_cfg) -> None:

        ik_config = IKSolverConfig.load_from_robot_config(
                            robot_cfg,
                            None,
                            rotation_threshold=IK_ROTATION_THRESHOLD,
                            position_threshold=IK_POSITION_THRESHOLD,
                            num_seeds=IK_NUM_SEEDS,
                            self_collision_check=False,
                            self_collision_opt=False,
                            tensor_args=tensor_args,
                            use_cuda_graph=True,
                        )
        self.ik_solver_curobo = IKSolver(ik_config)
        print("iksolver created with active joints ", self.ik_solver_curobo.dof)

        # MPC solver:
        mpc_config = MpcSolverConfig.load_from_robot_config(
                            robot_cfg,
                            None,
                            use_cuda_graph=True,
                            use_cuda_graph_metrics=True,
                            use_cuda_graph_full_step=False,
                            use_lbfgs=False,
                            use_es=False,
                            use_mppi=True,
                            store_rollouts=True,
                            step_dt=MPC_STEP_DT,
                        )
        self.mpc_solver = MpcSolver(mpc_config)
                            
    def set_robot(self, robot):
        self.robot = robot

    def compute_ik(self, eegoal_Pose_in_robotbase):
        """Compute inverse kinematics for the given end-effector goal pose."""
        cu_ik_result = self.ik_solver_curobo.solve_single(
            eegoal_Pose_in_robotbase,
            retract_config=self.retract_config,
            seed_config=self.seed_config
        )
        joint_commands_usd = cu_ik_result.js_solution.position.cpu().numpy()[0][0]
        return joint_commands_usd

    def get_configs(self):
        """Get current robot joint configuration."""
        values_usd = self.robot.get_joint_positions()
        values_urdf = values_usd[0:FRANKA_DOF]
        ndof = len(values_urdf)
        active_joints = self.robot.dof_names[0:FRANKA_DOF]

        cu_js_urdfformat = JointState(
            position=tensor_args.to_device(values_urdf),
            velocity=tensor_args.to_device(np.zeros(ndof)),
            acceleration=tensor_args.to_device(np.zeros(ndof)),
            jerk=tensor_args.to_device(np.zeros(ndof)),
            joint_names=active_joints,
        )
        self.retract_config = cu_js_urdfformat.position.view(-1, ndof)
        self.seed_config = cu_js_urdfformat.position.view(-1, 1, ndof)
        return cu_js_urdfformat

    def initialize_mpc_buffer(self, eegoal_Pose_in_robotbase):
        """Initialize MPC buffer with current state and goal pose ee_goal_pose."""
        cu_js_urdfformat = self.get_configs()
        self.goal_mpc = Goal(
            current_state=cu_js_urdfformat,
            goal_state=cu_js_urdfformat,
            goal_pose=eegoal_Pose_in_robotbase
        )
        self.goal_mpc_buffer = self.mpc_solver.setup_solve_single(self.goal_mpc, 1)
        self.mpc_solver.update_goal(self.goal_mpc_buffer)

    def step_MPC(self, world_to_robotbase, eegoal_Pose_in_robotbase):
        """Execute one step of Model Predictive Control."""
        cu_js_urdfformat = self.get_configs()
        self.goal_mpc_buffer.goal_pose.copy_(eegoal_Pose_in_robotbase)
        self.mpc_solver.update_goal(self.goal_mpc_buffer)
        self.mpc_result = self.mpc_solver.step(cu_js_urdfformat, shift_steps=MPC_SHIFT_STEPS, max_attempts=MPC_MAX_ATTEMPTS)

