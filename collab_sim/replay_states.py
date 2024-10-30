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
This script reads all the demo pkl files in demo_session_path and replays 
demos (state replay) in Isaac Sim
"""
############################################################
# EDIT config:
# demo dir in DEMOS_VR_DIR:
demo_session_dirname = "output_demos_vr_example" #one franka vr teleop example data
# demo_session_dirname = "output_2024_10_20_18_32_39_demo7_onefranka_resets"
sample_states = 1
steps_per_state = 1 #render steps ran for each new state
############################################################


############################################################
# external:
import argparse
import pickle
import os
import numpy as np
############################################################ 
COLLAB_DIR = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
DEMOS_VR_DIR = os.path.join(COLLAB_DIR, "data/demosVR") #dir for saving demo data for reply
DEMOSUSD_DIR = os.path.join(COLLAB_DIR, "data/demosUSD") #
SCENEUSD_DIR = os.path.join(COLLAB_DIR, "data/sceneUSD") #example USD envs
DATA_DIR = os.path.join(COLLAB_DIR, "collab_sim/data") # axis usd
############################################################
# ArgumentParser must be created before SimulationApp
demo_session_path = os.path.join(DEMOS_VR_DIR, demo_session_dirname)
PARSER = argparse.ArgumentParser("Replay teleoperation data")
PARSER.add_argument("--demo_session_path", type=str, default=demo_session_path, help="Path to demo session")
PARSER.add_argument("--steps_per_state", type=str, default=steps_per_state, help="The number of simulation steps per target state")
PARSER.add_argument("--sample_states", type=str, default=sample_states, help="Subsampling steps")
ARGS = PARSER.parse_args()
############################################################
# # adding args here for ease of debug:
# ARGS.scene_path = "/home/cpda/dev/collab-sim/data/demosUSD/demoUSD_stack6blocks/scene_demo.usda"
# # ARGS.scene_path  = 'C:/Users/C/dev/collab-sim/data/sceneUSD/sceneUSD_stack6blocks/scene.usda'
# ARGS.states_path = "/home/cpda/dev/collab-sim/data/demosVR/output_2023_09_07_16_56_06/states_2023_09_07_16_57_45.pkl"

ARGS.scene_path =  ARGS.demo_session_path + "/scene_demo.usda" # dir with demo scene USD
ARGS.states_path = ARGS.demo_session_path # dir with all demo files .pkl



from omni.isaac.kit import SimulationApp

APP = SimulationApp({"headless": False})
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
import omni.usd
############################################################
# collab-sim:
from collab_sim import collab_teleop_utils
############################################################


def main():
    # Replay all demo files in folder demo_session_path:

    # Load scene
    my_world = World(stage_units_in_meters=1.0)
    isaacsim_utils = collab_teleop_utils.IsaacSimUtils(world=my_world)
    ft = collab_teleop_utils.FramesTransforms()

    isaacsim_utils.reset_world_visual(1)
    world_prim = add_reference_to_stage(usd_path=ARGS.scene_path, prim_path="/World")
    my_world.stage.SetDefaultPrim(world_prim)
    isaacsim_utils.reset_world_visual(100)

    stage = omni.usd.get_context().get_stage()

    robot_prim_str_list = ["robot", "robot1", "robot2"]
    for robot_prim_str in robot_prim_str_list:
        robot_prim = isaacsim_utils.find_prim_by_name(stage, prim_name=robot_prim_str)
        if robot_prim:
            isaacsim_utils.disable_collisions(robot_prim)

    demos_data = collab_teleop_utils.SimDataLog.load_demos_data(ARGS.states_path)
    isaacsim_utils.reset_world_visual(100)

    # input("Press any key to continue...")

    for this_demo_data in demos_data:
        # example: this_demo_data["states_list"][0]["/World/robot"]["joint_state"]["position"]
        states = this_demo_data.get('states_list', [])
        sampled_states = states[: -1 : ARGS.sample_states] + states[-1:]
        print(f"Sampled {len(sampled_states)}/{len(states)} states")
        isaacsim_utils.animate_states(my_world, sampled_states, ARGS.steps_per_state)

    print('All demos replayed. Now simulating indefinitely...')
    while APP.is_running():
        isaacsim_utils.step_render(10)

############################################################

if __name__ == "__main__":
    main()
    APP.close()
