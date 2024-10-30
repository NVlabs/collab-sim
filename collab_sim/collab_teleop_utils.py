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


# pip install transforms3d



############################################################
import carb
import numpy as np
import torch
import random
from typing import Optional
from typing import List
from typing import Any
import os
############################################################
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.prims.xform_prim import XFormPrim
from scipy.spatial.transform import Rotation 
from omni.isaac.core.objects import DynamicCuboid, VisualCuboid
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.prims.rigid_prim import RigidPrim
from pxr import Usd, UsdGeom, Gf, UsdPhysics, Sdf, UsdLux
from omni.isaac.core.utils.prims import add_reference_to_stage
from omni.isaac.core.prims.geometry_prim import GeometryPrim
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.franka.franka import Franka
from omni.isaac.franka import KinematicsSolver
from omni.isaac.debug_draw import _debug_draw
############################################################ 
COLLAB_DIR = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
DEMOS_VR_DIR = os.path.join(COLLAB_DIR, "data/demosVR") #dir for saving demo data for reply
DEMOSUSD_DIR = os.path.join(COLLAB_DIR, "data/demosUSD") #
SCENEUSD_DIR = os.path.join(COLLAB_DIR, "data/sceneUSD") #example USD envs
DATA_DIR = os.path.join(COLLAB_DIR, "collab_sim/data") # axis usd
############################################################


class IsaacSimUtils(object):

    def __init__(self, world, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0) -> None:
        print("Isaac Sim Utils")
        self.world = world  
        self.stage = self.world.stage
        self.ft = FramesTransforms()
        self.physics_dt = physics_dt
        self.rendering_dt = rendering_dt

        #scene manager:
        self.nucleus_server = get_assets_root_path()

    def step_render (self, n = 1):
        for i in range (n):
            self.world.render()

    def step_physics (self, n = 1):
        for i in range(n):
            self.world.step(render=False) 
            # step only steps physics if my_world.is_playing
            # if world is stopped, step only renders and keeps the gui interactive

    def step_physics_and_render (self, n=1):
        for i in range(n):
            self.world.step(render=True)

    def reset_world_visual(self, render_steps):
        self.world.reset()
        self.step_render(render_steps)

    def reset_world(self):
        self.world.reset()
        self.config_simulation_physics()

    def set_solver_TGS(self):
        self.world._physics_context.set_solver_type("TGS")

    def reset_world_set_default_prim(self):
        self.world.reset()
        self.config_simulation_physics()
        self.stage.SetDefaultPrim(self.stage.GetPrimAtPath("/World"))

    def reset_world_and_init_USD(self):
        self.world.reset() # leaves my_world.current_time_step_index = 2
        self.config_simulation_physics()
        # print (self.world.current_time_step_index)
        # reset USD:
        self.step_physics_and_render (1) #one physics step is needed for world to go to USD init state
        self.step_render(8) #it takes around 6 render steps to get the stage visually to the USD init state

    def config_simulation_physics(self):
        # physics_context=my_world.get_physics_context()
        self.world.set_simulation_dt(self.physics_dt, self.rendering_dt) #set_simulation_dt in simulation_context
        self.world.get_physics_dt()
        # physics_context.set_physics_dt(dt= 1.0 / 60.0, substeps = 1) 

    def add_lights(self):
        light_prim = UsdLux.DistantLight.Define(self.stage, Sdf.Path("/World/defaultLight"))
        light_prim.CreateIntensityAttr(2000)

    def create_target_cube (self, target_name, prim_path, position, orientation, color, scale=np.array([0.03, 0.03, 0.03])):
        """
        Adds target non-physics cube:
        """
        self.world.scene.add(
            VisualCuboid(
                name=target_name,
                prim_path= prim_path, #'/World/TargetCube',
                position=position,
                orientation=orientation,
                color=color,
                size=1.0,
                scale=scale / get_stage_units(),
            )
        )

    def create_tiny_visual_cube (self, name: str, prim_path: str, 
                                 position: List[float], orientation: List[float]) -> None:
        """
        Adds non-physics cube, very small to just use it for the frame display
        position: List[float]  
        orientation: List[float] Quaternion wxyz
        """
        self.world.scene.add(
            VisualCuboid(
                name=name,
                prim_path= prim_path, #'/World/TargetCube',
                position=position,
                orientation=orientation,
                color=np.array([0, 0, 1]),
                size=1.0,
                scale=np.array([0.01,0.01,0.01]) / get_stage_units(),
            )
        )


    def prim_to_xformable (self, prim, name, scale):
        xformable = XFormPrim(str(prim.GetPath()), name)  
        xformable.set_local_scale(scale*np.ones(3))
        return xformable
    
    def create_visual_frame(self, name, prim_path="/World/", 
                            position: Any = np.array([0.0, 0.0, 0.0]), 
                            orientation: Any = np.array([1.0, 0.0, 0.0, 0.0]),
                            scale=0.4) -> XFormPrim:
        """ 
        Creates a visual xyz frame
        """
        usd_path = os.path.join(DATA_DIR, "axis.usda")
        prim = add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        prim.CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
        xf = self.prim_to_xformable(prim, name, scale)
        xf.set_world_pose(position, orientation)
        return xf


    def draw_points(self, rollouts: torch.Tensor, world_to_robotbase): 
        # modified from cuobo/cumotion: added world_to_robotbase transform
        if rollouts is None:
            return
        draw = _debug_draw.acquire_debug_draw_interface()
        N = 100
        # if draw.get_num_points() > 0:
        draw.clear_points()
        cpu_rollouts = rollouts.cpu().numpy()
        b, h, _ = cpu_rollouts.shape
        point_list = []
        colors = []
        for i in range(b):
            # get list of points:
            point_list += [
                (cpu_rollouts[i, j, 0], cpu_rollouts[i, j, 1], cpu_rollouts[i, j, 2]) for j in range(h)
            ]
            colors += [(1.0 - (i + 1.0 / b), 0.3 * (i + 1.0 / b), 0.0, 0.1) for _ in range(h)]
        sizes = [10.0 for _ in range(b * h)]

        point_list_in_world = []
        for point in point_list:
            robotbase_to_points = self.ft.transform_from_pq(p=point)
            world_to_point = self.ft.concatenate_transforms(world_to_robotbase,  robotbase_to_points)
            point_list_in_world.append(self.ft.position_from_transform(world_to_point))
        draw.draw_points(point_list_in_world, colors, sizes)


    def load_workshop_table(self, p=[-2.75, 6.7, 0.0]): # From Nucleus
        p = np.array(p)
        # Assets from FrankaNutBolt:
        self.asset_folder = self.nucleus_server + "/Isaac/Samples/Examples/FrankaNutBolt/"
        self.asset_paths = {
            "shop_table": self.asset_folder + "SubUSDs/Shop_Table/Shop_Table.usd",
            "tooling_plate": self.asset_folder + "SubUSDs/Tooling_Plate/Tooling_Plate.usd",
            "nut": self.asset_folder + "SubUSDs/Nut/M20_Nut_Tight_R256_Franka_SI.usd",
            "bolt": self.asset_folder + "SubUSDs/Bolt/M20_Bolt_Tight_R512_Franka_SI.usd",
            "vibra_table_top": self.asset_folder + "SubUSDs/VibrationTable_Top/VibrationTable_Top.usd",
            "vibra_table_bot": self.asset_folder + "SubUSDs/VibrationTable_Base/VibrationTable_Base.usd",
            "vibra_table_collision": self.asset_folder + "SubUSDs/VibrationTable_Top_collision.usd",
            "vibra_table_clamps": self.asset_folder + "SubUSDs/Clamps/Clamps.usd",
            "pipe": self.asset_folder + "SubUSDs/Pipe/Pipe.usd",
        }

        self.world.scene.add(XFormPrim(prim_path="/World/collisionGroupsTable", name="collision_groups_xform_table"))
        # self._setup_simulation()

        self.load_usd (usd_path=self.asset_paths["shop_table"], prim_path="/World/env/table", prim_name="table_ref_geom", 
                            pos=p, quat=None, scale=0.01)

        self.load_usd (usd_path=self.asset_paths["tooling_plate"], prim_path="/World/env/tooling_plate", prim_name="tooling_plate_geom", 
                            pos=p+np.array([0, 0, 0.815]), quat=None, scale=0.01)
        
        # add_reference_to_stage(usd_path=self.asset_paths["shop_table"], prim_path="/World/env/table")
        # self.world.scene.add(GeometryPrim(prim_path="/World/env/table", name=f"table_ref_geom", collision=True))
        # self._table_position = np.array([0.5, 0.0, 0.0])
        # self._table_scale = 0.01
        # self._table_ref_geom.set_local_scale(np.array([self._table_scale]))
        # self._table_ref_geom.set_world_pose(position=self._table_position)

        # add_reference_to_stage(usd_path=self.asset_paths["tooling_plate"], prim_path="/World/env/tooling_plate")
        # self.world.scene.add(GeometryPrim(prim_path="/World/env/tooling_plate", name=f"tooling_plate_geom", collision=True))

        return



    def load_franka (self, world_to_robotbase, franka_name='Franka', prim_path='/World/robot'):
        self.world.scene.add(Franka(prim_path=prim_path, name=franka_name))
        franka_xform_prim = XFormPrim(prim_path=prim_path, visible=True)
        franka_xform_prim.set_world_pose(self.ft.position_from_transform(world_to_robotbase), self.ft.quat_from_transform(world_to_robotbase))
        ######################################
        my_franka = self.world.scene.get_object(franka_name)
        my_controller = KinematicsSolver(my_franka) #franka class only for IK debug
        articulation_controller = my_franka.get_articulation_controller() 
        return my_franka, my_controller, articulation_controller        
        ######################################


    def load_usd (self, usd_path, prim_path, prim_name, pos, quat, scale):
        prim = add_reference_to_stage(usd_path=usd_path, prim_path=prim_path) # Loads the USD at prim_path
        xform_prim = XFormPrim(prim_path=prim_path, position=[0, 0, 0], orientation=[1, 0, 0, 0], visible=True)
        xform_prim.set_world_pose(pos, quat)
        self.world.scene.add(GeometryPrim(prim_path=prim_path, name=prim_name, collision=True))
        p = self.world.scene.get_object(prim_name)
        p.set_local_scale(np.array([scale]))
        return prim, xform_prim


    def load_scene_usd_kitchencabinetreplaced (self):
        kitchencornerstatic_path = os.path.join(SCENEUSD_DIR, 'Collected_srl-kitchen-no-articulation_corner_1m_flattened_cabinethole/kitchencabinetreplaced.usd')
        prim_path = '/World/kitchencornerstatic'
        prim_name = "kitchencornerstatic"
        pos =  np.array([-1.1, 1.68, 0.5])#-0.4])
        quat = np.array([np.cos(np.radians(-90) / 2), 0, 0, np.sin(np.radians(-90) / 2)]) 
        prim, xform_prim = self.load_usd(kitchencornerstatic_path, prim_path, prim_name, pos, quat, scale=1)
        return prim, xform_prim

    def load_default_plane(self):
        self.world.scene.add_default_ground_plane()

    def load_cubes_in_poses(self):
        """
        Creates/loads cubes in poses stored in self._cube_initial_positions
        As many cubes as poses in _cube_initial_positions
        """
        self._cubes = []
        self._cube_size=np.array([0.045, 0.045, 0.045])
        colors = {
            0: ("Cube_red", np.array([1, 0, 0])),
            1: ("Cube_green", np.array([0, 1, 0])),
            2: ("Cube_blue", np.array([0, 0, 1])),
            3: ("Cube_yellow", np.array([1, 1, 0]))
        }

        for i in range(len(self._cube_initial_positions)):
            # color = np.random.uniform(size=(3,))
            name, color = colors[i % len(colors)]
            cube_prim_path = find_unique_string_name(
                initial_name="/World/Cube",
                is_unique_fn=lambda x: not is_prim_path_valid(x),
            )
            cube_name = find_unique_string_name(
                initial_name=name,
                is_unique_fn=lambda x: not self.world.scene.object_exists(x),
            )
            self._cubes.append(
                self.world.scene.add(
                    DynamicCuboid(
                        name=cube_name,
                        position=self._cube_initial_positions[i],
                        orientation=None,
                        prim_path=cube_prim_path,
                        scale=self._cube_size,
                        size=1.0,
                        color=color,
                    )
                )
            )

    def move_cubes_to_poses(self):
        for i, existing_cube in enumerate(self._cubes):
            # teleport the cube:
            existing_cube.set_world_pose(self._cube_initial_positions[i], np.array([1,0,0,0]))
            # todo: quat
            self.step_physics_and_render(50)
            

    def reshuffle_cubes(self, cube_xyz_ranges):
        self.generate_random_cube_poses(cube_xyz_ranges)
        self.move_cubes_to_poses()

    def load_random_cubes(self, cube_xyz_ranges):
        self.generate_random_cube_poses(cube_xyz_ranges)
        self.load_cubes_in_poses()
        
    def generate_random_cube_poses(self, cube_xyz_ranges):  
        # n = len(cube_xyz_ranges)
        self._cube_initial_positions = []
        for cube_pose_range in cube_xyz_ranges:
            x = random.uniform(cube_pose_range[0], cube_pose_range[1])
            y = random.uniform(cube_pose_range[2], cube_pose_range[3])
            z = random.uniform(cube_pose_range[4], cube_pose_range[5])
            self._cube_initial_positions.append([x, y, z])
        

    def load_four_cubes(self):
        self._cubes = []
        self._cube_initial_positions=np.array(
                [
                    [0.5, -0.1, 0.86],
                    [0.5, 0.1, 0.86],
                    [0.4, -0.2, 0.86],
                    [0.6, -0.2, 0.86],
                    [0.4, 0.2, 0.86],
                    # [0.6, 0.2, 0.86],
                ]
        )
        self.load_cubes_in_poses()


    @staticmethod
    def animate_states(world, states, steps_per_state):
        print (f"Playing new demo with {len(states)} states")
        bodies = {}

        init_renderer = True
        for i, state in enumerate(states):
            print(f'State: {i}/{len(states)})')
            for body_path, body_state in state.items():
                if body_path not in bodies:
                    joint_state = body_state.get('joint_state', None)
                    if joint_state is None:
                        body = RigidPrim(prim_path=body_path)
                        body.disable_rigid_body_physics()
                    else:
                        body = Articulation(prim_path=body_path)
                        body.initialize()
                        body.disable_gravity()
                        body.set_enabled_self_collisions(False)
                    bodies[body_path] = body

                body = bodies[body_path]
                IsaacSimUtils.set_body_state(body, body_state)
                
                for _ in range(steps_per_state):
                    world.step(render=True)

    
    @staticmethod
    def disable_collisions(prim):
        # Disable colliders
        for p in Usd.PrimRange(prim):
            if p.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI(p)
                collision_api.GetCollisionEnabledAttr().Set(False)


    def print_scene_registry (self):
        scene_registry = self.world.scene._scene_registry
        print('XForms:', list(scene_registry.xforms))
        print('Rigid objects:', list(scene_registry.rigid_objects))
        print('Articulated objects:', list(scene_registry.articulated_systems))
        print('Robots:', list(scene_registry.robots))
        print('Sensors:', list(scene_registry.sensors))


    @staticmethod
    def find_prim_by_name(stage, prim_name):
        """
        stage (Usd.Stage): The stage to search within.
        prim_name (str): The name of the prim to find.
        Usd.Prim: Prim with name prim_name or None if not found
        """
        # Check if the current stage's root prim has the name
        root_prim = stage.GetPseudoRoot()
        if root_prim.GetName() == prim_name:
            return root_prim

        # DFS
        stack = [root_prim]
        while stack:
            current_prim = stack.pop()
            for child in current_prim.GetAllChildren():
                if child.GetName() == prim_name:
                    return child   #self.prim_to_xformable(child, prim_name, scale = 1)
                stack.append(child)

        return None #not found


    @staticmethod
    def find_prims_by_name_containing(stage, search_string):
        """
        stage (Usd.Stage): The stage to search within.
        search_string (str): The substring to search for in the prim's name.
        list[Usd.Prim]: List of Prims whose names contain the search_string
        """
        result = []
        root_prim = stage.GetPseudoRoot()

        # DFS
        stack = [root_prim]
        while stack:
            current_prim = stack.pop()
            if search_string in current_prim.GetName():
                result.append(current_prim)
            for child in current_prim.GetAllChildren():
                stack.append(child)

        return result

    @staticmethod
    def get_all_prim_paths_to_log_states(world, relevant_objects_str):
        scene_registry = world.scene._scene_registry
        bodies = list(scene_registry.robots.values()) 
        prim_objects = []
        articulation_objects = []
        articulation_objects_dicts = [] #list of dicts
        prim_objects_dicts = [] #list of dicts

        for body in bodies: # objects in world.scene (only robot by now)
            body_prim = body.prim
            body_path = str(body_prim.GetPath())
            articulation_objects.append(body) 
            #bodies. Franka (articulation) instead of /World/robot (no articulation?)
            body_object_dict = {
                    "prim_path": body_path,
                    "xformable": None,
                    "prim_object": body_prim,
                    "object": body,
                    "type": "articulated"
                    }
            articulation_objects_dicts.append(body_object_dict)

        # for prim_str in relevant_objects_str:
        for relevant_str in relevant_objects_str:
            # find relevant prims by looking for a string in their name, such as "Cube":
            found_relevant_prims = IsaacSimUtils.find_prims_by_name_containing(world.stage, relevant_str)
            for prim in found_relevant_prims:
                prim_objects.append(prim)
                prim_path = str(prim.GetPath())
                xformable = XFormPrim(prim_path, name=prim_path)
                prim_object_dict = {
                    "prim_path": prim_path,
                    "xformable": xformable,
                    "prim_object": prim,
                    "type": "nonarticulated"
                    }
                prim_objects_dicts.append(prim_object_dict)

        return articulation_objects, prim_objects, articulation_objects_dicts, prim_objects_dicts

    @staticmethod
    def get_states_all_listed_prims(articulation_objects_dicts, prim_objects_dicts):
        # ~18msec
        world_state = {}
        # objects in world.scene:
        for body in articulation_objects_dicts:
            # start_time1 = time.time()
            body_state = IsaacSimUtils.get_articulation_state(body["object"])
            world_state[body["prim_path"]] = body_state
            # end_time = time.time()
            # print(f"get_states_all_listed_prims:articulation_objects: {end_time - start_time1} seconds")
        for prim in prim_objects_dicts:
            # start_time1 = time.time()
            body_state = IsaacSimUtils.get_nonarticulation_state(prim["xformable"])
            world_state[prim["prim_path"]] = body_state
            # end_time = time.time()
            # print(f"get_states_all_listed_prims:prim_objects: {end_time - start_time1} seconds")
        return world_state


    @staticmethod
    def get_xform_prim_names(root_path, my_world):
        """Collect names of all Xform prims under a specified path and register them.
        # Example usage
        # xform_prim_names = get_xform_prim_names("/World",my_world)
        # print(xform_prim_names)
        """
        # TODO for adding all prims in big usd scene to the scene registry?

        stage = omni.usd.get_context().get_stage()
        xform_prim_names = []
        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim:
            print(f"No prim found at the path: {root_path}")
            return xform_prim_names
        
        # Iterate through all children of the root prim
        for prim in Usd.PrimRange(root_prim):
            if prim.IsA(UsdGeom.Xform):
                prim_name = prim.GetPath().name
                xform_prim_names.append(prim_name)
                # my_world.scene.add(XFormPrim(prim_path=prim.GetPath(), name=prim.GetPath().name))
        return xform_prim_names



    ########################################################################
    # From basic SRL Isaac Sim utils:
    ####################################

    @staticmethod
    def get_pose(body):
        if isinstance(body, Articulation):
            pose = FramesTransforms.transform_from_pq(p=body.get_world_pose()[0], quat=body.get_world_pose()[1])
        elif isinstance(body,XFormPrim): # elif body and body.IsA(UsdGeom.Xformable):
                # xformable = UsdGeom.Xformable(body)
            # xformable = XFormPrim(str(body.GetPath()), "a")
            pose = FramesTransforms.transform_from_pq(p=body.get_world_pose()[0], quat=body.get_world_pose()[1])
        else:
            pose = FramesTransforms.identity_transform()
            print ("get_pose(body): Issue with poses zzzzzzzzzzzzzz")
        return pose

    @staticmethod
    def get_joint_state(body):
        # from nvidia.srl.basics.types import JointState
        if not isinstance(body, Articulation):
            return None
        joint_state = {
            'name': body.dof_names,
            'position': body.get_joint_positions(),
            'velocity': body.get_joint_velocities(),
        }
        return joint_state

    @staticmethod
    def get_body_state(body):
        # from nvidia.srl.basics.types import BodyState
        body_state = {
            'pose': IsaacSimUtils.get_pose(body),
            'joint_state': IsaacSimUtils.get_joint_state(body),
        }
        return body_state

    @staticmethod
    def get_articulation_state(body):
        body_state = {
            'pose': IsaacSimUtils.get_pose(body),
            'joint_state': IsaacSimUtils.get_joint_state(body),
        }
        return body_state

    @staticmethod
    def get_nonarticulation_state(body):
        body_state = {
            'pose': IsaacSimUtils.get_pose(body),
            'joint_state': None,
        }
        return body_state

    @staticmethod
    def get_world_state(world):
        scene_registry = world.scene._scene_registry
        bodies = list(scene_registry.robots.values()) + \
                list(scene_registry.rigid_objects.values())
        world_state = {}
        for body in bodies:
            body_prim = body.prim
            body_path = str(body_prim.GetPath())
            body_state = IsaacSimUtils.get_body_state(body)
            world_state[body_path] = body_state
        return world_state

    @staticmethod
    def set_pose(body, pose):
        position = pose[:3, 3]
        matrix = pose[:3, :3]
        # Convert from the [x, y, z, w] to [w, x, y, z] quaternion convention
        quat_xyzw = Rotation.from_matrix(matrix).as_quat() #todo change lib
        quat_wxyz = np.roll(quat_xyzw, shift=1)
        # ft.R_t_from_transform(pose)
        body.set_world_pose(position=position, orientation=quat_wxyz)
        return True

    @staticmethod
    def set_joint_state(body, joint_state):
        if joint_state is None:
            return False
        joint_positions = joint_state['position']
        body.set_joint_positions(positions=joint_positions, joint_indices=None)
        return True

    @staticmethod
    def set_body_state(body, body_state):
        pose = body_state["pose"]
        IsaacSimUtils.set_pose(body, pose)
        joint_state = body_state.get("joint_state", None)
        return IsaacSimUtils.set_joint_state(body, joint_state)

    ########################################################################



from datetime import datetime
import shutil
import sys
import pickle
class SimDataLog:

    def __init__(self, relevant_objects_str=None, world=None) -> None:  

        self.current_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.str_current_datetime = str(self.current_datetime)
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        self.states_list = []
        self.data_dict = {}
        self.data_dict['states_list'] = []

        self.relevant_objects_str = relevant_objects_str
        self.articulation_objects, self.prim_objects, self.articulation_objects_dicts, self.prim_objects_dicts = IsaacSimUtils.get_all_prim_paths_to_log_states(world, self.relevant_objects_str)
        # Create output dir for VR demos:
        self.output_dir = os.path.join(self.root_dir, 'data', 'demosVR', 'output_' + self.str_current_datetime)
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
            print('Removed:', self.output_dir)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def append_states_this_sim_step(self):
        states_i_dict = IsaacSimUtils.get_states_all_listed_prims(self.articulation_objects_dicts, self.prim_objects_dicts) #states_i_dict["/World/robot"]["joint_state"]["position"]
        self.states_list.append(states_i_dict) #append only to list, and get list into dict at saving step

    def save_world_usd(self, my_world):
        #Path for saving the usd (scene + robot):
        self.usd_path = os.path.join(self.output_dir, 'scene_demo.usda')
        my_world.stage.Export(self.usd_path)
        print('Exported USD:', self.usd_path)
        # todo: "collect as to save all assets standalone"

    def proccess_and_save_data(self, physics_dt):
        self.data_dict.setdefault('states_list', []).extend(self.states_list)
        self.save_data(self.data_dict) #don't update data_dict here as loop relies on .clear()
        self.states_list = []
        self.data_dict['states_list'] = []
        self.data_dict.update({'time_step': physics_dt})

    def save_data(self, data_dict):
        if not data_dict:
            return False
        
        current_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        str_current_datetime = str(current_datetime)
        filename = "states_" + str_current_datetime + ".pkl"
        states_path = os.path.join(self.output_dir, filename)
        with open(states_path, "wb") as file:
            pickle.dump(data_dict, file)

        states = data_dict.get('states', [])
        print(f'Saved {len(states)} states:', states_path)
        print("filename: " + filename)
        data_dict.clear()

        return True

    @staticmethod
    def load_demos_data(folder_path):
        """
        load data from all pkl data files in a folder
        """
        files = os.listdir(folder_path) # list all files in folder_path
        state_files = [file for file in files if file.startswith("states_")] # all files that start with 'states_'
        
        # Process each states_ file
        demos_data = []
        for state_file in state_files:
            states_file_path = os.path.join(folder_path, state_file)
            with open(states_file_path, "rb") as f:
                this_demo_data = pickle.load(f)
                demos_data.append(this_demo_data)
                print(f'Loaded data:', states_file_path)
                print(f"Processing {state_file}...")  
                # print(this_demo_data)  
        return demos_data
############################################################




import transforms3d as t3d
class FramesTransforms(object):

    @staticmethod
    def identity_transform():
        return np.eye(4)

    @staticmethod
    def identity_quat():
        return np.array([1.0, 0.0, 0.0, 0.0])

    @staticmethod
    def transform_from_pq (p, quat=np.array([1.0, 0.0, 0.0, 0.0])):
        """
        p (np.array) 
        quat (np.array)
        """
        transform = np.eye(4)
        rotation_matrix = t3d.quaternions.quat2mat(quat)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = p
        return transform
    
    @staticmethod
    def transform_from_rotmat_p (rotation_matrix, p=np.array([0.0, 0.0, 0.0])):
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = p
        return transform

    @staticmethod
    def rotation_from_transform(T):
        return T[:3, :3]
    
    @staticmethod
    def quat_from_transform(T):
        t = T[:3, :3]
        q = t3d.quaternions.mat2quat(t)
        return q

    @staticmethod
    def position_from_transform(T):
        return T[:3, 3]

    @staticmethod
    def concatenate_transforms(t1, t2):
        t = np.dot(t1, t2)
        return t
    
    @staticmethod
    def R_t_from_transform(T):
        R = FramesTransforms.rotation_from_transform(T)
        t = FramesTransforms.position_from_transform(T)
        return R, t

    @staticmethod
    def invert_homogeneous_transform(T):
        R,t = FramesTransforms.R_t_from_transform(T)

        R_inv = R.T
        t_inv = -np.dot(R_inv, t)

        T_inv = FramesTransforms.transform_from_rotmat_p(R_inv, t_inv)
        return T_inv

    @staticmethod
    def delta_transform(T1, T2): 
        R1,t1 = FramesTransforms.R_t_from_transform(T1)
        R2,t2 = FramesTransforms.R_t_from_transform(T2)

        R1_inv = R1.T
        R1_to_R2 = FramesTransforms.concatenate_transforms(R2, R1_inv) 

        delta_t = t2 - t1

        return delta_t, R1_to_R2

########################################################################


import matplotlib.pyplot as plt
import time
class Plotter():
    
    def __init__(self, n_variables=1):
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots()
        self.x_values = []
        self.y_values = [[] for _ in range(n_variables)]
        self.lines = []
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Extend 
        
        for i in range(n_variables):
            line, = self.ax.plot(self.x_values, self.y_values[i], self.colors[i % len(self.colors)] + '-', label=f'Variable {i+1}')
            self.lines.append(line)
        
        self.ax.legend(loc='upper left')  
        self.ax.set_xlim(0, 100)  
        self.ax.set_ylim(-4, 4) 


    def update_plot(self, i_sim, data_points):
        self.x_values.append(i_sim)
        
        # data_points: list 
        for idx, data_point in enumerate(data_points):
            self.y_values[idx].append(data_point)
            self.lines[idx].set_xdata(self.x_values)
            self.lines[idx].set_ydata(self.y_values[idx])
    
        if i_sim >= 100: #to do variable
            self.ax.set_xlim(i_sim - 100, i_sim)

        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

########################################################################


class KeyboardTeleopDebug(object): 
    # modified from Isaac Sim examples
    def __init__(self, franka) -> None:
        # Set up keyboard listener
        from omni import appwindow
        self._appwindow = appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)
        self.gripper_action = 0
        self.my_franka = franka

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        """
        [Summary]
        Keyboard subscriber callback to when kit is updated.
        """
        # reset event
        self._event_flag = False
        print ("++++++++debug: Keyboard event: ", event.input.name) # TODO(claudia): 'str' object has no attribute 'name'

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            print ("++++++++debug: KeyboardEventType.KEY_PRESS: ")
            # on pressing, the command is incremented
            if event.input.name == "UP":
                print ("++++++++debug:_input_keyboard_mapping: Open gripper")
                self.my_franka.gripper.open()
                self.gripper_action = +1
            elif event.input.name == "DOWN":
                print ("++++++++debug:_input_keyboard_mapping: Close gripper")
                self.my_franka.gripper.close()
                self.gripper_action = -1
            else:
                return

        return True

########################################################################



