# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License.

# Modified from:
# https://github.com/NVlabs/fast-explicit-teleop/blob/main/srl/teleop/assistance/ghost_franka.py



from typing import Optional, List
import numpy as np

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid
from pxr import Usd, UsdGeom, Gf, UsdPhysics, PhysxSchema, UsdShade, Sdf
import omni
from omni.isaac.core.materials.visual_material import VisualMaterial
from omni.isaac.franka import Franka


import os

MATERIAL_DIR_PATH = os.path.realpath(os.path.dirname(__file__))
def load_ghost_material():
    success = omni.kit.commands.execute(
        "CreateMdlMaterialPrim",
        mtl_url=os.path.join(MATERIAL_DIR_PATH, "GhostVolumetric.mdl"),
        mtl_name="voltest_02",
        mtl_path=Sdf.Path("/Looks/GhostVolumetric"),
    )
    shader = UsdShade.Shader(get_prim_at_path("/Looks/GhostVolumetric/Shader"))
    material = UsdShade.Material(get_prim_at_path("/Looks/GhostVolumetric"))

    shader.CreateInput("absorption", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.8, 0.8, 0.8))
    shader.CreateInput("scattering", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.5, 0.5, 0.5))
    shader.CreateInput("transmission_color", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.1, 1.0, 0.3)
    )
    shader.CreateInput("emission_color", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.1, 1.0, 0.3)
    )
    shader.CreateInput("distance_scale", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("emissive_scale", Sdf.ValueTypeNames.Float).Set(300.0)
    shader.CreateInput("transmission_color", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.3, 1.0, 0.3)
    )

    material = VisualMaterial(
        name="GhostVolumetric",
        prim_path=f"/Looks/GhostVolumetric",
        prim=get_prim_at_path(f"/Looks/GhostVolumetric"),
        shaders_list=[shader],
        material=material,
    )
    material_inputs = {}
    for input in material.shaders_list[0].GetInputs():
        material_inputs[input.GetFullName()] = input
    return material, material_inputs


class GhostFranka(Franka):
    """[summary]

        Args:
            prim_path (str): [description]
            name (str, optional): [description]. Defaults to "franka_robot".
            usd_path (Optional[str], optional): [description]. Defaults to None.
            position (Optional[np.ndarray], optional): [description]. Defaults to None.
            orientation (Optional[np.ndarray], optional): [description]. Defaults to None.
            end_effector_prim_name (Optional[str], optional): [description]. Defaults to None.
            gripper_dof_names (Optional[List[str]], optional): [description]. Defaults to None.
            gripper_open_position (Optional[np.ndarray], optional): [description]. Defaults to None.
            gripper_closed_position (Optional[np.ndarray], optional): [description]. Defaults to None.
        """

    def __init__(
        self,
        prim_path: str,
        name: str = "franka_robot",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        end_effector_prim_name: Optional[str] = None,
        gripper_dof_names: Optional[List[str]] = None,
        gripper_open_position: Optional[np.ndarray] = None,
        gripper_closed_position: Optional[np.ndarray] = None,
        disable_collisions: bool = True,
    ) -> None:
            super().__init__(prim_path, name, usd_path, position, orientation,end_effector_prim_name, gripper_dof_names, gripper_open_position, gripper_closed_position)

            self.material, self.material_inputs = load_ghost_material()
            self.material_inputs["inputs:transmission_color"].Set((1, 1, 1))
            self.imageable = UsdGeom.Imageable(self.prim)
            self.apply_visual_material(self.material)
            if disable_collisions:
                self.disable_collisions()
            self._current_color = None
            self._current_opacity = None
            # Populate simplifed meshes under the right links of the robot
            self.viz_palm = add_reference_to_stage(usd_path=os.path.join(MATERIAL_DIR_PATH, "panda_hand_viz.usd"), prim_path=prim_path + "/panda_hand/viz")
            self.viz_left_finger = add_reference_to_stage(usd_path=os.path.join(MATERIAL_DIR_PATH, "panda_leftfinger_viz.usd"), prim_path=prim_path + "/panda_leftfinger/viz")
            self.viz_right_finger = add_reference_to_stage(usd_path=os.path.join(MATERIAL_DIR_PATH, "panda_rightfinger_viz.usd"), prim_path=prim_path + "/panda_rightfinger/viz")
            for p in [self.viz_left_finger, self.viz_right_finger, self.viz_palm]:
                viz_mesh = get_prim_at_path(f"{p.GetPath()}/mesh")
                viz_mesh.CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
            # FIXME: Cannot remove ancestral prim...
            # deleted = delete_prim(self.camera.GetPath())

    def disable_collisions(self):
        # Disable colliders

        for p in Usd.PrimRange(self.prim):
            if p.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI(p)
                collision_api.GetCollisionEnabledAttr().Set(False)

    def hide(self):
        self.imageable.MakeInvisible()

    def show(self, gripper_only=False):
        if not gripper_only:
            self.imageable.MakeVisible()
        else:
            for p in [self.viz_left_finger, self.viz_right_finger, self.viz_palm]:
                UsdGeom.Imageable(p).MakeVisible()

    def set_color(self, color, opacity=1.0):
        if color == self._current_color and opacity == self._current_opacity:
            # idempotent
            return
        transmission = 1.0 - opacity

        def clip(value):
            # Inputs seem to behave differently for 0 and close to 0 for some reason...
            return Gf.Vec3f(*np.clip(value, 0.0001, 1.0))
        # The colors you don't absorb will shine through.
        # The color you emit shows in the absence of other colors
        if color == "red":
            self.material_inputs["inputs:emission_color"].Set((1., .0, .0))
            self.material_inputs["inputs:absorption"].Set(clip((.0, transmission, transmission)))
        elif color == "yellow":
            self.material_inputs["inputs:emission_color"].Set((1., 1., .0))
            self.material_inputs["inputs:absorption"].Set(clip((.0, .0, transmission)))
        elif color == "green":
            self.material_inputs["inputs:emission_color"].Set((.0, 1., .0))
            self.material_inputs["inputs:absorption"].Set(clip((transmission, .0, transmission)))
        elif color == "grey":
            self.material_inputs["inputs:emission_color"].Set((1., 1., 1.))
            self.material_inputs["inputs:absorption"].Set(clip((transmission, transmission, transmission)))
        else:
            return

        self._current_color = color
        self._current_opacity = opacity
