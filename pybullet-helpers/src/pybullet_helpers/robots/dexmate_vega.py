"""Dexmate Vega humanoid robots."""

import re
from pathlib import Path

from dexmate_urdf import get_robot_path

from pybullet_helpers.joint import JointPositions
from pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot


class DexmateVega1UPyBulletRobot(SingleArmPyBulletRobot):
    """Dexmate Vega 1U humanoid; the left arm is exposed as the actuated arm.

    The kinematic chain from base to the left end-effector includes the prismatic
    Lift, the torso_flip revolute joint, and the seven left arm joints
    L_arm_j1..L_arm_j7.
    """

    @classmethod
    def get_name(cls) -> str:
        return "dexmate-vega-1u"

    @property
    def default_urdf_path(self) -> Path:
        # PyBullet can't load the .glb meshes shipped with dexmate-urdf, so we
        # rewrite the URDF to point at the .obj collision meshes from vega_1
        # (which exist for the head and arm links) and strip the visual/collision
        # blocks for base/lift/torso_flip (which have no .obj alternative).
        robot_dir = get_robot_path("humanoid", "vega_1u")
        original_path = robot_dir / "vega_1u.urdf"
        urdf_str = original_path.read_text(encoding="utf-8")
        urdf_str = re.sub(
            r'\.\./vega_1/meshes/visual/([^"]+)\.glb',
            r"../vega_1/meshes/collision/\1.obj",
            urdf_str,
        )
        urdf_str = re.sub(
            r"\s*<(visual|collision)>\s*<origin[^/]*/>\s*<geometry>\s*"
            r'<mesh filename="meshes/visual/[^"]+\.glb"\s*/>\s*'
            r"</geometry>\s*</\1>",
            "",
            urdf_str,
        )
        # The collision OBJs have no MTL, so without a URDF material PyBullet
        # renders them pure black. Inject a neutral gray into every <visual>.
        material_xml = (
            '<material name="dexmate_vega_default">'
            '<color rgba="0.75 0.75 0.78 1.0"/>'
            "</material>"
        )
        urdf_str = re.sub(
            r"(</geometry>)(\s*</visual>)",
            r"\1" + material_xml + r"\2",
            urdf_str,
        )
        # Write next to the original so the relative mesh paths still resolve.
        new_path = original_path.parent / "vega_1u-PYBULLET-HELPERS.urdf"
        new_path.write_text(urdf_str, encoding="utf-8")
        return new_path

    @property
    def default_home_joint_positions(self) -> JointPositions:
        # Lift, torso_flip, L_arm_j1..L_arm_j7 — all zeros are within limits.
        return [0.0] * 9

    @property
    def end_effector_name(self) -> str:
        return "L_ee_j0"

    @property
    def tool_link_name(self) -> str:
        return "L_ee"
