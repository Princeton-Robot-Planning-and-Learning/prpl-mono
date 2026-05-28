"""Dexmate Vega humanoid robots."""

import re
import warnings
from functools import cached_property
from pathlib import Path

from dexmate_urdf import get_robot_path

from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.link import get_link_pose, get_relative_link_pose
from pybullet_helpers.robots import _dexmate_vega_ik as _vega_ik
from pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot

_EAIK_FALLBACK_WARNED = False

_EAIK_INSTALL_HINT = (
    "EAIK is not installed; falling back to pybullet's iterative IK, which is "
    "slower and less accurate for the Vega arm. To enable analytic-quality IK, "
    "install the [dexmate-vega] extra (requires Eigen3 headers: `brew install "
    "eigen` on macOS or `apt install libeigen3-dev` on Debian/Ubuntu)."
)

_LEFT_ARM_JOINT_NAMES = [
    "L_arm_j1",
    "L_arm_j2",
    "L_arm_j3",
    "L_arm_j4",
    "L_arm_j5",
    "L_arm_j6",
    "L_arm_j7",
]


class DexmateVega1UPyBulletRobot(SingleArmPyBulletRobot):
    """Dexmate Vega 1U humanoid; the left arm is exposed as the actuated arm.

    Only the 7 left-arm joints (L_arm_j1..L_arm_j7) are part of arm_joints —
    the prismatic Lift and the torso_flip revolute joint are not actuated by
    this wrapper; they remain at whatever position pybullet's URDF load left
    them in (zero by default). This matches the standard humanoid pattern of
    treating the torso as a fixed offset for arm IK.

    When the optional `eaik` package is installed, custom_inverse_kinematics
    runs a 2D-search-with-refinement on top of EAIK's 5R closed-form solver,
    locking L_arm_j4 (elbow) and L_arm_j7 (wrist roll) per inner call. See
    _dexmate_vega_ik.py for details. Otherwise the framework's pybullet IK is
    used.
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

    @cached_property
    def arm_joints(self) -> list[int]:
        # Expose only the 7 left-arm joints; Lift and torso_flip are excluded.
        # This matches the kinematic chain EAIK solves for, and is the standard
        # humanoid pattern where torso/base motions are planned separately from
        # arm motions.
        return [self.joint_from_name(n) for n in _LEFT_ARM_JOINT_NAMES]

    @property
    def default_home_joint_positions(self) -> JointPositions:
        return [0.0] * 7

    @property
    def end_effector_name(self) -> str:
        return "L_ee_j0"

    @property
    def tool_link_name(self) -> str:
        return "L_ee"

    @property
    def default_inverse_kinematics_method(self) -> str:
        if _vega_ik.EAIK_AVAILABLE:
            return "custom"
        global _EAIK_FALLBACK_WARNED  # pylint: disable=global-statement
        if not _EAIK_FALLBACK_WARNED:
            warnings.warn(_EAIK_INSTALL_HINT, RuntimeWarning, stacklevel=2)
            _EAIK_FALLBACK_WARNED = True
        return "pybullet"

    @cached_property
    def _arm_center_link_id(self) -> int:
        return self.link_from_name("arm_center")

    @cached_property
    def _l_arm_l7_link_id(self) -> int:
        return self.link_from_name("L_arm_l7")

    @cached_property
    def _l_ee_in_l_arm_l7(self) -> Pose:
        # Fixed transform from L_arm_l7 to L_ee (computed once via pybullet).
        return get_relative_link_pose(
            self.robot_id,
            self.tool_link_id,
            self._l_arm_l7_link_id,
            self.physics_client_id,
        )

    def custom_inverse_kinematics(
        self,
        end_effector_pose: Pose,
        validate: bool = True,
        best_effort: bool = False,
        validation_atol: float = 1e-3,
    ) -> JointPositions | None:
        if not _vega_ik.EAIK_AVAILABLE:
            return None

        # Convert the world-frame L_ee target into a L_arm_l7 target expressed
        # in arm_center's frame, which is what the EAIK solver expects.
        world_from_arm_center = get_link_pose(
            self.robot_id, self._arm_center_link_id, self.physics_client_id
        )
        target_in_arm_center = multiply_poses(
            world_from_arm_center.invert(),
            end_effector_pose,
            self._l_ee_in_l_arm_l7.invert(),
        )

        q = _vega_ik.solve_arm_ik(
            target_in_arm_center.to_matrix(), _vega_ik.get_arm_ik_params("L")
        )
        if q is None:
            return None
        return [float(v) for v in q]
