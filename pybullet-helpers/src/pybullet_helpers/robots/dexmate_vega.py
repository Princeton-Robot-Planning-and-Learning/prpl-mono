"""Dexmate Vega humanoid robots."""

import warnings
import xml.etree.ElementTree as ET
from functools import cached_property
from pathlib import Path

from dexmate_urdf import get_robot_path

from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.link import get_link_pose, get_relative_link_pose
from pybullet_helpers.robots import _dexmate_vega_ik as _vega_ik
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot

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

# The parallel-jaw gripper has two revolute jaw joints; the URDF mimics j2 off j1,
# but PyBullet does not enforce <mimic>, so we drive both to the same value.
_LEFT_GRIPPER_JOINT_NAMES = ["L_gripper_j1", "L_gripper_j2"]
_GRIPPER_OPEN = 0.0
_GRIPPER_CLOSED = 0.7854

_PYBULLET_MATERIAL_NAME = "dexmate_vega_default"


def prepare_pybullet_urdf(
    robot_dir: Path,
    urdf_filename: str,
    mesh_substitutions: dict[str, str] | None = None,
) -> Path:
    """Rewrite a dexmate URDF so PyBullet can load it, returning the new path.

    PyBullet cannot read the .glb visual meshes shipped with dexmate-urdf. For each
    visual/collision mesh that references a .glb, we rewrite it to the sibling collision
    .obj when one exists on disk, and otherwise drop that visual/collision element (some
    base/lift/torso/connector links ship only a .glb). A neutral gray material is
    injected into every remaining visual because the collision OBJs have no MTL and
    would otherwise render pure black. The output is written next to the original so the
    relative mesh paths still resolve.

    mesh_substitutions maps substrings to replacements applied to every mesh filename
    (e.g. swapping one gripper's meshes for another's).
    """
    mesh_substitutions = mesh_substitutions or {}
    original_path = robot_dir / urdf_filename
    tree = ET.parse(original_path)
    root = tree.getroot()
    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for element in list(link.findall(tag)):
                mesh = element.find("geometry/mesh")
                if mesh is None:
                    continue
                filename = mesh.get("filename", "")
                for old, new in mesh_substitutions.items():
                    filename = filename.replace(old, new)
                if not filename.endswith(".glb"):
                    mesh.set("filename", filename)
                    continue
                obj_rel = filename.replace("/visual/", "/collision/")[:-4] + ".obj"
                if not (robot_dir / obj_rel).resolve().exists():
                    link.remove(element)
                    continue
                mesh.set("filename", obj_rel)
                if tag == "visual" and element.find("material") is None:
                    material = ET.SubElement(element, "material")
                    material.set("name", _PYBULLET_MATERIAL_NAME)
                    color = ET.SubElement(material, "color")
                    color.set("rgba", "0.75 0.75 0.78 1.0")
    new_path = original_path.parent / (original_path.stem + "-PYBULLET-HELPERS.urdf")
    tree.write(new_path, encoding="unicode")
    return new_path


class DexmateVega1UPyBulletRobot(FingeredSingleArmPyBulletRobot[float]):
    """Dexmate Vega 1U humanoid; the left arm with its parallel-jaw gripper is exposed
    as the actuated arm.

    arm_joints are the 7 left-arm joints (L_arm_j1..L_arm_j7) plus the 2 gripper jaw
    joints. The prismatic Lift and the torso_flip revolute joint are not actuated by
    this wrapper; they remain at whatever position pybullet's URDF load left them in
    (zero by default). This matches the standard humanoid pattern of treating the torso
    as a fixed offset for arm IK.

    When the optional `eaik` package is installed, custom_inverse_kinematics runs a
    2D-search-with-refinement on top of EAIK's 5R closed-form solver, locking L_arm_j4
    (elbow) and L_arm_j7 (wrist roll) per inner call. See _dexmate_vega_ik.py for
    details. Otherwise the framework's pybullet IK is used.
    """

    @classmethod
    def get_name(cls) -> str:
        return "dexmate-vega-1u"

    @property
    def default_urdf_path(self) -> Path:
        # vega_1u_gripper.urdf ships with the DexGripper D (forked fingertips). The
        # DexGripper S is kinematically identical (same mount, joints, and limits) and
        # differs only in its meshes, so we swap the mesh paths to use it.
        return prepare_pybullet_urdf(
            get_robot_path("humanoid", "vega_1u"),
            "vega_1u_gripper.urdf",
            mesh_substitutions={"hands/dexd_gripper/": "hands/dexs_gripper/"},
        )

    @cached_property
    def arm_joints(self) -> list[int]:
        # The 7 left-arm joints followed by the gripper jaw joints. Lift and torso_flip
        # are excluded: this matches the kinematic chain EAIK solves for, and the
        # standard humanoid pattern where torso/base motions are planned separately.
        arm = [self.joint_from_name(n) for n in _LEFT_ARM_JOINT_NAMES]
        return arm + self.finger_ids

    @property
    def default_home_joint_positions(self) -> JointPositions:
        return [0.0] * len(_LEFT_ARM_JOINT_NAMES) + self.finger_state_to_joints(
            self.open_fingers_state
        )

    @property
    def end_effector_name(self) -> str:
        return "L_ee_j0"

    @property
    def tool_link_name(self) -> str:
        return "L_ee"

    @property
    def finger_joint_names(self) -> list[str]:
        return _LEFT_GRIPPER_JOINT_NAMES

    @property
    def open_fingers_state(self) -> float:
        return _GRIPPER_OPEN

    @property
    def closed_fingers_state(self) -> float:
        return _GRIPPER_CLOSED

    def finger_state_to_joints(self, state: float) -> list[float]:
        return [state, state]

    def joints_to_finger_state(self, joint_positions: list[float]) -> float:
        assert len(joint_positions) == 2
        return joint_positions[0]

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
        # The arm solution covers the 7 arm joints; keep the fingers where they are.
        current_fingers = self.finger_state_to_joints(self.get_finger_state())
        return [float(v) for v in q] + current_fingers
