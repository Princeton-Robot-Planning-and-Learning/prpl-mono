"""The bimanual Dexmate Vega 1U as a prpl_kinematics Robot.

Vega has two 7-DOF arms with parallel-jaw grippers (plus a lift, torso flip, and
head), so it is the multi-manipulator case: per-arm joint and gripper groups and
two ``Manipulator``s, each with its own
:class:`~prpl_kinematics.robots.vega_ik.VegaArmIK` solver. The IK solves the
7-DOF arm to the gripper tool frame (``*_ee``); non-arm joints are held at home.
"""

from __future__ import annotations

from prpl_kinematics.collision import discover_allowed_pairs
from prpl_kinematics.ik.interface import InverseKinematics
from prpl_kinematics.ik.ssik import SSIKSolver
from prpl_kinematics.loading import load_urdf
from prpl_kinematics.planning.joint_space import JointSpace
from prpl_kinematics.robots.robot import Manipulator, Robot
from prpl_kinematics.robots.vega_ik import VegaArmIK
from prpl_kinematics.tree.kinematic_tree import Configuration
from prpl_kinematics.utils import get_assets_path


def _arm_joints(side: str) -> list[str]:
    return [f"{side}_arm_j{i}" for i in range(1, 8)]


def _gripper_joints(side: str) -> list[str]:
    return [f"{side}_gripper_j1", f"{side}_gripper_j2"]


# A natural, manipulation-ready home (per-arm j1..j7): forearms down-forward with
# the grippers angled downward at chest height, a ready posture from which a
# tabletop grasp is a short, smooth reach. The left posture was optimized to
# reduce static gravity torque while staying well inside its joint limits; the
# right home is its mirror, so the robot rests symmetrically. Between the two
# candidate postures, the left one holds gravity with the lower peak joint torque
# (about 15 vs 21 N*m in the URDF model, 2.9 vs 4.0 A on hardware), which matters
# for a long-lived parked pose; even so, the shoulder joints work noticeably to
# hold it. Vega's arms are not exact mechanical mirrors, so the mirrored posture's
# gravity load and the symmetry of the gripper poses (~2.5 cm residual) are
# approximate. See prpl-mono issue #529 for the measurements behind this choice.
_LEFT_ARM_HOME = [1.809, 0.636, -0.244, -2.04, 0.841, 0.129, -0.833]
# The left-to-right mirror map (from the URDF joint limits) flips the sign of
# every joint except j4, the elbow, whose range is identical on both arms.
_RIGHT_ARM_HOME = [q if i == 3 else -q for i, q in enumerate(_LEFT_ARM_HOME)]
# Rest the parallel jaws open (each finger is revolute over [0, 0.785]); like the
# Panda's open home, this is the ready-to-grasp pose, and a grasp closes on the
# object rather than starting clamped shut through it.
_GRIPPER_OPEN = 0.6


def make_vega(ik: str = "eaik") -> Robot:
    """Assemble a bimanual Dexmate Vega 1U Robot (with parallel-jaw grippers).

    ``ik`` selects the per-arm analytic IK backend: ``"eaik"`` (default, the
    bundled :class:`~prpl_kinematics.robots.vega_ik.VegaArmIK` lock-and-search) or
    ``"ssik"`` (the optional :class:`~prpl_kinematics.ik.ssik.SSIKSolver`, which
    requires the ``ssik`` package). Both solve Vega's non-SRS 7R arm.
    """
    urdf = get_assets_path() / "urdf" / "vega" / "vega_1u_gripper.urdf"
    tree = load_urdf(str(urdf))
    groups = {
        "left_arm": JointSpace(tree, _arm_joints("L")),
        "right_arm": JointSpace(tree, _arm_joints("R")),
        "left_gripper": JointSpace(tree, _gripper_joints("L")),
        "right_gripper": JointSpace(tree, _gripper_joints("R")),
    }

    def make_ik(prefix: str) -> InverseKinematics:
        arm, ee, tool = _arm_joints(prefix), f"{prefix}_arm_l7", f"{prefix}_ee"
        if ik == "ssik":
            return SSIKSolver(tree, arm, ee, urdf, tool_frame=tool)
        if ik == "eaik":
            return VegaArmIK(tree, arm, ee, tool_frame=tool)
        raise ValueError(f"unknown ik backend {ik!r}; expected 'eaik' or 'ssik'")

    manipulators = {
        side_name: Manipulator(f"{side_name}_arm", f"{prefix}_ee", make_ik(prefix))
        for side_name, prefix in [("left", "L"), ("right", "R")]
    }
    rest: dict[str, list[float]] = {}
    for i, (left, right) in enumerate(zip(_LEFT_ARM_HOME, _RIGHT_ARM_HOME), start=1):
        rest[f"L_arm_j{i}"] = [left]
        rest[f"R_arm_j{i}"] = [right]
    for side in ("L", "R"):
        for finger in _gripper_joints(side):
            rest[finger] = [_GRIPPER_OPEN]
    home: Configuration = {
        name: rest.get(name, [0.0] * tree.joint(name).num_dof)
        for name in tree.actuated_joint_names()
    }
    return Robot(
        name="vega-1u",
        tree=tree,
        groups=groups,
        manipulators=manipulators,
        home=home,
        allowed_collision_pairs=discover_allowed_pairs(tree, home),
    )
