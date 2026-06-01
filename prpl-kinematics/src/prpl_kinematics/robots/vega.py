"""The bimanual Dexmate Vega 1U as a prpl_kinematics Robot.

Vega has two 7-DOF arms with parallel-jaw grippers (plus a lift, torso flip, and
head), so it is the multi-manipulator case: per-arm joint and gripper groups and
two ``Manipulator``s, each with its own
:class:`~prpl_kinematics.robots.vega_ik.VegaArmIK` solver. The IK solves the
7-DOF arm to the gripper tool frame (``*_ee``); non-arm joints are held at home.
"""

from __future__ import annotations

from prpl_kinematics.collision import discover_allowed_pairs
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
# the grippers angled downward at chest height. Each arm was optimized to minimize
# static gravity torque (so the arms nearly hold themselves) and stay well inside
# its joint limits, while pointing the gripper down-forward -- a ready posture from
# which a tabletop grasp is a short, smooth reach. The two arms are optimized
# independently (Vega's arms are not a simple sign-flip mirror of each other).
_LEFT_ARM_HOME = [1.809, 0.636, -0.244, -2.04, 0.841, 0.129, -0.833]
_RIGHT_ARM_HOME = [-1.043, -1.278, -0.793, -1.778, -0.148, -0.396, 0.417]


def make_vega() -> Robot:
    """Assemble a bimanual Dexmate Vega 1U Robot (with parallel-jaw grippers)."""
    tree = load_urdf(str(get_assets_path() / "urdf" / "vega" / "vega_1u_gripper.urdf"))
    groups = {
        "left_arm": JointSpace(tree, _arm_joints("L")),
        "right_arm": JointSpace(tree, _arm_joints("R")),
        "left_gripper": JointSpace(tree, _gripper_joints("L")),
        "right_gripper": JointSpace(tree, _gripper_joints("R")),
    }
    manipulators = {
        side_name: Manipulator(
            f"{side_name}_arm",
            f"{prefix}_ee",
            VegaArmIK(
                tree, _arm_joints(prefix), f"{prefix}_arm_l7", tool_frame=f"{prefix}_ee"
            ),
        )
        for side_name, prefix in [("left", "L"), ("right", "R")]
    }
    arm_home: dict[str, list[float]] = {}
    for i, (left, right) in enumerate(zip(_LEFT_ARM_HOME, _RIGHT_ARM_HOME), start=1):
        arm_home[f"L_arm_j{i}"] = [left]
        arm_home[f"R_arm_j{i}"] = [right]
    home: Configuration = {
        name: arm_home.get(name, [0.0] * tree.joint(name).num_dof)
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
