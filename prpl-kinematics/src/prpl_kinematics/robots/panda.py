"""The Franka Panda as a prpl_kinematics Robot.

``make_panda`` assembles the Panda's tree, named joint groups, IKFast solver,
home configuration, and intrinsic allowed-collision pairs into a :class:`Robot`,
so callers no longer hand-wire a ``JointSpace``, ``IKFastInfo``, and ACM
discovery at each use site.
"""

from __future__ import annotations

import numpy as np

from prpl_kinematics.collision import discover_allowed_pairs
from prpl_kinematics.ik.ikfast import IKFastInfo, IKFastSolver
from prpl_kinematics.loading import load_urdf
from prpl_kinematics.planning.joint_space import JointSpace
from prpl_kinematics.robots.robot import Manipulator, Robot
from prpl_kinematics.tree.kinematic_tree import Configuration
from prpl_kinematics.utils import get_assets_path

_ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
_FINGER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]
# The IKFast module solves for panda_link8; tool_link is the grasp frame 10cm
# beyond it, and is the robot's end effector.
_IKFAST_EE = "panda_link8"
_EE_FRAME = "tool_link"
# The Franka "ready" arm pose, with the gripper open.
_ARM_HOME = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
_FINGER_OPEN = 0.04

_IKFAST_INFO = IKFastInfo(
    module_dir="panda_arm",
    module_name="ikfast_panda_arm",
    base_link="panda_link0",
    ee_link=_IKFAST_EE,
    free_joints=["panda_joint7"],
)


def make_panda(rng: np.random.Generator | None = None) -> Robot:
    """Assemble a Franka Panda Robot (compiles IKFast on first call)."""
    if rng is None:
        rng = np.random.default_rng(0)
    path = get_assets_path() / "urdf" / "panda_arm_hand.urdf"
    tree = load_urdf(str(path))
    groups = {
        "arm": JointSpace(tree, _ARM_JOINTS),
        "gripper": JointSpace(tree, _FINGER_JOINTS),
    }
    home: Configuration = {
        **{name: [value] for name, value in zip(_ARM_JOINTS, _ARM_HOME)},
        **{name: [_FINGER_OPEN] for name in _FINGER_JOINTS},
    }
    ik = IKFastSolver(tree, _IKFAST_INFO, _ARM_JOINTS, rng, tool_frame=_EE_FRAME)
    return Robot(
        name="panda",
        tree=tree,
        groups=groups,
        manipulators={"arm": Manipulator("arm", _EE_FRAME, ik)},
        home=home,
        allowed_collision_pairs=discover_allowed_pairs(tree, home),
    )
