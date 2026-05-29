"""The Kinova Gen3 (7-DOF) as a prpl_kinematics Robot.

``make_kinova`` assembles the Gen3's tree, arm joint group, IKFast solver, home
configuration, and intrinsic allowed-collision pairs into a :class:`Robot`.
Joints 1, 3, 5, and 7 are continuous (unlimited), so the arm group exercises
``JointSpace``'s wrap-around handling of continuous joints.
"""

from __future__ import annotations

import numpy as np

from prpl_kinematics.collision import discover_allowed_pairs
from prpl_kinematics.ik.ikfast import IKFastInfo, IKFastSolver
from prpl_kinematics.loading import load_urdf
from prpl_kinematics.planning.joint_space import JointSpace
from prpl_kinematics.robots.robot import Robot
from prpl_kinematics.tree.kinematic_tree import Configuration
from prpl_kinematics.utils import get_assets_path

_ARM_JOINTS = [f"joint_{i}" for i in range(1, 8)]
# The IKFast module solves for end_effector_link; tool_frame is the grasp frame
# beyond it, and is the robot's end effector.
_IKFAST_EE = "end_effector_link"
_EE_FRAME = "tool_frame"
_ARM_HOME = [-4.3, -1.6, -4.8, -1.8, -1.4, -1.1, 1.6]

_IKFAST_INFO = IKFastInfo(
    module_dir="kortex",
    module_name="ikfast_kortex",
    base_link="base_link",
    ee_link=_IKFAST_EE,
    free_joints=["joint_7"],
)


def make_kinova(rng: np.random.Generator | None = None) -> Robot:
    """Assemble a Kinova Gen3 Robot (compiles IKFast on first call)."""
    if rng is None:
        rng = np.random.default_rng(0)
    path = get_assets_path() / "urdf" / "gen3_7dof.urdf"
    tree = load_urdf(str(path))
    groups = {"arm": JointSpace(tree, _ARM_JOINTS)}
    home: Configuration = {name: [value] for name, value in zip(_ARM_JOINTS, _ARM_HOME)}
    ik = IKFastSolver(tree, _IKFAST_INFO, _ARM_JOINTS, rng, tool_frame=_EE_FRAME)
    return Robot(
        name="kinova-gen3",
        tree=tree,
        groups=groups,
        ee_frame=_EE_FRAME,
        ik=ik,
        home=home,
        allowed_collision_pairs=discover_allowed_pairs(tree, home),
    )
