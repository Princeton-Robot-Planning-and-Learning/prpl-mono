"""TidyBot (Kinova Gen3 on a planar mobile base) as a prpl_kinematics Robot.

The mobile base is a ``PlanarJoint`` edge ``world -> mobile_base`` (the SE(2)
"base" group); the Kinova arm is re-parented onto the base via the fixed mount
transform, so one tree holds the whole mobile manipulator. The arm reuses the
Gen3's kortex IKFast solver (the base pose flows through forward kinematics, so
IK works at any base pose).
"""

from __future__ import annotations

import numpy as np
from spatialmath import SE3

from prpl_kinematics.collision import discover_allowed_pairs
from prpl_kinematics.geometry.shapes import BoxShape, MeshShape
from prpl_kinematics.ik.ikfast import IKFastInfo, IKFastSolver
from prpl_kinematics.loading import load_urdf
from prpl_kinematics.planning.configuration_space import ConfigurationSpace
from prpl_kinematics.planning.joint_space import JointSpace
from prpl_kinematics.planning.se2_space import SE2Space
from prpl_kinematics.robots.robot import Robot
from prpl_kinematics.tree.joints import PlanarJoint
from prpl_kinematics.tree.kinematic_tree import Configuration, Edge, Node
from prpl_kinematics.utils import get_assets_path

_ARM_JOINTS = [f"joint_{i}" for i in range(1, 8)]
_BASE_JOINT = "base"
_EE_FRAME = "tool_frame"
_ARM_HOME = [-4.3, -1.6, -4.8, -1.8, -1.4, -1.1, 1.6]
# The Kinova base_link mounts here on the mobile base.
_MOUNT = SE3(0.1199, 0.0, 0.3948)
# Base collision box (from tidybot_base.urdf), raised so it clears the arm mount.
_BASE_BOX = BoxShape(size=(0.548, 0.508, 0.365), origin=SE3(0.0, 0.0, 0.1825))

_IKFAST_INFO = IKFastInfo(
    module_dir="kortex",
    module_name="ikfast_kortex",
    base_link="base_link",
    ee_link="end_effector_link",
    free_joints=["joint_7"],
)


def make_tidybot(
    rng: np.random.Generator | None = None,
    base_bounds: tuple[float, float] = (-2.0, 2.0),
) -> Robot:
    """Assemble a TidyBot Robot (compiles the Gen3 IKFast on first call)."""
    if rng is None:
        rng = np.random.default_rng(0)
    tree = load_urdf(str(get_assets_path() / "urdf" / "gen3_7dof.urdf"))
    meshes = get_assets_path() / "urdf" / "tidybot" / "meshes"
    tree.add_node(
        Node(
            "mobile_base",
            visuals=[
                MeshShape(str(meshes / "body.stl")),
                MeshShape(str(meshes / "arm_plate.stl")),
            ],
            collisions=[_BASE_BOX],
        )
    )
    tree.add_edge(Edge(tree.root, "mobile_base", PlanarJoint(name=_BASE_JOINT)))
    tree.attach("base_link", "mobile_base", _MOUNT)

    groups: dict[str, ConfigurationSpace] = {
        "base": SE2Space(_BASE_JOINT, base_bounds, base_bounds),
        "arm": JointSpace(tree, _ARM_JOINTS),
    }
    home: Configuration = {
        _BASE_JOINT: [0.0, 0.0, 0.0],
        **{name: [value] for name, value in zip(_ARM_JOINTS, _ARM_HOME)},
    }
    ik = IKFastSolver(tree, _IKFAST_INFO, _ARM_JOINTS, rng, tool_frame=_EE_FRAME)
    return Robot(
        name="tidybot",
        tree=tree,
        groups=groups,
        ee_frame=_EE_FRAME,
        ik=ik,
        home=home,
        allowed_collision_pairs=discover_allowed_pairs(tree, home),
    )
