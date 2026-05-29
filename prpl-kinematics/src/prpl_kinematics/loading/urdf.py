"""Load a URDF into a KinematicTree.

Parsing is delegated to ``yourdfpy`` (pure-Python, lazy mesh loading); this
module maps its link/joint graph onto our ``Node``/``Edge``/``Joint`` types.
"""

from __future__ import annotations

import math
from pathlib import Path

import yourdfpy
from spatialmath import SE3

from prpl_kinematics.tree.joints import (
    FixedJoint,
    Joint,
    PrismaticJoint,
    RevoluteJoint,
)
from prpl_kinematics.tree.kinematic_tree import Edge, KinematicTree, Node


def load_urdf(path: Path | str, root: str | None = None) -> KinematicTree:
    """Build a ``KinematicTree`` from a URDF file.

    URDF links become nodes and joints become edges. ``revolute`` and
    ``prismatic`` joints carry their position limits; ``continuous`` maps to an
    unlimited revolute joint; ``fixed`` becomes a zero-DOF joint. Each joint's
    ``origin`` is taken from the URDF joint frame. ``planar`` and ``floating``
    URDF joints are not supported.

    ``root`` defaults to the URDF's base link.
    """
    urdf = yourdfpy.URDF.load(
        str(path),
        load_meshes=False,
        build_scene_graph=True,
        build_collision_scene_graph=False,
    )
    base = root if root is not None else urdf.base_link
    tree = KinematicTree(root=base)
    for link_name in urdf.link_map:
        if link_name != base:
            tree.add_node(Node(link_name))
    for joint in urdf.robot.joints:
        tree.add_edge(
            Edge(parent=joint.parent, child=joint.child, joint=_convert_joint(joint))
        )
    return tree


def _convert_joint(joint: yourdfpy.Joint) -> Joint:
    origin = SE3(joint.origin, check=False)
    if joint.type == "fixed":
        return FixedJoint(name=joint.name, origin=origin)
    axis = (float(joint.axis[0]), float(joint.axis[1]), float(joint.axis[2]))
    if joint.type == "prismatic":
        return PrismaticJoint(
            name=joint.name,
            origin=origin,
            axis=axis,
            lower=float(joint.limit.lower),
            upper=float(joint.limit.upper),
        )
    if joint.type == "revolute":
        return RevoluteJoint(
            name=joint.name,
            origin=origin,
            axis=axis,
            lower=float(joint.limit.lower),
            upper=float(joint.limit.upper),
        )
    if joint.type == "continuous":
        return RevoluteJoint(
            name=joint.name, origin=origin, axis=axis, lower=-math.inf, upper=math.inf
        )
    raise ValueError(f"Unsupported URDF joint type: {joint.type}")
