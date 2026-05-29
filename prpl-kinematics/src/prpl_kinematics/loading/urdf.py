"""Load a URDF into a KinematicTree.

Parsing is delegated to ``yourdfpy`` (pure-Python, lazy mesh loading); this
module maps its link/joint graph onto our ``Node``/``Edge``/``Joint`` types and
attaches each link's visual and collision geometry to its node.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from pathlib import Path

import yourdfpy
from spatialmath import SE3

from prpl_kinematics.geometry.shapes import (
    BoxShape,
    CylinderShape,
    MeshShape,
    Shape,
    SphereShape,
)
from prpl_kinematics.tree.joints import (
    FixedJoint,
    Joint,
    PrismaticJoint,
    RevoluteJoint,
)
from prpl_kinematics.tree.kinematic_tree import Edge, KinematicTree, Node


def load_urdf(path: Path | str, root: str | None = None) -> KinematicTree:
    """Build a ``KinematicTree`` from a URDF file.

    URDF links become nodes (carrying their visual and collision geometry) and
    joints become edges. ``revolute`` and ``prismatic`` joints carry their
    position limits; ``continuous`` maps to an unlimited revolute joint;
    ``fixed`` becomes a zero-DOF joint. Each joint's ``origin`` is taken from the
    URDF joint frame. ``planar`` and ``floating`` URDF joints are not supported.

    ``root`` defaults to the URDF's base link.
    """
    handler = _make_filename_handler(path)
    urdf = yourdfpy.URDF.load(
        str(path),
        load_meshes=False,
        build_scene_graph=True,
        build_collision_scene_graph=False,
        filename_handler=handler,
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
    for link_name, link in urdf.link_map.items():
        node = tree.nodes[link_name]
        node.visuals = [_convert_geometry(v, handler) for v in link.visuals]
        node.collisions = [_convert_geometry(c, handler) for c in link.collisions]
    return tree


def _make_filename_handler(path: Path | str) -> Callable[[str], str]:
    directory = os.path.dirname(os.path.abspath(str(path)))
    prefix = "package://"

    def handler(filename: str) -> str:
        if filename.startswith(prefix):
            return os.path.join(directory, filename[len(prefix) :])
        if os.path.isabs(filename):
            return filename
        return os.path.join(directory, filename)

    return handler


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


def _convert_geometry(element: yourdfpy.Visual, handler: Callable[[str], str]) -> Shape:
    origin = SE3(element.origin, check=False) if element.origin is not None else SE3()
    geometry = element.geometry
    if geometry.box is not None:
        box = geometry.box.size
        size = (float(box[0]), float(box[1]), float(box[2]))
        return BoxShape(size=size, origin=origin)
    if geometry.cylinder is not None:
        return CylinderShape(
            radius=float(geometry.cylinder.radius),
            length=float(geometry.cylinder.length),
            origin=origin,
        )
    if geometry.sphere is not None:
        return SphereShape(radius=float(geometry.sphere.radius), origin=origin)
    if geometry.mesh is not None:
        raw = geometry.mesh.scale
        scale = (
            (1.0, 1.0, 1.0)
            if raw is None
            else (float(raw[0]), float(raw[1]), float(raw[2]))
        )
        return MeshShape(
            filename=handler(geometry.mesh.filename), scale=scale, origin=origin
        )
    raise ValueError("URDF geometry has no recognized shape")
