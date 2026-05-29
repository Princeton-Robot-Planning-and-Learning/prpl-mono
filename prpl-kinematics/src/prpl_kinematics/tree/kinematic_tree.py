"""The KinematicTree: one scene graph in which everything is an edge.

A ``KinematicTree`` is a directed tree of named frames (``Node``) rooted at a
single node (conventionally ``"world"``). Each non-root node has exactly one
incoming ``Edge`` carrying a :class:`~prpl_kinematics.tree.joints.Joint`. The
union of all joints with ``num_dof > 0`` is the tree's configuration.

This single structure unifies what other libraries treat as separate cases:

* A robot arm is a chain of revolute/prismatic edges.
* A mobile base is a ``PlanarJoint`` edge from ``world`` to the base frame.
* A grasp is :meth:`attach` -- re-parenting an object's edge onto a gripper
  frame with a ``FixedJoint``. Release is just ``attach`` back onto ``world``.
* A bimanual robot is one tree with two arm chains; multiple robots and free
  objects all live under ``world``.

Forward kinematics composes the joint transforms along the path from the root.
The tree owns no physics: it computes frame poses but performs no collision
checking or rendering itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from spatialmath import SE3

from prpl_kinematics.tree.joints import FixedJoint, Joint, JointValues

Configuration = Mapping[str, JointValues]


@dataclass
class Node:
    """A named frame, optionally carrying collision/visual geometry.

    ``geometry`` is an opaque handle (e.g. a path to a mesh or a primitive-shape
    spec). The tree never interprets it; only consumers that render or check
    collisions do.
    """

    name: str
    geometry: object | None = None


@dataclass
class Edge:
    """A directed parent-to-child connection carrying a joint."""

    parent: str
    child: str
    joint: Joint


class KinematicTree:
    """A directed tree of frames connected by joints, rooted at ``root``."""

    def __init__(self, root: str = "world") -> None:
        self._root = root
        self._nodes: dict[str, Node] = {root: Node(root)}
        # Each non-root child has exactly one incoming edge, keyed by child name.
        self._edges: dict[str, Edge] = {}

    @property
    def root(self) -> str:
        """The name of the root frame."""
        return self._root

    @property
    def nodes(self) -> Mapping[str, Node]:
        """All frames, keyed by name."""
        return self._nodes

    def add_node(self, node: Node) -> None:
        """Register a frame.

        Raises if the name is already present.
        """
        if node.name in self._nodes:
            raise ValueError(f"Node already exists: {node.name}")
        self._nodes[node.name] = node

    def add_edge(self, edge: Edge) -> None:
        """Connect ``edge.child`` to ``edge.parent`` via ``edge.joint``."""
        if edge.parent not in self._nodes:
            raise ValueError(f"Unknown parent: {edge.parent}")
        if edge.child not in self._nodes:
            raise ValueError(f"Unknown child: {edge.child}")
        if edge.child in self._edges:
            raise ValueError(f"Node already has a parent: {edge.child}")
        self._edges[edge.child] = edge

    def path_from_root(self, name: str) -> list[Edge]:
        """Edges from the root down to ``name`` (empty for the root itself)."""
        if name not in self._nodes:
            raise ValueError(f"Unknown node: {name}")
        edges: list[Edge] = []
        cursor = name
        while cursor != self._root:
            edge = self._edges[cursor]
            edges.append(edge)
            cursor = edge.parent
        edges.reverse()
        return edges

    def actuated_joint_names(self) -> list[str]:
        """Names of all joints with at least one DOF, in insertion order."""
        return [e.joint.name for e in self._edges.values() if e.joint.num_dof > 0]

    def joint(self, joint_name: str) -> Joint:
        """Look up a joint by name.

        Raises ``KeyError`` if absent.
        """
        for edge in self._edges.values():
            if edge.joint.name == joint_name:
                return edge.joint
        raise KeyError(f"No joint named {joint_name}")

    def forward_kinematics(self, name: str, config: Configuration) -> SE3:
        """World-frame pose of frame ``name`` under ``config``.

        Joints absent from ``config`` are taken at their zero values.
        """
        pose = SE3()
        for edge in self.path_from_root(name):
            joint = edge.joint
            values = list(config.get(joint.name, [0.0] * joint.num_dof))
            pose = pose * joint.transform(values)
        return pose

    def relative_pose(self, a: str, b: str, config: Configuration) -> SE3:
        """Pose of frame ``b`` expressed in frame ``a``."""
        return self.forward_kinematics(a, config).inv() * self.forward_kinematics(
            b, config
        )

    def attach(self, child: str, new_parent: str, transform: SE3) -> None:
        """Re-parent ``child`` onto ``new_parent`` via a fixed ``transform``.

        This is the grasp operation: the held object's frame becomes rigidly
        fixed to (e.g.) a gripper frame. The joint name is preserved if ``child``
        already had one so that snapshots stay stable.
        """
        if new_parent not in self._nodes:
            raise ValueError(f"Unknown parent: {new_parent}")
        existing = self._edges.get(child)
        joint_name = existing.joint.name if existing else f"{child}_attachment"
        self._edges[child] = Edge(
            parent=new_parent,
            child=child,
            joint=FixedJoint(name=joint_name, origin=transform),
        )
