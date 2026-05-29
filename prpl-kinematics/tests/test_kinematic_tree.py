"""Unit tests for the KinematicTree."""

import math

import numpy as np
import pytest
from spatialmath import SE3

from prpl_kinematics.tree import (
    Edge,
    FixedJoint,
    KinematicTree,
    Node,
    RevoluteJoint,
)


def _two_link_tree() -> KinematicTree:
    """World -[revolute j1]-> link1 -[fixed +x]-> link2."""
    tree = KinematicTree()
    tree.add_node(Node("link1"))
    tree.add_node(Node("link2"))
    tree.add_edge(Edge("world", "link1", RevoluteJoint(name="j1")))
    tree.add_edge(Edge("link1", "link2", FixedJoint(name="mount", origin=SE3(1, 0, 0))))
    return tree


def test_forward_kinematics_default_config():
    """Unspecified joints default to zero, so link2 sits at its origin offset."""
    tree = _two_link_tree()
    pose = tree.forward_kinematics("link2", {})
    assert np.allclose(pose.t, [1.0, 0.0, 0.0])


def test_forward_kinematics_propagates_rotation():
    """Rotating j1 by 90 degrees swings link2's +x offset onto +y."""
    tree = _two_link_tree()
    pose = tree.forward_kinematics("link2", {"j1": [math.pi / 2]})
    assert np.allclose(pose.t, [0.0, 1.0, 0.0], atol=1e-6)


def test_actuated_joint_names_excludes_fixed():
    """Only joints with DOF are reported as actuated."""
    tree = _two_link_tree()
    assert tree.actuated_joint_names() == ["j1"]


def test_path_from_root():
    """The path lists edges root-to-leaf; the root itself has an empty path."""
    tree = _two_link_tree()
    assert [edge.child for edge in tree.path_from_root("link2")] == ["link1", "link2"]
    assert not tree.path_from_root("world")


def test_joint_lookup():
    """Joints are retrievable by name; unknown names raise KeyError."""
    tree = _two_link_tree()
    assert tree.joint("j1").num_dof == 1
    with pytest.raises(KeyError):
        tree.joint("missing")


def test_attach_makes_object_follow_new_parent():
    """A grasped object's pose tracks the gripper frame times the grasp transform."""
    tree = _two_link_tree()
    tree.add_node(Node("mug"))
    tree.add_edge(Edge("world", "mug", FixedJoint(name="mug_free")))
    grasp = SE3(0.0, 0.0, 0.1)
    tree.attach("mug", "link2", grasp)
    config = {"j1": [math.pi / 2]}
    mug_pose = tree.forward_kinematics("mug", config)
    link2_pose = tree.forward_kinematics("link2", config)
    assert np.allclose(mug_pose.A, (link2_pose * grasp).A, atol=1e-6)


def test_relative_pose_recovers_grasp_transform():
    """The object's pose relative to its parent equals the attachment transform."""
    tree = _two_link_tree()
    tree.add_node(Node("mug"))
    tree.add_edge(Edge("world", "mug", FixedJoint(name="mug_free")))
    grasp = SE3(0.0, 0.0, 0.1)
    tree.attach("mug", "link2", grasp)
    relative = tree.relative_pose("link2", "mug", {"j1": [0.3]})
    assert np.allclose(relative.A, grasp.A, atol=1e-6)


def test_add_edge_unknown_parent_raises():
    """Adding an edge to a missing parent is an error."""
    tree = KinematicTree()
    tree.add_node(Node("a"))
    with pytest.raises(ValueError):
        tree.add_edge(Edge("ghost", "a", FixedJoint(name="x")))


def test_add_duplicate_node_raises():
    """Registering the same node name twice is an error."""
    tree = KinematicTree()
    tree.add_node(Node("a"))
    with pytest.raises(ValueError):
        tree.add_node(Node("a"))


def test_add_second_parent_raises():
    """A node may have only one incoming edge."""
    tree = _two_link_tree()
    with pytest.raises(ValueError):
        tree.add_edge(Edge("world", "link2", FixedJoint(name="dup")))
