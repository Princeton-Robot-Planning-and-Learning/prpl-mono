"""Unit tests for KinematicState snapshots."""

from prpl_kinematics.tree import (
    Edge,
    KinematicState,
    KinematicTree,
    Node,
    PlanarJoint,
    RevoluteJoint,
)


def _base_and_arm_tree() -> KinematicTree:
    """World -[planar base]-> base -[revolute arm]-> arm."""
    tree = KinematicTree()
    tree.add_node(Node("base"))
    tree.add_node(Node("arm"))
    tree.add_edge(Edge("world", "base", PlanarJoint(name="base_joint")))
    tree.add_edge(Edge("base", "arm", RevoluteJoint(name="arm_joint")))
    return tree


def test_state_captures_all_actuated_joints():
    """A snapshot records every actuated joint, preserving multi-DOF values."""
    tree = _base_and_arm_tree()
    config = {"base_joint": [1.0, 2.0, 0.5], "arm_joint": [0.3]}
    state = KinematicState.from_tree(tree, config)
    assert set(state.joint_values) == {"base_joint", "arm_joint"}
    assert state.joint_values["base_joint"] == (1.0, 2.0, 0.5)


def test_state_defaults_absent_joints_to_zero():
    """Joints missing from the config are snapshotted at their zero values."""
    tree = _base_and_arm_tree()
    state = KinematicState.from_tree(tree, {"arm_joint": [0.3]})
    assert state.joint_values["base_joint"] == (0.0, 0.0, 0.0)


def test_state_round_trips_to_configuration():
    """as_configuration reproduces a usable forward-kinematics config."""
    tree = _base_and_arm_tree()
    config = {"base_joint": [1.0, 2.0, 0.5], "arm_joint": [0.3]}
    state = KinematicState.from_tree(tree, config)
    assert state.as_configuration() == config
