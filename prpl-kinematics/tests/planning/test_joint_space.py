"""Unit tests for the JointSpace configuration space."""

import math

import numpy as np
import pytest
from spatialmath import SE3

from prpl_kinematics.geometry.shapes import BoxShape
from prpl_kinematics.planning import (
    ConfigurationSpace,
    JointSpace,
    SE2Space,
)
from prpl_kinematics.tree.joints import FixedJoint, PrismaticJoint, RevoluteJoint
from prpl_kinematics.tree.kinematic_tree import Edge, KinematicTree, Node


def _gantry_tree() -> KinematicTree:
    """An XY gantry: a small box robot that slides in x then y, plus a central
    block obstacle it must steer around."""
    tree = KinematicTree()
    tree.add_node(Node("jx"))
    tree.add_node(Node("robot", collisions=[BoxShape(size=(0.2, 0.2, 0.2))]))
    tree.add_node(Node("obstacle", collisions=[BoxShape(size=(2.0, 2.0, 2.0))]))
    tree.add_edge(
        Edge(
            "world",
            "jx",
            PrismaticJoint(name="jx_joint", axis=(1, 0, 0), lower=-1, upper=5),
        )
    )
    tree.add_edge(
        Edge(
            "jx",
            "robot",
            PrismaticJoint(name="jy_joint", axis=(0, 1, 0), lower=-1, upper=5),
        )
    )
    tree.add_edge(
        Edge("world", "obstacle", FixedJoint(name="ofix", origin=SE3(2.5, 2.5, 0)))
    )
    return tree


def test_joint_space_geometry():
    """A JointSpace samples within bounds and converts vectors round-trip."""
    space = JointSpace(_gantry_tree(), ["jx_joint", "jy_joint"])
    assert space.dimension == 2
    rng = np.random.default_rng(0)
    for _ in range(50):
        sample = space.sample(rng)
        assert np.all(sample >= -1) and np.all(sample <= 5)
    config = {"jx_joint": [1.5], "jy_joint": [-0.5]}
    assert space.to_configuration(space.to_vector(config)) == config
    assert space.distance(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == pytest.approx(
        5.0
    )
    assert np.allclose(space.clamp(np.array([-3.0, 7.0])), [-1.0, 5.0])


def _continuous_space() -> JointSpace:
    tree = KinematicTree()
    tree.add_node(Node("a"))
    tree.add_edge(
        Edge("world", "a", RevoluteJoint(name="cont", lower=-math.inf, upper=math.inf))
    )
    return JointSpace(tree, ["cont"])


def test_continuous_joint_distance_wraps_around():
    """A continuous joint measures the shorter way around 2*pi."""
    space = _continuous_space()
    assert space.distance(np.array([3.0]), np.array([-3.0])) == pytest.approx(
        2 * math.pi - 6.0
    )


def test_continuous_joint_samples_within_pi():
    """A continuous joint (infinite limits) samples over [-pi, pi]."""
    space = _continuous_space()
    rng = np.random.default_rng(0)
    for _ in range(50):
        value = space.sample(rng)[0]
        assert -math.pi <= value <= math.pi


def test_continuous_joint_interpolates_short_way():
    """Interpolation crosses the +-pi seam instead of unwinding the long way."""
    space = _continuous_space()
    waypoints = [
        w[0] for w in space.interpolate(np.array([3.0]), np.array([-3.0]), 0.1)
    ]
    steps = np.diff([3.0] + waypoints)
    assert np.all(np.abs(steps) <= 0.1 + 1e-9)
    assert (waypoints[-1] - (-3.0)) % (2 * math.pi) == pytest.approx(0.0, abs=1e-9)


def test_finite_revolute_distance_is_euclidean():
    """A limited revolute joint does not wrap; distance stays Euclidean."""
    tree = KinematicTree()
    tree.add_node(Node("a"))
    tree.add_edge(
        Edge("world", "a", RevoluteJoint(name="r", lower=-math.pi, upper=math.pi))
    )
    space = JointSpace(tree, ["r"])
    assert space.distance(np.array([3.0]), np.array([-3.0])) == pytest.approx(6.0)


def test_joint_and_se2_spaces_conform_to_protocol():
    """Both space types satisfy the ConfigurationSpace protocol."""
    assert isinstance(_continuous_space(), ConfigurationSpace)
    assert isinstance(SE2Space("base", (-1, 1), (-1, 1)), ConfigurationSpace)


def test_interpolate_resolution_and_endpoint():
    """Interpolation steps stay within resolution and end exactly at the target."""
    space = JointSpace(_gantry_tree(), ["jx_joint", "jy_joint"])
    a, b = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    waypoints = list(space.interpolate(a, b, resolution=0.1))
    assert len(waypoints) == 10
    assert np.allclose(waypoints[-1], b)
    steps = np.diff([a] + waypoints, axis=0)
    assert np.all(np.linalg.norm(steps, axis=1) <= 0.1 + 1e-9)
