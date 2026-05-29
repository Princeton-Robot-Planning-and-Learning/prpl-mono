"""Unit tests for joint types."""

import math

import numpy as np
from spatialmath import SE3

from prpl_kinematics.tree.joints import (
    FixedJoint,
    PlanarJoint,
    PrismaticJoint,
    RevoluteJoint,
)


def test_fixed_joint_ignores_values():
    """A fixed joint has zero DOF and always returns its origin."""
    origin = SE3(1.0, 2.0, 3.0)
    joint = FixedJoint(name="mount", origin=origin)
    assert joint.num_dof == 0
    assert np.allclose(joint.transform([]).A, origin.A)
    assert not joint.lower_limits
    assert not joint.upper_limits


def test_revolute_rotates_about_axis():
    """A revolute joint rotates about its axis by the given angle."""
    joint = RevoluteJoint(name="r")  # Default axis is +z.
    pose = joint.transform([math.pi / 2])
    assert joint.num_dof == 1
    assert np.allclose(pose.R @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-6)
    assert np.allclose(joint.lower_limits, [-math.pi])
    assert np.allclose(joint.upper_limits, [math.pi])


def test_revolute_composes_with_origin():
    """A revolute joint applies its rotation on top of its origin offset."""
    joint = RevoluteJoint(name="r", origin=SE3(0.0, 0.0, 1.0))
    pose = joint.transform([0.0])
    assert np.allclose(pose.t, [0.0, 0.0, 1.0])


def test_prismatic_translates_along_axis():
    """A prismatic joint translates along its axis with its limits."""
    joint = PrismaticJoint(name="p", axis=(1.0, 0.0, 0.0), lower=0.0, upper=2.0)
    pose = joint.transform([0.5])
    assert joint.num_dof == 1
    assert np.allclose(pose.t, [0.5, 0.0, 0.0])
    assert np.allclose(joint.lower_limits, [0.0])
    assert np.allclose(joint.upper_limits, [2.0])


def test_planar_joint_is_se2():
    """A planar joint applies an (x, y, yaw) SE(2) transform."""
    joint = PlanarJoint(name="base")
    pose = joint.transform([1.0, 2.0, math.pi / 2])
    assert joint.num_dof == 3
    assert np.allclose(pose.t, [1.0, 2.0, 0.0])
    assert np.allclose(pose.R @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-6)
