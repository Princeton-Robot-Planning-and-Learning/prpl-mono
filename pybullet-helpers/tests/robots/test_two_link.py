"""Tests for two-link robot."""

import numpy as np

from pybullet_helpers.geometry import Pose
from pybullet_helpers.robots.two_link import (
    TwoLinkPyBulletRobot,
)


def test_two_link_pybullet_robot(physics_client_id):
    """Tests for TwoLinkPyBulletRobot()."""
    robot = TwoLinkPyBulletRobot(
        physics_client_id,
    )
    assert robot.get_name() == "two-link"
    assert robot.arm_joint_names == [
        "joint1",
        "joint2",
    ]
    assert np.allclose(robot.action_space.low, robot.joint_lower_limits)
    assert np.allclose(robot.action_space.high, robot.joint_upper_limits)

    ee_pose = robot.get_end_effector_pose()
    assert ee_pose.allclose(
        Pose(
            position=(2.0, 0.0, 0.0),
            orientation=(0.5, 0.5, 0.5, 0.5),
        )
    )
