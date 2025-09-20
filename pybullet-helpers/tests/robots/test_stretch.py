"""Tests for Stretch robot."""

import numpy as np
import pybullet as p

from pybullet_helpers.geometry import Pose
from pybullet_helpers.inverse_kinematics import inverse_kinematics
from pybullet_helpers.robots.stretch import StretchPyBulletRobot


def test_stretch_pybullet_robot(physics_client_id):
    """Tests for StretchPyBulletRobot()."""
    robot = StretchPyBulletRobot(physics_client_id)
    assert robot.get_name() == "stretch"
    assert not robot.fixed_base
    assert len(robot.default_home_joint_positions) == 10

    # Test inverse kinematics. Cannot currently do better than atol=1e-1.
    current_ee_pose = robot.get_end_effector_pose()
    positions = [
        np.add(current_ee_pose.position, (0, 0, 0.0)),
        np.add(current_ee_pose.position, (0, 0, -0.2)),
        np.add(current_ee_pose.position, (0, 0, -0.4)),
    ]
    orientations = [
        p.getQuaternionFromEuler((np.pi / 2, 0, np.pi / 4)),
        p.getQuaternionFromEuler((np.pi / 2, 0, 0)),
        p.getQuaternionFromEuler((np.pi / 2, 0, -np.pi / 4)),
        p.getQuaternionFromEuler((0, 0, np.pi / 2)),
        p.getQuaternionFromEuler((0, np.pi / 2, 0)),
    ]
    for position in positions:
        for orientation in orientations:
            pose = Pose(position, orientation)
            joint_positions = inverse_kinematics(robot, pose)
            robot.set_joints(joint_positions)
            recovered_pose = robot.get_end_effector_pose()
            assert pose.allclose(recovered_pose, atol=1e-1)

    robot.action_space.seed(123)
    init_joints = robot.get_joint_positions()
    for _ in range(10):
        unknown_joints = robot.action_space.sample()
        robot.set_joints(unknown_joints)
        target_pose = robot.get_end_effector_pose()
        robot.set_joints(init_joints)
        joint_positions = inverse_kinematics(robot, target_pose)
        robot.set_joints(joint_positions)
        recovered_pose = robot.get_end_effector_pose()
        assert target_pose.allclose(recovered_pose, atol=1e-1)
