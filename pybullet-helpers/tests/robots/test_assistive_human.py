"""Tests for assistive human robots."""

from pybullet_helpers.geometry import Pose
from pybullet_helpers.inverse_kinematics import (
    inverse_kinematics,
)
from pybullet_helpers.robots.assistive_human import AssistiveHumanPyBulletRobot


def test_assistive_human_pybullet_robot(physics_client_id):
    """Tests for AssistiveHumanPyBulletRobot()."""

    base_pose = Pose((0.0, 0.0, 0.0))
    robot = AssistiveHumanPyBulletRobot(
        physics_client_id,
        base_pose=base_pose,
    )
    assert robot.get_name() == "assistive-human"

    robot.action_space.seed(123)
    for _ in range(10):
        joint_positions = robot.action_space.sample()
        robot.set_joints(joint_positions)
        ee_target = robot.get_end_effector_pose()

        joint_target = inverse_kinematics(robot, ee_target, validate=True)
        robot.set_joints(joint_target)
        assert robot.get_end_effector_pose().allclose(ee_target, atol=1e-3)
