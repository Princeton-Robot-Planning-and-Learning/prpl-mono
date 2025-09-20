"""Tests for Human()."""

import numpy as np

from pybullet_helpers.geometry import Pose
from pybullet_helpers.inverse_kinematics import inverse_kinematics
from pybullet_helpers.math_utils import sample_within_sphere
from pybullet_helpers.robots.human import Human


def test_human(physics_client_id):
    """Tests for Human()."""

    right_leg_kwargs = {
        "home_joint_positions": [-np.pi / 2, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    left_leg_kwargs = {
        "home_joint_positions": [-np.pi / 2, 0.0, 0.0, 0.0, 0.0, 0.0],
    }

    base_pose = Pose.from_rpy((0.1, 0.0, 0.0), (0.0, 0.0, np.pi / 2))
    human = Human(
        physics_client_id,
        base_pose=base_pose,
        right_leg_kwargs=right_leg_kwargs,
        left_leg_kwargs=left_leg_kwargs,
    )

    # Test IK for the right arm.
    right_arm = human.right_arm

    # We want it to be possible to reach positions in a small sphere around
    # the end effector.
    resting_pose = right_arm.get_end_effector_pose()
    center = resting_pose.position
    orientation = resting_pose.orientation
    radius = 0.05
    rng = np.random.default_rng(123)
    for _ in range(5):
        position = sample_within_sphere(center, 0.0, radius, rng)
        pose = Pose(position, orientation)
        joints = inverse_kinematics(right_arm, pose)
        right_arm.set_joints(joints)
        assert right_arm.get_end_effector_pose().allclose(pose, atol=1e-3)
