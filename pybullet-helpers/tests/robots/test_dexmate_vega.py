"""Tests for the Dexmate Vega robot."""

import numpy as np

from pybullet_helpers.robots.dexmate_vega import DexmateVega1UPyBulletRobot


def test_dexmate_vega_1u_robot(physics_client_id):
    """Tests for DexmateVega1UPyBulletRobot."""
    robot = DexmateVega1UPyBulletRobot(physics_client_id)
    assert robot.get_name() == "dexmate-vega-1u"
    assert robot.arm_joint_names == [
        "Lift",
        "torso_flip",
        "L_arm_j1",
        "L_arm_j2",
        "L_arm_j3",
        "L_arm_j4",
        "L_arm_j5",
        "L_arm_j6",
        "L_arm_j7",
    ]
    assert np.allclose(robot.action_space.low, robot.joint_lower_limits)
    assert np.allclose(robot.action_space.high, robot.joint_upper_limits)
    # Moving each joint to its midpoint produces an EE pose within reach
    # (forward_kinematics doesn't raise).
    for i in range(len(robot.arm_joints)):
        q = list(robot.home_joint_positions)
        q[i] = 0.5 * (robot.joint_lower_limits[i] + robot.joint_upper_limits[i])
        robot.forward_kinematics(q)
