"""Tests for Kinova robots."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pybullet as p
import pytest

from pybullet_helpers.geometry import Pose
from pybullet_helpers.inverse_kinematics import (
    check_collisions_with_held_object,
    inverse_kinematics,
)
from pybullet_helpers.robots.kinova import (
    KinovaGen3NoGripperPyBulletRobot,
    KinovaGen3RobotiqGripperPyBulletRobot,
)


def test_kinova_gen3_no_gripper_pybullet_robot(physics_client_id):
    """Tests for KinovaGen3NoGripperPyBulletRobot()."""
    base_pose = Pose((0.75, 0.7441, 0.0))
    robot = KinovaGen3NoGripperPyBulletRobot(
        physics_client_id,
        base_pose=base_pose,
    )
    assert robot.get_name() == "kinova-gen3-no-gripper"
    assert robot.arm_joint_names == [
        "Actuator1",
        "Actuator2",
        "Actuator3",
        "Actuator4",
        "Actuator5",
        "Actuator6",
        "Actuator7",
    ]
    assert np.allclose(robot.action_space.low, robot.joint_lower_limits)
    assert np.allclose(robot.action_space.high, robot.joint_upper_limits)

    ee_pose = robot.get_end_effector_pose()
    ee_target_position = np.add(ee_pose.position, (0.05, -0.1, 0.05))
    ee_orn = ee_pose.orientation
    ee_target = Pose(ee_target_position, ee_orn)
    joint_target = inverse_kinematics(robot, ee_target, validate=False)
    action_arr = np.array(joint_target, dtype=np.float32)

    # Not a valid control mode.
    robot._control_mode = "not a real control mode"  # pylint: disable=protected-access
    with pytest.raises(NotImplementedError) as e:
        robot.set_motors(action_arr)
    assert "Unrecognized pybullet_control_mode" in str(e)

    # Reset control mode.
    robot._control_mode = "reset"  # pylint: disable=protected-access
    robot.set_motors(action_arr)  # just make sure it doesn't crash

    # Position control mode.
    robot._pybullet_control_mode = "position"  # pylint: disable=protected-access
    robot.set_motors(action_arr)
    for _ in range(20):
        p.stepSimulation(physicsClientId=physics_client_id)
    recovered_ee_pos = robot.get_end_effector_pose().position

    # IK is currently not precise enough to increase this tolerance.
    assert np.allclose(ee_target_position, recovered_ee_pos, atol=1e-2)
    # Test forward kinematics.
    fk_result = robot.forward_kinematics(action_arr)
    assert np.allclose(fk_result.position, ee_target_position, atol=1e-2)

    # Test creating a robot with a custom URDF file. Note that the file needs
    # to be in the same directory as the original URDF file for rel paths.
    with tempfile.NamedTemporaryFile(
        dir=robot.urdf_path.parent, suffix=".urdf"
    ) as temp_urdf:
        shutil.copy2(robot.urdf_path, temp_urdf.name)
        custom_urdf_robot = KinovaGen3NoGripperPyBulletRobot(
            physics_client_id,
            base_pose=base_pose,
            custom_urdf_path=Path(temp_urdf.name),
        )
        # Kinematics should be the same as before.
        custom_fk_result = custom_urdf_robot.forward_kinematics(action_arr)
        assert custom_fk_result.allclose(fk_result, atol=1e-5)


def test_kinova_gen3_robotiq_gripper_pybullet_robot(physics_client_id):
    """Tests for KinovaGen3RobotiqGripperPyBulletRobot()."""
    base_pose = Pose((0.75, 0.7441, 0.0))
    robot = KinovaGen3RobotiqGripperPyBulletRobot(
        physics_client_id,
        base_pose=base_pose,
    )
    assert robot.get_name() == "kinova-gen3"
    assert robot.arm_joint_names == [
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
        "joint_7",
        "finger_joint",
        "right_outer_knuckle_joint",
        "left_inner_knuckle_joint",
        "right_inner_knuckle_joint",
        "left_inner_finger_joint",
        "right_inner_finger_joint",
    ]
    assert np.allclose(robot.action_space.low, robot.joint_lower_limits)
    assert np.allclose(robot.action_space.high, robot.joint_upper_limits)

    ee_pose = robot.get_end_effector_pose()
    ee_target_position = np.add(ee_pose.position, (0.05, -0.1, 0.05))
    ee_orn = ee_pose.orientation
    ee_target = Pose(ee_target_position, ee_orn)
    joint_target = inverse_kinematics(robot, ee_target, validate=False)
    finger_state = 0.03
    finger_values = robot.finger_state_to_joints(finger_state)
    assert np.allclose(
        finger_values,
        [
            finger_state,
            finger_state,
            finger_state,
            finger_state,
            -finger_state,
            -finger_state,
        ],
    )
    for idx, value in zip(robot.finger_joint_idxs, finger_values, strict=True):
        joint_target[idx] = value
    action_arr = np.array(joint_target, dtype=np.float32)

    # Not a valid control mode.
    robot._control_mode = "not a real control mode"  # pylint: disable=protected-access
    with pytest.raises(NotImplementedError) as e:
        robot.set_motors(action_arr)
    assert "Unrecognized pybullet_control_mode" in str(e)

    # Reset control mode.
    robot._control_mode = "reset"  # pylint: disable=protected-access
    robot.set_motors(action_arr)  # just make sure it doesn't crash
    recovered_ee_pos = robot.get_end_effector_pose().position

    # IK is currently not precise enough to increase this tolerance.
    assert np.allclose(ee_target_position, recovered_ee_pos, atol=1e-2)
    # Test forward kinematics.
    fk_result = robot.forward_kinematics(action_arr)
    assert np.allclose(fk_result.position, ee_target_position, atol=1e-2)

    # Test joint limits.
    oob_joint_state = [-100] * len(robot.home_joint_positions)
    assert not robot.check_joint_limits(oob_joint_state)

    # Test self-collisions.
    assert not check_collisions_with_held_object(
        robot,
        collision_bodies=set(),
        physics_client_id=robot.physics_client_id,
        held_object=None,
        base_link_to_held_obj=None,
        joint_state=robot.home_joint_positions,
    )

    collision_joint_state = [
        -0.105,
        -0.761,
        -4.105,
        -2.352,
        0.211,
        -1.948,
        2.421,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert robot.check_joint_limits(collision_joint_state)
    assert check_collisions_with_held_object(
        robot,
        collision_bodies=set(),
        physics_client_id=robot.physics_client_id,
        held_object=None,
        base_link_to_held_obj=None,
        joint_state=collision_joint_state,
    )
