"""Tests for Kinova robots."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pybullet as p
import pytest

from pybullet_helpers.geometry import Pose, multiply_poses, set_pose
from pybullet_helpers.inverse_kinematics import (
    check_collisions_with_held_object,
    inverse_kinematics,
)
from pybullet_helpers.link import get_link_pose
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.kinova import (
    KinovaGen3NoGripperPyBulletRobot,
    KinovaGen3RobotiqGripperPyBulletRobot,
    KinovaGen3VisionModuleRobotiqGripperPyBulletRobot,
)
from pybullet_helpers.utils import create_pybullet_block


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


def _wrist_probe_collides(
    robot: KinovaGen3RobotiqGripperPyBulletRobot, link_name: str, offset: Pose
) -> bool:
    """Whether a 4 mm cube at `offset` in `link_name`'s frame touches the robot.

    The robot is at its home configuration, which is free of self-collisions.
    """
    robot.set_joints(robot.home_joint_positions)
    link_pose = get_link_pose(
        robot.robot_id, robot.link_from_name(link_name), robot.physics_client_id
    )
    probe = create_pybullet_block(
        (1.0, 0.0, 0.0, 1.0), (0.002, 0.002, 0.002), robot.physics_client_id
    )
    set_pose(probe, multiply_poses(link_pose, offset), robot.physics_client_id)
    return check_collisions_with_held_object(
        robot,
        collision_bodies={probe},
        physics_client_id=robot.physics_client_id,
        held_object=None,
        base_link_to_held_obj=None,
        joint_state=robot.home_joint_positions,
    )


# Inside the vision module's housing, 2.75 cm beyond the bare bracelet's -y extent.
VISION_HOUSING_OFFSET = Pose((0.0, -0.065, -0.04))
# The centre of the gripper-mounted RealSense bounding box in gen3_7dof.urdf.
REALSENSE_BOX_OFFSET = Pose((0.05, 0.0, 0.02))


def test_kinova_gen3_vision_module_pybullet_robot(physics_client_id):
    """Tests for KinovaGen3VisionModuleRobotiqGripperPyBulletRobot()."""
    base_pose = Pose((0.75, 0.7441, 0.0))
    robot = create_pybullet_robot(
        "kinova-gen3-vision", physics_client_id, base_pose=base_pose
    )
    assert isinstance(robot, KinovaGen3VisionModuleRobotiqGripperPyBulletRobot)
    assert robot.get_name() == "kinova-gen3-vision"

    # Only the wrist geometry differs: the kinematics are the stock arm's.
    reference_client_id = p.connect(p.DIRECT)
    try:
        reference = KinovaGen3RobotiqGripperPyBulletRobot(
            reference_client_id, base_pose=base_pose
        )
        assert robot.arm_joint_names == reference.arm_joint_names
        assert np.allclose(robot.joint_lower_limits, reference.joint_lower_limits)
        assert np.allclose(robot.joint_upper_limits, reference.joint_upper_limits)
        rng = np.random.default_rng(0)
        # joint_7 is continuous, so its limits are infinite.
        low = np.clip(reference.joint_lower_limits, -np.pi, np.pi)
        high = np.clip(reference.joint_upper_limits, -np.pi, np.pi)
        for _ in range(5):
            joints = list(rng.uniform(low, high))
            assert robot.forward_kinematics(joints).allclose(
                reference.forward_kinematics(joints), atol=1e-6
            )
    finally:
        p.disconnect(reference_client_id)

    # The home configuration stays free of self-collisions with the larger bracelet.
    assert not check_collisions_with_held_object(
        robot,
        collision_bodies=set(),
        physics_client_id=robot.physics_client_id,
        held_object=None,
        base_link_to_held_obj=None,
        joint_state=robot.home_joint_positions,
    )


def test_kinova_gen3_vision_module_collides_with_the_camera_housing(
    physics_client_id,
):
    """The vision module is solid; the stock bracelet passes through it."""
    stock = KinovaGen3RobotiqGripperPyBulletRobot(physics_client_id)
    assert not _wrist_probe_collides(stock, "bracelet_link", VISION_HOUSING_OFFSET)

    vision_client_id = p.connect(p.DIRECT)
    try:
        vision = KinovaGen3VisionModuleRobotiqGripperPyBulletRobot(vision_client_id)
        assert _wrist_probe_collides(vision, "bracelet_link", VISION_HOUSING_OFFSET)
    finally:
        p.disconnect(vision_client_id)


def test_kinova_gen3_vision_module_has_no_realsense_box(physics_client_id):
    """The stock model keeps its RealSense box; the vision-module model has none."""
    stock = KinovaGen3RobotiqGripperPyBulletRobot(physics_client_id)
    assert _wrist_probe_collides(stock, "robotiq_arg2f_base_link", REALSENSE_BOX_OFFSET)

    vision_client_id = p.connect(p.DIRECT)
    try:
        vision = KinovaGen3VisionModuleRobotiqGripperPyBulletRobot(vision_client_id)
        assert not _wrist_probe_collides(
            vision, "robotiq_arg2f_base_link", REALSENSE_BOX_OFFSET
        )
    finally:
        p.disconnect(vision_client_id)
