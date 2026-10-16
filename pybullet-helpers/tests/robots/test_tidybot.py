"""Tests for tidybot robots."""

from pathlib import Path

import pybullet as p

from pybullet_helpers.geometry import SE2Pose
from pybullet_helpers.inverse_kinematics import check_body_collisions
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.kinova import (
    KinovaGen3RobotiqGripperPyBulletRobot,
    KinovaGen3VisionModuleRobotiqGripperPyBulletRobot,
)
from pybullet_helpers.robots.tidybot import TidyBotKinova, TidyBotMobileBase


def _collision_meshes(robot, link_name: str) -> list[str]:
    """The mesh file stems of `link_name`'s collision shapes ("" for a primitive)."""
    shapes = p.getCollisionShapeData(
        robot.robot_id,
        robot.link_from_name(link_name),
        physicsClientId=robot.physics_client_id,
    )
    return [Path(shape[4].decode()).stem for shape in shapes]


def test_tidybot_mobile_base(physics_client_id):
    """Tests for TidyBotMobileBase()."""
    robot = TidyBotMobileBase(
        physics_client_id,
        z=0.0,
    )
    assert robot.get_name() == "tidybot-base"


def test_tidybot_kinova(physics_client_id):
    """Tests for TidyBotKinova()."""
    robot = TidyBotKinova(
        physics_client_id,
        base_z=0.0,
    )
    assert robot.get_name() == "tidybot-kinova"

    # The TidyBot's camera is the Kinova vision module in the bracelet, and it has no
    # RealSense on the gripper, so its arm is the vision-module variant.
    assert isinstance(robot.arm, KinovaGen3VisionModuleRobotiqGripperPyBulletRobot)
    assert _collision_meshes(robot.arm, "bracelet_link") == [
        "bracelet_with_vision_link"
    ]
    assert _collision_meshes(robot.arm, "robotiq_arg2f_base_link") == [
        "robotiq_arg2f_base_link"
    ]


def test_stock_kinova_gen3_keeps_its_wrist(physics_client_id):
    """Moving the TidyBot to the vision variant leaves "kinova-gen3" as it was."""
    robot = create_pybullet_robot("kinova-gen3", physics_client_id)
    assert isinstance(robot, KinovaGen3RobotiqGripperPyBulletRobot)
    assert not isinstance(robot, KinovaGen3VisionModuleRobotiqGripperPyBulletRobot)
    assert _collision_meshes(robot, "bracelet_link") == ["bracelet_no_vision_link"]
    # The gripper's own mesh plus the RealSense bounding box (a primitive).
    assert len(_collision_meshes(robot, "robotiq_arg2f_base_link")) == 2

    # Uncomment to debug.
    # import pybullet as p
    # while True:
    #     p.getMouseEvents(physics_client_id)


def test_tidybot_mobile_base_collision(physics_client_id):
    """Test collision checking with the mobile base."""
    robot = TidyBotMobileBase(
        physics_client_id,
        z=0.0,
    )

    obstacle_collision_id = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[0.3, 0.3, 0.5],
        physicsClientId=physics_client_id,
    )
    obstacle_mass = 0
    obstacle_position = (1.0, 0.0, 0.0)
    obstacle_id = p.createMultiBody(
        obstacle_mass,
        obstacle_collision_id,
        basePosition=obstacle_position,
        physicsClientId=physics_client_id,
    )

    robot.set_pose(SE2Pose(0.0, 0.0, 0.0))
    assert not check_body_collisions(
        robot.robot_id,
        obstacle_id,
        physics_client_id,
    )

    robot.set_pose(SE2Pose(1.0, 0.0, 0.0))
    assert check_body_collisions(
        robot.robot_id,
        obstacle_id,
        physics_client_id,
    )
