"""Tests for tidybot robots."""

import pybullet as p

from pybullet_helpers.geometry import SE2Pose
from pybullet_helpers.inverse_kinematics import check_body_collisions
from pybullet_helpers.robots.tidybot import TidyBotKinova, TidyBotMobileBase


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
