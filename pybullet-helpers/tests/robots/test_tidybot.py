"""Tests for tidybot robots."""

from pybullet_helpers.geometry import Pose
from pybullet_helpers.robots.tidybot import TidyBotMobileBase


def test_tidybot_mobile_base(physics_client_id):
    """Tests for TidyBotMobileBase()."""

    robot = TidyBotMobileBase(
        physics_client_id,
        z=0.0,
    )
    assert robot.get_name() == "tidybot-base"

    # import pybullet as p
    # while True:
    #     p.getMouseEvents(physics_client_id)
