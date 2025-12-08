"""Tests for tidybot robots."""

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
