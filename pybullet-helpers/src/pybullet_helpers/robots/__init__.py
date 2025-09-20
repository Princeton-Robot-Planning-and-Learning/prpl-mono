"""Create PyBullet robots."""

from typing import Type

from pybullet_helpers.robots.assistive_human import AssistiveHumanPyBulletRobot
from pybullet_helpers.robots.fetch import FetchPyBulletRobot
from pybullet_helpers.robots.human import (
    LeftArmHumanPyBulletRobot,
    LeftLegHumanPyBulletRobot,
    RightArmHumanPyBulletRobot,
    RightLegHumanPyBulletRobot,
)
from pybullet_helpers.robots.kinova import (
    KinovaGen3NoGripperPyBulletRobot,
    KinovaGen3RobotiqGripperPyBulletRobot,
)
from pybullet_helpers.robots.panda import PandaPyBulletRobot
from pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot
from pybullet_helpers.robots.spot import SpotPyBulletRobot
from pybullet_helpers.robots.stretch import StretchPyBulletRobot
from pybullet_helpers.robots.two_link import TwoLinkPyBulletRobot

_BUILT_IN_ROBOT_CLASSES: list[Type[SingleArmPyBulletRobot]] = [
    FetchPyBulletRobot,
    PandaPyBulletRobot,
    KinovaGen3NoGripperPyBulletRobot,
    KinovaGen3RobotiqGripperPyBulletRobot,
    TwoLinkPyBulletRobot,
    StretchPyBulletRobot,
    AssistiveHumanPyBulletRobot,
    RightArmHumanPyBulletRobot,
    LeftArmHumanPyBulletRobot,
    RightLegHumanPyBulletRobot,
    LeftLegHumanPyBulletRobot,
    SpotPyBulletRobot,
]


def create_pybullet_robot(
    robot_name: str, physics_client_id: int, *args, **kwargs
) -> SingleArmPyBulletRobot:
    """Create a known PyBullet robot from its name."""
    for cls in _BUILT_IN_ROBOT_CLASSES:
        if robot_name == cls.get_name():
            return cls(physics_client_id, *args, **kwargs)
    raise NotImplementedError(f"Unknown robot {robot_name}")
