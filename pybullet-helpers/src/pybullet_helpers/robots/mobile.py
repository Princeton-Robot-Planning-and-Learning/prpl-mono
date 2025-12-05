"""Base classes for mobile bases and mobile manipulators."""

import abc
from dataclasses import dataclass

from pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot


class MobileBase(abc.ABC):
    """Base class for a mobile base."""


@dataclass
class SingleArmMobileManipulator:
    """A single arm mounted on a mobile base."""

    arm: SingleArmPyBulletRobot
    base: MobileBase
