"""Robots: compositions over a KinematicTree (named groups, EE, IK, home, ACM)."""

from prpl_kinematics.robots.kinova import make_kinova
from prpl_kinematics.robots.panda import make_panda
from prpl_kinematics.robots.robot import Manipulator, Robot
from prpl_kinematics.robots.tidybot import make_tidybot
from prpl_kinematics.robots.vega import make_vega

__all__ = [
    "Robot",
    "Manipulator",
    "make_panda",
    "make_kinova",
    "make_tidybot",
    "make_vega",
]
