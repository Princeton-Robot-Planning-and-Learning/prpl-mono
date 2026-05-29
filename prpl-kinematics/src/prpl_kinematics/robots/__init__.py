"""Robots: compositions over a KinematicTree (named groups, EE, IK, home, ACM)."""

from prpl_kinematics.robots.kinova import make_kinova
from prpl_kinematics.robots.panda import make_panda
from prpl_kinematics.robots.robot import Robot

__all__ = ["Robot", "make_panda", "make_kinova"]
