"""Joints: parameterized transforms between a parent frame and a child frame.

Every edge in a :class:`~prpl_kinematics.tree.kinematic_tree.KinematicTree` carries
a joint. A joint maps a vector of joint values (length ``num_dof``) to the rigid
transform from its parent frame to its child frame. Robot actuators, a mobile
base, and even a rigid grasp are all just joints:

* ``FixedJoint`` (0 DOF) -- rigid attachment, e.g. a grasp or a sensor mount.
* ``RevoluteJoint`` (1 DOF) -- a rotary actuator.
* ``PrismaticJoint`` (1 DOF) -- a linear actuator.
* ``PlanarJoint`` (3 DOF) -- an SE(2) mobile base ``(x, y, yaw)``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from spatialmath import SE3

JointValues = list[float]  # Length must equal ``num_dof``.

Axis = tuple[float, float, float]


@dataclass(frozen=True)
class Joint(ABC):
    """A parameterized parent-to-child transform.

    ``origin`` is the parent-to-joint transform at zero configuration (the URDF
    joint origin); each subclass applies its motion on top of ``origin``.
    """

    name: str
    origin: SE3 = field(default_factory=SE3)

    @property
    @abstractmethod
    def num_dof(self) -> int:
        """Number of scalar values this joint consumes."""

    @abstractmethod
    def transform(self, values: JointValues) -> SE3:
        """Parent-to-child transform for the given joint values."""

    @property
    def lower_limits(self) -> list[float]:
        """Lower bound for each DOF (``-inf`` if unbounded)."""
        return [-math.inf] * self.num_dof

    @property
    def upper_limits(self) -> list[float]:
        """Upper bound for each DOF (``+inf`` if unbounded)."""
        return [math.inf] * self.num_dof


@dataclass(frozen=True)
class FixedJoint(Joint):
    """A rigid (0-DOF) attachment; the transform is always ``origin``."""

    @property
    def num_dof(self) -> int:
        return 0

    def transform(self, values: JointValues) -> SE3:
        return self.origin


@dataclass(frozen=True)
class RevoluteJoint(Joint):
    """A 1-DOF rotary joint about ``axis`` with position limits."""

    axis: Axis = (0.0, 0.0, 1.0)
    lower: float = -math.pi
    upper: float = math.pi

    @property
    def num_dof(self) -> int:
        return 1

    def transform(self, values: JointValues) -> SE3:
        (theta,) = values
        return self.origin * SE3.AngVec(theta, self.axis)

    @property
    def lower_limits(self) -> list[float]:
        return [self.lower]

    @property
    def upper_limits(self) -> list[float]:
        return [self.upper]


@dataclass(frozen=True)
class PrismaticJoint(Joint):
    """A 1-DOF linear joint along ``axis`` with position limits."""

    axis: Axis = (0.0, 0.0, 1.0)
    lower: float = 0.0
    upper: float = 1.0

    @property
    def num_dof(self) -> int:
        return 1

    def transform(self, values: JointValues) -> SE3:
        (d,) = values
        return self.origin * SE3(d * self.axis[0], d * self.axis[1], d * self.axis[2])

    @property
    def lower_limits(self) -> list[float]:
        return [self.lower]

    @property
    def upper_limits(self) -> list[float]:
        return [self.upper]


@dataclass(frozen=True)
class PlanarJoint(Joint):
    """A 3-DOF SE(2) joint ``(x, y, yaw)`` -- e.g. a mobile base."""

    @property
    def num_dof(self) -> int:
        return 3

    def transform(self, values: JointValues) -> SE3:
        x, y, yaw = values
        return self.origin * SE3(x, y, 0.0) * SE3.Rz(yaw)
