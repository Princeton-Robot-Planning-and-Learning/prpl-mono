"""Data structures."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from prpl_utils.utils import wrap_angle
from tomsgeoms2d.structs import Geom2D


class ZOrder(Enum):
    """Used for collision checking."""

    FLOOR = 1  # collides with things on the floor
    SURFACE = 2  # collides with things at the table surface level
    ALL = 100  # collides with everyone (except NONE)
    NONE = 0  # collides with nothing, draw it first.


def z_orders_may_collide(z0: ZOrder, z1: ZOrder) -> bool:
    """Defines the semantics of ZOrder collisions."""
    if ZOrder.NONE in (z0, z1):
        return False
    if ZOrder.ALL in (z0, z1):
        return True
    return z0 == z1


@dataclass(frozen=True)
class Body2D:
    """A body consists a geom, a z order (for collision checks), and rendering kwargs
    (for visualization)."""

    geom: Geom2D
    z_order: ZOrder
    rendering_kwargs: dict
    name: str = "root"

    def plot(self, ax: plt.Axes) -> None:
        """Render the body in matplotlib."""
        self.geom.plot(ax=ax, zorder=self.z_order.value, **self.rendering_kwargs)


@dataclass(frozen=True)
class MultiBody2D:
    """A container for bodies."""

    name: str
    bodies: list[Body2D]

    def plot(self, ax: plt.Axes) -> None:
        """Render the bodies in matplotlib."""
        for body in self.bodies:
            body.plot(ax)

    def get_body(self, name: str) -> Body2D:
        """Retrieve a body by its name."""
        for body in self.bodies:
            if body.name == name:
                return body
        raise ValueError(f"Multibody {self.name} does not contain body {name}")


@dataclass(frozen=True)
class SE2Pose:
    """Container for an SE2Pose.

    In the future, may want to move this to a more general repository, e.g.,
    tomsgeoms2d.
    """

    x: float
    y: float
    theta: float  # [-pi, pi]

    @functools.cached_property
    def inverse(self) -> SE2Pose:
        """Invert the pose."""
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        return SE2Pose(-self.x * c - self.y * s, self.x * s - self.y * c, -self.theta)

    def __mul__(self, other: SE2Pose) -> SE2Pose:
        """Multiply two poses together."""
        rotated_pos = self.rotation_matrix.dot((other.x, other.y))
        return SE2Pose(
            self.x + rotated_pos[0],
            self.y + rotated_pos[1],
            wrap_angle(self.theta + other.theta),
        )

    @functools.cached_property
    def rotation_matrix(self) -> NDArray[np.float32]:
        """Create and cache the rotation matrix for this pose."""
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        return np.array([[c, -s], [s, c]])
