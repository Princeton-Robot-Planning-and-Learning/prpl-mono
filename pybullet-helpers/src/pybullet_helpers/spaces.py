"""Utilities for gymnasium-like spaces."""

from typing import Any

import gymnasium as gym
import numpy as np

from pybullet_helpers.geometry import Pose


class PoseSpace(gym.spaces.Space[Pose]):
    """A space of poses.

    NOTE: sampling from this space will not result in uniform rotations. We
    implement it this way so that it's easy to bound angles in roll-pitch-yaw,
    but the downside is that sampling will be biased. If you need careful
    sampling, you should sample poses externally to the space.
    """

    def __init__(
        self,
        x_min: float = -np.inf,
        x_max: float = np.inf,
        y_min: float = -np.inf,
        y_max: float = np.inf,
        z_min: float = -np.inf,
        z_max: float = np.inf,
        roll_min: float = -np.pi,
        roll_max: float = np.pi,
        pitch_min: float = -np.pi,
        pitch_max: float = np.pi,
        yaw_min: float = -np.pi,
        yaw_max: float = np.pi,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        super().__init__(shape=None, dtype=None, seed=seed)
        self._x_min, self._x_max = x_min, x_max
        self._y_min, self._y_max = y_min, y_max
        self._z_min, self._z_max = z_min, z_max
        self._roll_min, self._roll_max = roll_min, roll_max
        self._pitch_min, self._pitch_max = pitch_min, pitch_max
        self._yaw_min, self._yaw_max = yaw_min, yaw_max
        self._min = [
            self._x_min,
            self._y_min,
            self._z_min,
            self._roll_min,
            self._pitch_min,
            self._yaw_min,
        ]
        self._max = [
            self._x_max,
            self._y_max,
            self._z_max,
            self._roll_max,
            self._pitch_max,
            self._yaw_max,
        ]

    def sample(self, mask: Any | None = None, _probability: Any | None = None) -> Pose:
        # See docstring note!
        x, y, z, roll, pitch, yaw = self.np_random.uniform(self._min, self._max)
        return Pose.from_rpy((x, y, z), (roll, pitch, yaw))

    def contains(self, x: Any) -> bool:
        if not isinstance(x, Pose):
            return False
        pos_in_bounds = (
            self._x_min <= x.position[0] <= self._x_max
            and self._y_min <= x.position[1] <= self._y_max
            and self._z_min <= x.position[2] <= self._z_max
        )
        rpy_in_bounds = (
            self._roll_min <= x.rpy[0] <= self._roll_max
            and self._pitch_min <= x.rpy[1] <= self._pitch_max
            and self._yaw_min <= x.rpy[2] <= self._yaw_max
        )
        return pos_in_bounds and rpy_in_bounds

    @property
    def is_np_flattenable(self) -> bool:
        return False

    def __repr__(self) -> str:
        return (
            f"PoseSpace(x_range=({self._x_min}, {self._x_max}), "
            f"y_range=({self._y_min}, {self._y_max}), "
            f"z_range=({self._z_min}, {self._z_max}), "
            f"roll_range=({self._roll_min}, {self._roll_max}), "
            f"pitch_range=({self._pitch_min}, {self._pitch_max}), "
            f"yaw_range=({self._yaw_min}, {self._yaw_max}))"
        )
