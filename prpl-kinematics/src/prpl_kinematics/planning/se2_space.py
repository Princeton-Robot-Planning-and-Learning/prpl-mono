"""An SE(2) configuration space for a planar mobile base.

``SE2Space`` is the configuration space of a single ``PlanarJoint`` -- a base
that translates in ``x``/``y`` and rotates by ``yaw``. Translation is sampled
within a finite workspace box; ``yaw`` wraps at 2*pi, so distance and
interpolation take the shorter way around. The yaw contribution to distance is
scaled by ``yaw_weight`` (radians-to-metres) so it can be traded off against
translation.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import numpy as np

from prpl_kinematics.tree.joints import JointValues

Bounds = tuple[float, float]


class SE2Space:
    """The (x, y, yaw) configuration space of one planar-base joint."""

    def __init__(
        self,
        joint_name: str,
        x_bounds: Bounds,
        y_bounds: Bounds,
        yaw_weight: float = 1.0,
    ) -> None:
        self._joint_name = joint_name
        self._yaw_weight = yaw_weight
        self._lower = np.array([x_bounds[0], y_bounds[0], -np.pi], dtype=float)
        self._upper = np.array([x_bounds[1], y_bounds[1], np.pi], dtype=float)

    @property
    def dimension(self) -> int:
        """Always 3: x, y, yaw."""
        return 3

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """A uniform pose within the workspace box, yaw over [-pi, pi]."""
        return rng.uniform(self._lower, self._upper)

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Translation distance plus yaw-weighted shortest angular distance."""
        return float(np.linalg.norm(self._delta(a, b)))

    def interpolate(
        self, a: np.ndarray, b: np.ndarray, resolution: float
    ) -> Iterator[np.ndarray]:
        """Yield poses from ``a`` toward ``b`` (``a`` excluded), yaw the short way."""
        a = np.asarray(a, dtype=float)
        raw = np.subtract(b, a, dtype=float)
        raw[2] = self._wrap(raw[2])
        num_steps = max(
            1, int(np.ceil(float(np.linalg.norm(self._delta(a, b))) / resolution))
        )
        for step in range(1, num_steps + 1):
            yield a + raw * (step / num_steps)

    def clamp(self, vector: np.ndarray) -> np.ndarray:
        """Clip x/y to the workspace box; yaw is unbounded (it wraps)."""
        clamped = np.array(vector, dtype=float)
        clamped[:2] = np.clip(clamped[:2], self._lower[:2], self._upper[:2])
        return clamped

    def to_configuration(self, vector: np.ndarray) -> dict[str, JointValues]:
        """The single planar joint's (x, y, yaw) values."""
        return {self._joint_name: [float(v) for v in vector[:3]]}

    def to_vector(self, config: Mapping[str, JointValues]) -> np.ndarray:
        """This base joint's (x, y, yaw) from ``config``."""
        return np.asarray(config[self._joint_name], dtype=float)

    def _delta(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Displacement a->b with yaw wrapped and weighted (for the metric)."""
        delta = np.subtract(b, a, dtype=float)
        delta[2] = self._wrap(delta[2]) * self._yaw_weight
        return delta

    @staticmethod
    def _wrap(angle: float) -> float:
        return (angle + np.pi) % (2 * np.pi) - np.pi
