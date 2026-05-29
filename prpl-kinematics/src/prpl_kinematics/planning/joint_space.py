"""A joint-space view of a KinematicTree for motion planning.

A :class:`JointSpace` is a group of actuated joints, in a fixed order, together
with the geometry a sampling planner needs: the bounds to sample within, a
distance metric, and straight-line interpolation. It also converts between a
flat coordinate vector (what a planner manipulates) and a
:class:`~prpl_kinematics.tree.kinematic_tree.Configuration` (what forward
kinematics and collision checking consume).
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence

import numpy as np

from prpl_kinematics.tree.joints import JointValues, RevoluteJoint
from prpl_kinematics.tree.kinematic_tree import KinematicTree


class JointSpace:
    """An ordered group of actuated joints with sampling and interpolation."""

    def __init__(self, tree: KinematicTree, joint_names: Sequence[str]) -> None:
        self._joint_names = list(joint_names)
        lower: list[float] = []
        upper: list[float] = []
        self._dof_per_joint: list[int] = []
        continuous: list[bool] = []
        for name in self._joint_names:
            joint = tree.joint(name)
            lower.extend(joint.lower_limits)
            upper.extend(joint.upper_limits)
            self._dof_per_joint.append(joint.num_dof)
            # A revolute joint with infinite limits wraps around at 2*pi.
            wraps = isinstance(joint, RevoluteJoint) and not (
                math.isfinite(joint.lower) and math.isfinite(joint.upper)
            )
            continuous.extend([wraps] * joint.num_dof)
        self._lower = np.asarray(lower, dtype=float)
        self._upper = np.asarray(upper, dtype=float)
        self._continuous = np.asarray(continuous, dtype=bool)

    @property
    def joint_names(self) -> list[str]:
        """The joints spanned by this space, in coordinate order."""
        return list(self._joint_names)

    @property
    def dimension(self) -> int:
        """Total number of scalar coordinates across all joints."""
        return int(self._lower.size)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """A uniform random coordinate vector within the joint bounds.

        Continuous joints (no limits) are sampled over ``[-pi, pi]``.
        """
        low = np.where(self._continuous, -np.pi, self._lower)
        high = np.where(self._continuous, np.pi, self._upper)
        return rng.uniform(low, high)

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Euclidean distance, measuring continuous joints the short way around."""
        return float(np.linalg.norm(self._delta(a, b)))

    def clamp(self, vector: np.ndarray) -> np.ndarray:
        """Clip a coordinate vector to the joint bounds."""
        return np.clip(vector, self._lower, self._upper)

    def interpolate(
        self, a: np.ndarray, b: np.ndarray, resolution: float
    ) -> Iterator[np.ndarray]:
        """Yield waypoints stepping from ``a`` toward ``b`` (``a`` excluded).

        Steps are at most ``resolution`` apart. Continuous joints take the
        shorter wrapped path, so the final waypoint reaches ``b`` modulo 2*pi.
        """
        a = np.asarray(a, dtype=float)
        delta = self._delta(a, b)
        num_steps = max(1, int(np.ceil(float(np.linalg.norm(delta)) / resolution)))
        for step in range(1, num_steps + 1):
            yield a + delta * (step / num_steps)

    def _delta(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """The displacement from ``a`` to ``b``, wrapped on continuous joints."""
        delta = np.subtract(b, a, dtype=float)
        wrapped = (delta + np.pi) % (2 * np.pi) - np.pi
        return np.where(self._continuous, wrapped, delta)

    def to_configuration(self, vector: np.ndarray) -> dict[str, JointValues]:
        """Split a coordinate vector into per-joint values."""
        config: dict[str, JointValues] = {}
        index = 0
        for name, dof in zip(self._joint_names, self._dof_per_joint):
            config[name] = [float(v) for v in vector[index : index + dof]]
            index += dof
        return config

    def to_vector(self, config: Mapping[str, JointValues]) -> np.ndarray:
        """Concatenate this space's joint values from ``config`` into a vector."""
        values: list[float] = []
        for name in self._joint_names:
            values.extend(config[name])
        return np.asarray(values, dtype=float)
