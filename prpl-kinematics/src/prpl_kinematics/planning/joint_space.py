"""A joint-space view of a KinematicTree for motion planning.

A :class:`JointSpace` is a group of actuated joints, in a fixed order, together
with the geometry a sampling planner needs: the bounds to sample within, a
distance metric, and straight-line interpolation. It also converts between a
flat coordinate vector (what a planner manipulates) and a
:class:`~prpl_kinematics.tree.kinematic_tree.Configuration` (what forward
kinematics and collision checking consume).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

import numpy as np

from prpl_kinematics.tree.joints import JointValues
from prpl_kinematics.tree.kinematic_tree import KinematicTree


class JointSpace:
    """An ordered group of actuated joints with sampling and interpolation."""

    def __init__(self, tree: KinematicTree, joint_names: Sequence[str]) -> None:
        self._joint_names = list(joint_names)
        lower: list[float] = []
        upper: list[float] = []
        self._dof_per_joint: list[int] = []
        for name in self._joint_names:
            joint = tree.joint(name)
            lower.extend(joint.lower_limits)
            upper.extend(joint.upper_limits)
            self._dof_per_joint.append(joint.num_dof)
        self._lower = np.asarray(lower, dtype=float)
        self._upper = np.asarray(upper, dtype=float)

    @property
    def joint_names(self) -> list[str]:
        """The joints spanned by this space, in coordinate order."""
        return list(self._joint_names)

    @property
    def dimension(self) -> int:
        """Total number of scalar coordinates across all joints."""
        return int(self._lower.size)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """A uniform random coordinate vector within the joint bounds."""
        return rng.uniform(self._lower, self._upper)

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Euclidean distance between two coordinate vectors."""
        return float(np.linalg.norm(np.subtract(a, b)))

    def clamp(self, vector: np.ndarray) -> np.ndarray:
        """Clip a coordinate vector to the joint bounds."""
        return np.clip(vector, self._lower, self._upper)

    def interpolate(
        self, a: np.ndarray, b: np.ndarray, resolution: float
    ) -> Iterator[np.ndarray]:
        """Yield waypoints stepping from ``a`` toward ``b`` (``a`` excluded).

        Steps are at most ``resolution`` apart; the final waypoint is exactly
        ``b``.
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        num_steps = max(1, int(np.ceil(self.distance(a, b) / resolution)))
        for step in range(1, num_steps + 1):
            yield a + (b - a) * (step / num_steps)

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
