"""The configuration-space interface a sampling planner operates over.

A ``ConfigurationSpace`` supplies the operations a planner like ``BiRRTPlanner``
needs -- sampling, a distance metric, straight-line interpolation -- plus
conversion between a flat coordinate vector and a
:class:`~prpl_kinematics.tree.kinematic_tree.Configuration`. It is a ``Protocol``
so one planner spans joint space (``JointSpace``), an SE(2) mobile base
(``SE2Space``), or any future space without a shared base class.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Protocol, runtime_checkable

import numpy as np

from prpl_kinematics.tree.joints import JointValues


@runtime_checkable
class ConfigurationSpace(Protocol):
    """Sampling, distance, interpolation, and vector<->config conversion."""

    @property
    def dimension(self) -> int:
        """Number of scalar coordinates."""
        raise NotImplementedError

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """A uniform random coordinate vector within the space."""
        raise NotImplementedError

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Finite ``(lower, upper)`` sampling bounds per coordinate.

        Unbounded coordinates (continuous joints) report ``[-pi, pi]`` so a
        planner that needs explicit bounds has a finite range to sample.
        """
        raise NotImplementedError

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Distance between two coordinate vectors."""
        raise NotImplementedError

    def interpolate(
        self, a: np.ndarray, b: np.ndarray, resolution: float
    ) -> Iterator[np.ndarray]:
        """Waypoints stepping from ``a`` toward ``b`` (``a`` excluded)."""
        raise NotImplementedError

    def to_configuration(self, vector: np.ndarray) -> dict[str, JointValues]:
        """Split a coordinate vector into per-joint values."""
        raise NotImplementedError

    def to_vector(self, config: Mapping[str, JointValues]) -> np.ndarray:
        """Concatenate this space's joint values from ``config`` into a vector."""
        raise NotImplementedError
