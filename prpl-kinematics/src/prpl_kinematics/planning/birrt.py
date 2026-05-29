"""Bidirectional RRT motion planning over a JointSpace.

``BiRRTPlanner`` adapts the generic ``BiRRT`` from ``prpl_utils`` to a
:class:`~prpl_kinematics.planning.joint_space.JointSpace`: states are flat
coordinate vectors, and a collision test on a vector is the user-supplied
``config -> bool`` callable evaluated on the full configuration (the planned
joints merged over the start configuration, so non-planned joints stay fixed).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from prpl_utils.motion_planning import BiRRT

from prpl_kinematics.planning.joint_space import JointSpace
from prpl_kinematics.tree.kinematic_tree import Configuration


class BiRRTPlanner:
    """Plans a collision-free joint-space path with a bidirectional RRT."""

    def __init__(
        self,
        space: JointSpace,
        collision_fn: Callable[[Configuration], bool],
        rng: np.random.Generator,
        resolution: float = 0.05,
        num_attempts: int = 10,
        num_iters: int = 100,
        smooth_amt: int = 50,
    ) -> None:
        self._space = space
        self._collision_fn = collision_fn
        self._rng = rng
        self._resolution = resolution
        self._num_attempts = num_attempts
        self._num_iters = num_iters
        self._smooth_amt = smooth_amt

    def plan(
        self, start: Configuration, goal: Configuration
    ) -> list[Configuration] | None:
        """Return a path of configurations from ``start`` to ``goal``, or ``None``.

        Each returned configuration carries every joint of ``start``, with the
        planned joints varied along the path.
        """
        base = dict(start)

        def in_collision(vector: np.ndarray) -> bool:
            return self._collision_fn({**base, **self._space.to_configuration(vector)})

        rrt: BiRRT[np.ndarray] = BiRRT(
            sample_fn=lambda _: self._space.sample(self._rng),
            extend_fn=lambda a, b: self._space.interpolate(a, b, self._resolution),
            collision_fn=in_collision,
            distance_fn=self._space.distance,
            rng=self._rng,
            num_attempts=self._num_attempts,
            num_iters=self._num_iters,
            smooth_amt=self._smooth_amt,
        )
        path = rrt.query(self._space.to_vector(base), self._space.to_vector(dict(goal)))
        if path is None:
            return None
        return [{**base, **self._space.to_configuration(v)} for v in path]
