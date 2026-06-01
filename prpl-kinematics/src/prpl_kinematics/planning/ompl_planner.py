"""OMPL-backed motion planning over a ConfigurationSpace.

``OMPLPlanner`` wraps OMPL's ``RRTConnect`` (via ``SimpleSetup``) so it satisfies
the same ``MotionPlanner`` interface as ``BiRRTPlanner``. The OMPL state space is
a ``RealVectorStateSpace`` bounded by the configuration space's ``bounds()``;
validity is the user's ``config -> bool`` collision callable, and the OMPL
low-level state handling is kept entirely inside this class.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou

from prpl_kinematics.planning.configuration_space import ConfigurationSpace
from prpl_kinematics.tree.kinematic_tree import Configuration

ou.setLogLevel(ou.LogLevel.LOG_ERROR)

# OMPL's RNG is process-global: ``ou.RNG.setSeed`` only takes effect before the
# RNG's first use. ``seed_ompl`` records the one seed applied this process and
# refuses a conflicting re-seed rather than silently dropping it.
_ompl_seed: list[int] = []


def seed_ompl(seed: int) -> None:
    """Seed OMPL's process-global RNG, once per process.

    OMPL's RNG is shared across every ``OMPLPlanner`` and only honors a seed set
    before its first use, so call this once, before constructing any planner.
    Re-calling with the same seed is a no-op; re-calling with a *different* seed
    raises -- the second seed could not take effect, so a silent no-op would be a
    reproducibility trap. To collect IID samples, seed once and run repeatedly
    (consecutive draws from the seeded stream are independent); for fully
    reproducible per-seed runs, use one process per seed.
    """
    if _ompl_seed:
        if _ompl_seed[0] != seed:
            raise RuntimeError(
                f"OMPL's RNG is process-global and already seeded with "
                f"{_ompl_seed[0]}; it cannot be re-seeded to {seed} in the same "
                f"process. Use one process per seed for reproducible per-seed runs."
            )
        return
    ou.RNG.setSeed(int(seed))
    _ompl_seed.append(int(seed))


class OMPLPlanner:
    """Plans with OMPL's RRTConnect over a ConfigurationSpace.

    Unlike :class:`BiRRTPlanner`, this takes no per-instance ``rng``: OMPL's RNG is
    process-global (see :func:`seed_ompl`), so a per-instance seed could not be
    honored. Seed the process once with ``seed_ompl`` for reproducibility, or leave
    it unseeded for OMPL's default.
    """

    def __init__(
        self,
        space: ConfigurationSpace,
        collision_fn: Callable[[Configuration], bool],
        timeout: float = 5.0,
        simplify: bool = True,
    ) -> None:
        self._space = space
        self._collision_fn = collision_fn
        self._timeout = timeout
        self._simplify = simplify

    def plan(
        self, start: Configuration, goal: Configuration
    ) -> list[Configuration] | None:
        """A collision-free path from ``start`` to ``goal``, or ``None``."""
        base = dict(start)
        dimension = self._space.dimension
        lower, upper = self._space.bounds()

        state_space = ob.RealVectorStateSpace(dimension)
        bounds = ob.RealVectorBounds(dimension)
        for i in range(dimension):
            bounds.setLow(i, float(lower[i]))
            bounds.setHigh(i, float(upper[i]))
        state_space.setBounds(bounds)

        setup = og.SimpleSetup(state_space)
        space_information = setup.getSpaceInformation()
        setup.setStateValidityChecker(
            lambda state: not self._collision_fn(self._config(state, base, dimension))
        )
        setup.setStartAndGoalStates(
            self._state(space_information, self._space.to_vector(start), dimension),
            self._state(space_information, self._space.to_vector(goal), dimension),
        )
        setup.setPlanner(og.RRTConnect(space_information))

        setup.solve(self._timeout)
        if not setup.haveExactSolutionPath():
            return None
        if self._simplify:
            setup.simplifySolution()
        path = setup.getSolutionPath()
        path.interpolate()
        return [self._config(state, base, dimension) for state in path.getStates()]

    def _config(
        self, state: object, base: Configuration, dimension: int
    ) -> Configuration:
        vector = np.array([state[i] for i in range(dimension)])  # type: ignore[index]
        return {**dict(base), **self._space.to_configuration(vector)}

    @staticmethod
    def _state(space_information: object, vector: np.ndarray, dimension: int) -> object:
        state = space_information.allocState()  # type: ignore[attr-defined]
        for i in range(dimension):
            state[i] = float(vector[i])
        return state
