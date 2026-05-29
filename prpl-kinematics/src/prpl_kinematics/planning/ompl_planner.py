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

# OMPL's RNG is process-global and can only be seeded before its first use, so
# we seed it once (from the first planner's seed) rather than per ``plan`` call.
_ompl_seeded: set[bool] = set()


def _seed_ompl_once(seed: int) -> None:
    if not _ompl_seeded:
        ou.RNG.setSeed(seed)
        _ompl_seeded.add(True)


class OMPLPlanner:
    """Plans with OMPL's RRTConnect over a ConfigurationSpace.

    Caveat: OMPL's RNG is process-global, so only the *first* planner's ``rng``
    seeds it; later instances' seeds are ignored. This makes ``rng`` a partly
    false affordance for multi-seed experiments -- to be reworked into an
    explicit one-time ``seed_ompl`` (see the deferred design issue).
    """

    def __init__(
        self,
        space: ConfigurationSpace,
        collision_fn: Callable[[Configuration], bool],
        rng: np.random.Generator,
        timeout: float = 5.0,
        simplify: bool = True,
    ) -> None:
        self._space = space
        self._collision_fn = collision_fn
        self._timeout = timeout
        self._simplify = simplify
        _seed_ompl_once(int(rng.integers(2**31)))

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
