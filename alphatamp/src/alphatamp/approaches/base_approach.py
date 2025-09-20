"""Base class for AlphaTAMP approaches."""

import abc
from typing import Generic, TypeVar

import numpy as np
from bilevel_planning.structs import Plan, PlanningProblem, SesameModels

_O = TypeVar("_O")  # observation
_X = TypeVar("_X")  # state
_U = TypeVar("_U")  # action


class BaseApproach(abc.ABC, Generic[_O, _X, _U]):
    """Base class for AlphaTAMP approaches."""

    def __init__(
        self,
        env_models: SesameModels[_O, _X, _U],
        seed: int,
    ) -> None:
        self._env_models = env_models
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def train(self, init_obs: _O) -> None:
        """Run training given one more initial observation.

        Note that step_training() is called sequentially, one training problem at a
        time. If the approach uses batch training over multiple problems, it should
        store problems as they are received and then call batch training at some
        interval.
        """
        problem = self._observation_to_planning_problem(init_obs)
        self._train(problem)

    def run_planning(self, init_obs: _O, timeout: float) -> Plan[_X, _U]:
        """Run planning from an initial observation.

        If no plan is found within the timeout, raises a TimeoutError.

        This should produce the best plan possible as quickly as possible. Some
        approaches may want to run other "slower" kinds of planning during training;
        that should happen in a different approach-specific method.
        """
        problem = self._observation_to_planning_problem(init_obs)
        return self._run_planning(problem, timeout)

    @abc.abstractmethod
    def _train(self, problem: PlanningProblem[_X, _U]) -> None:
        """The main training code, called from train()."""

    @abc.abstractmethod
    def _run_planning(
        self, problem: PlanningProblem[_X, _U], timeout: float
    ) -> Plan[_X, _U]:
        """The main planning code, called from run_planning()."""

    def _observation_to_planning_problem(self, init_obs: _O) -> PlanningProblem[_X, _U]:
        """Convert an initial observation into a planning problem."""
        init_state = self._env_models.observation_to_state(init_obs)
        goal = self._env_models.goal_deriver(init_state)
        problem = PlanningProblem(
            self._env_models.state_space,
            self._env_models.action_space,
            init_state,
            self._env_models.transition_fn,
            goal,
        )
        return problem
