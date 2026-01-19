"""A refiner that runs backtracking search upon sampling failures."""

import time
from typing import Hashable, TypeVar

import numpy as np

from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.refiners.refiner import Refiner
from bilevel_planning.structs import Plan
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySampler,
    TrajectorySamplingFailure,
)

_X = TypeVar("_X")  # state
_U = TypeVar("_U")  # action
_S = TypeVar("_S", bound=Hashable)  # abstract state
_A = TypeVar("_A", bound=Hashable)  # abstract action


class BacktrackingRefiner(Refiner[_X, _U, _S, _A]):
    """A refiner that runs backtracking search upon sampling failures."""

    def __init__(
        self,
        trajectory_sampler: TrajectorySampler[_X, _U, _S, _A],
        num_sampling_attempts_per_step: int,
        seed: int,
    ) -> None:
        self._trajectory_sampler = trajectory_sampler
        self._num_sampling_attempts_per_step = num_sampling_attempts_per_step
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        super().__init__()

    def __call__(
        self,
        x0: _X,
        s_plan: list[_S],
        a_plan: list[_A],
        timeout: float,
        bpg: BilevelPlanningGraph[_X, _U, _S, _A],
    ) -> Plan | None:
        """Returns a plan or None if none are found."""
        assert len(s_plan) > 0 and len(a_plan) > 0, "Abstract plans cannot be empty"
        try:
            success, full_plan = self._refine_from_step(
                0, x0, s_plan, a_plan, timeout, bpg
            )
        except TimeoutError:
            return None
        if success:
            assert full_plan is not None
            return self._combine_plan_steps(full_plan)
        return None

    def _refine_from_step(
        self,
        index: int,
        x: _X,
        s_plan: list[_S],
        a_plan: list[_A],
        remaining_time: float,
        bpg: BilevelPlanningGraph[_X, _U, _S, _A],
    ) -> tuple[bool, list[tuple[list[_X], list[_U]]] | None]:

        if remaining_time <= 0:
            raise TimeoutError("Timed out during backtracking refinement")

        if index == len(a_plan):
            return True, []  # successfully refined all transitions
        s = s_plan[index]
        a = a_plan[index]
        ns = s_plan[index + 1]

        start_time = time.perf_counter()
        for _ in range(self._num_sampling_attempts_per_step):
            try:
                x_traj, u_traj = self._trajectory_sampler(x, s, a, ns, bpg, self._rng)
                time_elapsed = time.perf_counter() - start_time
                success, remainder = self._refine_from_step(
                    index + 1,
                    x_traj[-1],
                    s_plan,
                    a_plan,
                    remaining_time - time_elapsed,
                    bpg,
                )
                if success:
                    assert remainder is not None
                    return True, [(x_traj, u_traj)] + remainder
            except TrajectorySamplingFailure:
                continue

        return False, None  # backtrack

    def _combine_plan_steps(self, plan_steps: list[tuple[list[_X], list[_U]]]) -> Plan:
        x_plan: list[_X] = []
        u_plan: list[_U] = []
        for idx in range(len(plan_steps)):
            x_traj, u_traj = plan_steps[idx]
            # Remove duplicate start/end states.
            if idx > 0:
                x_traj = x_traj[1:]
            x_plan.extend(x_traj)
            u_plan.extend(u_traj)
        return Plan(x_plan, u_plan)
