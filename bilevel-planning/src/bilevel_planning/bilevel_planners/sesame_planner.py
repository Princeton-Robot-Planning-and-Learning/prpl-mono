"""Multi-abstract plan + backtracking refinement planner."""

import time
from typing import Hashable, TypeVar

from bilevel_planning.abstract_plan_generators.abstract_plan_generator import (
    AbstractPlanGenerator,
)
from bilevel_planning.bilevel_planners.bilevel_planner import BilevelPlanner
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.refiners.backtracking_refiner import BacktrackingRefiner
from bilevel_planning.refiners.refiner import Refiner
from bilevel_planning.structs import Plan, PlanningProblem
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySampler,
)

_X = TypeVar("_X")  # state
_U = TypeVar("_U")  # action
_S = TypeVar("_S", bound=Hashable)  # abstract state
_A = TypeVar("_A", bound=Hashable)  # abstract action


class SesamePlanner(BilevelPlanner[_X, _U, _S, _A]):
    """Multi-abstract plan + backtracking refinement planner."""

    def __init__(
        self,
        abstract_plan_generator: AbstractPlanGenerator[_X, _S, _A],
        trajectory_sampler: TrajectorySampler[_X, _U, _S, _A],
        max_abstract_plans: int,
        num_sampling_attempts_per_step: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._abstract_plan_generator = abstract_plan_generator
        self._trajectory_sampler = trajectory_sampler
        self._max_abstract_plans = max_abstract_plans
        self._refiner: Refiner[_X, _U, _S, _A] = BacktrackingRefiner(
            self._trajectory_sampler, num_sampling_attempts_per_step, seed=self._seed
        )

    def run(
        self, problem: PlanningProblem[_X, _U], timeout: float
    ) -> tuple[Plan | None, BilevelPlanningGraph]:
        start_time = time.perf_counter()

        # Get the initial abstract state.
        x0 = problem.initial_state
        s0 = self._state_abstractor(x0)

        # Initialize the bilevel planning graph.
        bpg: BilevelPlanningGraph[_X, _U, _S, _A] = BilevelPlanningGraph()
        bpg.add_state_node(x0)
        bpg.add_abstract_state_node(s0)
        bpg.add_state_abstractor_edge(x0, s0)

        # Generate abstract plans and attempt to refine them.
        gen = self._abstract_plan_generator(
            x0,
            s0,
            problem.goal,
            timeout,
            bpg,
        )
        num_abstract_plans = 0

        while (
            num_abstract_plans < self._max_abstract_plans
            and time.perf_counter() - start_time < timeout
        ):
            # Get the next abstract plan.
            try:
                s_plan, a_plan = next(gen)
                num_abstract_plans += 1
            except StopIteration:
                break
            # Quit early if timeout.
            remaining_time = timeout - (time.perf_counter() - start_time)
            if remaining_time < 0:
                break
            # Try to refine this abstract plan.
            plan = self._refiner(x0, s_plan, a_plan, remaining_time, bpg)
            # Plan successfully found.
            if plan is not None:
                return plan, bpg

        return None, bpg
