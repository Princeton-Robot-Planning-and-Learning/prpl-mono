"""Abstract breadth-first-search planner."""

import time
from typing import Hashable, TypeVar

from bilevel_planning.bilevel_planners.bilevel_planner import BilevelPlanner
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.structs import Plan, PlanningProblem
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySampler,
    TrajectorySamplingFailure,
)

_X = TypeVar("_X")  # state
_U = TypeVar("_U")  # action
_S = TypeVar("_S", bound=Hashable)  # abstract state
_A = TypeVar("_A", bound=Hashable)  # abstract action


class AbstractBFSPlanner(BilevelPlanner[_X, _U, _S, _A]):
    """Abstract breadth-first-search planner."""

    def __init__(
        self,
        trajectory_sampler: TrajectorySampler[_X, _U, _S, _A],
        num_sampling_attempts_per_step: int,
        *args,
        **kwargs,
    ) -> None:
        self._trajectory_sampler = trajectory_sampler
        self._num_sampling_attempts_per_step = num_sampling_attempts_per_step
        super().__init__(*args, **kwargs)

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

        # Initialize the queue and visited set.
        queue = [s0]
        visited = {s0}

        # Run BFS until timeout.
        while time.perf_counter() - start_time < timeout:
            # Get the next abstract state to expand.
            s = queue.pop(0)
            # Iterate over abstract successors.
            for a, ns in self._abstract_successor_function(s):
                # Quit early if timeout.
                if time.perf_counter() - start_time >= timeout:
                    break
                # This edge is new regardless of whether ns has been visited.
                bpg.add_abstract_action_edge(s, a, ns)
                # If we've already visited this abstract state, don't revisit it.
                if ns in visited:
                    continue
                # New abstract state.
                bpg.add_abstract_state_node(ns)
                visited.add(ns)
                # Attempt refinement.
                some_refinement_succeeded = False
                for _ in range(self._num_sampling_attempts_per_step):
                    # Quit early if timeout.
                    if time.perf_counter() - start_time >= timeout:
                        break
                    x = bpg.sample_state_from_abstract_state(s, self._rng)
                    try:
                        # NOTE: this updates bpg in-place.
                        x_traj, _ = self._trajectory_sampler(
                            x, s, a, ns, bpg, self._rng
                        )
                        some_refinement_succeeded = True
                    except TrajectorySamplingFailure:
                        continue
                    # Check if we've achieved the goal. Make the simplifying assumption
                    # that the goal only needs to be checked at the end of a trajectory.
                    if problem.goal.check_state(x_traj[-1]):
                        # Plan found, return immediately.
                        plan = bpg.extract_plan(x_traj[-1])
                        return plan, bpg
                # If at least one refinement attempt succeeded, update queue.
                if some_refinement_succeeded:
                    queue.append(ns)

        return None, bpg
