"""Tests for backtracking_refiner.py."""

import numpy as np

from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.refiners.backtracking_refiner import BacktrackingRefiner
from bilevel_planning.structs import Plan
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySampler,
    TrajectorySamplingFailure,
)


class SimpleTrajectorySampler(TrajectorySampler[str, int, str, str]):
    """A simple trajectory sampler for testing."""

    def __init__(self, success_rate: float = 1.0):
        self.success_rate = success_rate

    def __call__(
        self,
        x: str,
        s: str,
        a: str,
        ns: str,
        bpg: BilevelPlanningGraph[str, int, str, str],
        rng: np.random.Generator,
    ) -> tuple[list[str], list[int]]:
        """Samples a simple trajectory."""
        if rng.random() > self.success_rate:
            raise TrajectorySamplingFailure()

        # Create a simple trajectory: x -> x_new.
        x_new = f"{x}_after_{a}"
        u = hash(a) % 10  # simple action mapping

        # Update the graph.
        bpg.add_state_node(x_new)
        bpg.add_action_edge(x, u, x_new)
        bpg.add_abstract_state_node(ns)
        bpg.add_state_abstractor_edge(x_new, ns)

        return [x, x_new], [u]


def test_backtracking_refiner_success():
    """Test successful refinement."""
    sampler = SimpleTrajectorySampler(success_rate=1.0)
    refiner = BacktrackingRefiner(sampler, num_sampling_attempts_per_step=3, seed=123)

    x0 = "start"
    s_plan = ["s1", "s2", "s3"]
    a_plan = ["a1", "a2"]
    bpg = BilevelPlanningGraph()
    bpg.add_state_node(x0)
    bpg.add_abstract_state_node("s1")
    bpg.add_state_abstractor_edge(x0, "s1")

    plan = refiner(x0, s_plan, a_plan, timeout=10.0, bpg=bpg)

    assert plan is not None
    assert isinstance(plan, Plan)
    assert len(plan.states) == 3  # start -> start_after_a1 -> start_after_a1_after_a2
    assert len(plan.actions) == 2  # a1, a2


def test_backtracking_refiner_failure():
    """Test refinement failure when trajectory sampling always fails."""
    sampler = SimpleTrajectorySampler(success_rate=0.0)
    refiner = BacktrackingRefiner(sampler, num_sampling_attempts_per_step=2, seed=123)

    x0 = "start"
    s_plan = ["s1", "s2"]
    a_plan = ["a1"]
    bpg = BilevelPlanningGraph()
    bpg.add_state_node(x0)
    bpg.add_abstract_state_node("s1")
    bpg.add_state_abstractor_edge(x0, "s1")

    plan = refiner(x0, s_plan, a_plan, timeout=10.0, bpg=bpg)

    assert plan is None


def test_backtracking_refiner_timeout():
    """Test refinement timeout."""
    sampler = SimpleTrajectorySampler(success_rate=1.0)
    refiner = BacktrackingRefiner(sampler, num_sampling_attempts_per_step=10, seed=123)

    x0 = "start"
    s_plan = ["s1", "s2"]
    a_plan = ["a1"]
    bpg = BilevelPlanningGraph()
    bpg.add_state_node(x0)
    bpg.add_abstract_state_node("s1")
    bpg.add_state_abstractor_edge(x0, "s1")

    plan = refiner(x0, s_plan, a_plan, timeout=0.0, bpg=bpg)

    assert plan is None


def test_backtracking_refiner_partial_success():
    """Test refinement with partial success (some sampling attempts fail)."""
    # Create a sampler that fails 50% of the time.
    sampler = SimpleTrajectorySampler(success_rate=0.5)
    refiner = BacktrackingRefiner(sampler, num_sampling_attempts_per_step=5, seed=123)

    x0 = "start"
    s_plan = ["s1", "s2"]
    a_plan = ["a1"]
    bpg = BilevelPlanningGraph()
    bpg.add_state_node(x0)
    bpg.add_abstract_state_node("s1")
    bpg.add_state_abstractor_edge(x0, "s1")

    plan = refiner(x0, s_plan, a_plan, timeout=10.0, bpg=bpg)

    # Should succeed eventually with enough attempts.
    assert plan is not None
    assert isinstance(plan, Plan)
    assert len(plan.states) == 2
    assert len(plan.actions) == 1
