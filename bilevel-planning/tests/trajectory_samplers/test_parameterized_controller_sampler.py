"""Tests for parameterized_controller_sampler.py."""

import numpy as np
import pytest

from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.structs import ParameterizedController
from bilevel_planning.trajectory_samplers.parameterized_controller_sampler import (
    ParameterizedControllerTrajectorySampler,
)
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)


class MockController(ParameterizedController[int, str]):
    """A mock controller for testing."""

    def __init__(self, actions: list[str], termination_step: int):
        self.actions = actions
        self.termination_step = termination_step
        self.step_count = 0
        self.current_params: dict = {}

    def sample_parameters(self, x: int, rng: np.random.Generator) -> dict:
        return {"param": x}

    def reset(self, x: int, params: dict) -> None:
        self.step_count = 0
        self.current_params = params

    def terminated(self) -> bool:
        return self.step_count >= self.termination_step

    def step(self) -> str:
        if self.step_count < len(self.actions):
            action = self.actions[self.step_count]
        else:
            action = "default"
        self.step_count += 1
        return action

    def observe(self, x: int) -> None:
        pass


def test_parameterized_controller_trajectory_sampler_success():
    """Test successful trajectory sampling."""

    controller = MockController(["up", "right"], termination_step=2)
    controllers = {"move": controller}

    def transition_fn(x: int, u: str) -> int:
        if u == "up":
            return x + 10
        if u == "right":
            return x + 1
        return x

    def state_abstractor(x: int) -> str:
        return f"region_{x // 10}"

    sampler = ParameterizedControllerTrajectorySampler(
        controller_generator=controllers.get,
        transition_function=transition_fn,
        state_abstractor=state_abstractor,
        max_trajectory_steps=10,
    )

    x = 5
    s = "region_0"
    a = "move"
    ns = "region_1"
    bpg = BilevelPlanningGraph()
    bpg.add_state_node(x)
    bpg.add_abstract_state_node(s)
    bpg.add_abstract_state_node(ns)
    bpg.add_abstract_action_edge(s, a, ns)
    rng = np.random.default_rng(123)

    x_traj, u_traj = sampler(x, s, a, ns, bpg, rng)

    assert x_traj == [5, 15, 16]
    assert u_traj == ["up", "right"]
    assert len(bpg.states) == 3
    assert len(bpg.action_edges) == 2


def test_parameterized_controller_trajectory_sampler_failure():
    """Test trajectory sampling failure when target abstract state not reached."""

    controller = MockController(["up"], termination_step=1)
    controllers = {"move": controller}

    def transition_fn(x: int, u: str) -> int:
        del u  # not used
        return x + 1  # Small step, won't reach target region

    def state_abstractor(x: int) -> str:
        return f"region_{x // 10}"

    sampler = ParameterizedControllerTrajectorySampler(
        controller_generator=controllers.get,
        transition_function=transition_fn,
        state_abstractor=state_abstractor,
        max_trajectory_steps=10,
    )

    # Test
    x = 5
    s = "region_0"
    a = "move"
    ns = "region_1"  # target region not reached
    bpg = BilevelPlanningGraph()
    bpg.add_state_node(x)
    bpg.add_abstract_state_node(s)
    bpg.add_abstract_state_node(ns)
    bpg.add_abstract_action_edge(s, a, ns)
    rng = np.random.default_rng(123)

    with pytest.raises(TrajectorySamplingFailure):
        sampler(x, s, a, ns, bpg, rng)


def test_parameterized_controller_trajectory_sampler_max_steps():
    """Test trajectory sampling with max steps limit."""

    controller = MockController(["up"] * 15, termination_step=20)  # won't terminate
    controllers = {"move": controller}

    def transition_fn(x: int, u: str) -> int:
        del u  # not used
        return x + 1

    def state_abstractor(x: int) -> str:
        return f"region_{x // 10}"

    sampler = ParameterizedControllerTrajectorySampler(
        controller_generator=controllers.get,
        transition_function=transition_fn,
        state_abstractor=state_abstractor,
        max_trajectory_steps=2,  # small limit
    )

    x = 5
    s = "region_0"
    a = "move"
    ns = "region_1"
    bpg = BilevelPlanningGraph()
    bpg.add_state_node(x)
    bpg.add_abstract_state_node(s)
    bpg.add_abstract_state_node(ns)
    bpg.add_abstract_action_edge(s, a, ns)
    rng = np.random.default_rng(123)

    with pytest.raises(TrajectorySamplingFailure):
        sampler(x, s, a, ns, bpg, rng)
