"""Tests for abstract_bfs_planner.py."""

from typing import Iterator, TypeAlias

import numpy as np
from gymnasium.spaces import Box, Discrete, Tuple

from bilevel_planning.bilevel_planners.abstract_bfs_planner import AbstractBFSPlanner
from bilevel_planning.structs import (
    FunctionalGoal,
    ParameterizedController,
    PlanningProblem,
)
from bilevel_planning.trajectory_samplers.parameterized_controller_sampler import (
    ParameterizedControllerTrajectorySampler,
)

X: TypeAlias = tuple[float, float]  # 2D position of robot
U: TypeAlias = tuple[float, float]  # dx, dy
S: TypeAlias = tuple[int, int]  # rounded x, y
A: TypeAlias = tuple[int, int]  # rounded dx, dy


class MockController(ParameterizedController[X, U]):
    """A mock controller for testing.

    Represents a parameterized policy that moves in the direction of the abstract state,
    but with some small random offset.
    """

    def __init__(self, abstract_action: tuple[int, int]) -> None:
        super().__init__()
        self._abstract_action = abstract_action
        self._current_params: tuple[float, float] = (0.0, 0.0)
        self._current_state: X | None = None
        self._current_action: U | None = None
        self._target_abstract_state: S | None = None

    def sample_parameters(self, x: X, rng: np.random.Generator) -> tuple[float, float]:
        offset_dx, offset_dy = rng.uniform(-0.1, 0.1, size=2)
        return (offset_dx, offset_dy)

    def reset(self, x: X, params: tuple[float, float]) -> None:
        self._current_params = params
        self._current_state = x
        offset_dx, offset_dy = params
        dx = self._abstract_action[0] / 2 + offset_dx
        dy = self._abstract_action[1] / 2 + offset_dy
        self._current_action = (dx, dy)
        self._target_abstract_state = (
            round(x[0]) + self._abstract_action[0],
            round(x[1]) + self._abstract_action[1],
        )

    def terminated(self) -> bool:
        if self._current_state is None:
            return False
        # Terminate when we've reached the target.
        current_abstract_state = (
            round(self._current_state[0]),
            round(self._current_state[1]),
        )
        return current_abstract_state == self._target_abstract_state

    def step(self) -> U:
        assert self._current_action is not None
        return self._current_action

    def observe(self, x: X) -> None:
        self._current_state = x


def test_abstract_bfs_planner_success():
    """Test successful planning."""

    state_space = Tuple([Box(0.0, 10.0), Box(0.0, 10.0)])
    action_space = Tuple([Box(-0.5, 0.5), Box(-0.5, 0.5)])
    initial_state = (10.0, 5.0)

    def transition_fn(x: X, u: U) -> X:
        return (x[0] + u[0], x[1] + u[1])

    def goal_check(x: X) -> bool:
        return (x[0] - 5.0) ** 2 + (x[1] - 5.0) ** 2 < 1

    problem = PlanningProblem(
        state_space,
        action_space,
        initial_state,
        transition_fn,
        FunctionalGoal(goal_check),
    )

    abstract_state_space = Tuple([Discrete(11), Discrete(11)])

    def state_abstractor(x: X) -> S:
        return (round(x[0]), round(x[1]))

    def abstract_successor_function(s: S) -> Iterator[tuple[A, S]]:
        for a in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ns = (s[0] + a[0], s[1] + a[1])
            if not abstract_state_space.contains(ns):
                continue
            yield (a, ns)

    controllers = {}
    for abstract_action in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        controller = MockController(abstract_action)
        controllers[abstract_action] = controller

    trajectory_sampler = ParameterizedControllerTrajectorySampler(
        controller_generator=controllers.get,
        transition_function=transition_fn,
        state_abstractor=state_abstractor,
        max_trajectory_steps=5,
    )

    num_sampling_attempts_per_action = 5
    planner = AbstractBFSPlanner(
        trajectory_sampler,
        num_sampling_attempts_per_action,
        abstract_successor_function,
        state_abstractor,
        seed=123,
    )

    plan, bpg = planner.run(problem, timeout=10)
    assert plan is not None

    # Uncomment to make GIF.
    # bpg.set_state_position_function(lambda x: x)
    # bpg.set_abstract_state_position_function(lambda x: x)
    # bpg.render_gif(save_path="abstract_bfs_test.gif", final_state=plan.states[-1])

    del bpg
