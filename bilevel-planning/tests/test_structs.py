"""Tests for structs.py."""

import numpy as np
from gymnasium.spaces import Box, Discrete

from bilevel_planning.structs import FunctionalGoal, PlanningProblem


def test_planning_problem():
    """Tests for PlanningProblem()."""
    state_space = Box(0.0, 1.0)
    action_space = Discrete(2)
    initial_state = 0.5

    def _transition_fn(x, u):
        dx = -0.1 if u else 0.1
        return min(1.0, max(0.0, x + dx))

    goal = FunctionalGoal(lambda x: x < 0.05)

    problem = PlanningProblem(
        state_space, action_space, initial_state, _transition_fn, goal
    )

    assert problem.state_space == state_space
    assert problem.action_space == action_space
    assert problem.initial_state == initial_state
    xs = np.linspace(0.0, 1.0, num=10, endpoint=True)
    us = [0, 1]
    assert all(
        problem.transition_function(x, u) == _transition_fn(x, u)
        for x in xs
        for u in us
    )
    assert all(problem.goal.check_state(x) == goal.check_state(x) for x in xs)
