"""Tests for heuristic_search_plan_generator.py."""

import pytest

from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    HeuristicSearchAbstractPlanGenerator,
)
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.structs import FunctionalGoal


def test_heuristic_search_plan_generator_success():
    """Test successful abstract plan generation."""

    # Simple 2D grid world.
    def abstract_successor_function(s):
        x, y = s
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3:
                yield ((dx, dy), (nx, ny))

    def goal_check_abstract(s):
        return s == (2, 2)

    goal = FunctionalGoal(
        check_state_fn=lambda x: False,  # Not used in abstract planning
        check_abstract_state_fn=goal_check_abstract,
    )

    x0 = "start_state"
    s0 = (0, 0)
    bpg = BilevelPlanningGraph()

    # Get the first plan.
    generator = HeuristicSearchAbstractPlanGenerator(
        heuristic_factory=lambda _1, _2: lambda _: 0,
        abstract_successor_function=abstract_successor_function,
        seed=123,
    )
    plan_iterator = generator(
        x0,
        s0,
        goal,
        100,
        bpg,
    )
    s_plan, a_plan = next(plan_iterator)

    # Should reach goal state (2, 2).
    assert s_plan[-1] == (2, 2)
    assert len(s_plan) == len(a_plan) + 1


def test_heuristic_search_plan_generator_no_goal():
    """Test behavior when no abstract goal check is provided."""

    def abstract_successor_function(s):
        x, y = s
        for dx, dy in [(0, 1), (1, 0)]:
            nx, ny = x + dx, y + dy
            if nx < 2 and ny < 2:
                yield ((dx, dy), (nx, ny))

    # Goal without abstract state check.
    goal = FunctionalGoal(check_state_fn=lambda x: False)

    x0 = "start"
    s0 = (0, 0)
    bpg = BilevelPlanningGraph()

    # Should raise assertion error.
    generator = HeuristicSearchAbstractPlanGenerator(
        heuristic_factory=lambda _1, _2: lambda _: 0,
        abstract_successor_function=abstract_successor_function,
        seed=123,
    )
    with pytest.raises(AssertionError):
        list(
            generator(
                x0,
                s0,
                goal,
                100,
                bpg,
            )
        )
        assert False, "Expected assertion error"


def test_heuristic_search_plan_generator_unreachable_goal():
    """Test behavior when goal is unreachable."""

    def abstract_successor_function(s):
        # Only allow moving right, never up.
        x, y = s
        if x < 1:
            yield ((1, 0), (x + 1, y))

    def goal_check_abstract(s):
        return s == (0, 1)  # goal is at top-left, but we can only move right

    goal = FunctionalGoal(
        check_state_fn=lambda x: False,
        check_abstract_state_fn=goal_check_abstract,
    )

    x0 = "start"
    s0 = (0, 0)
    bpg = BilevelPlanningGraph()

    # Try to get a plan - should raise StopIteration if no plans exist.
    generator = HeuristicSearchAbstractPlanGenerator(
        heuristic_factory=lambda _1, _2: lambda _: 0,
        abstract_successor_function=abstract_successor_function,
        seed=123,
    )
    plan_iterator = generator(
        x0,
        s0,
        goal,
        100,
        bpg,
    )
    with pytest.raises(StopIteration):
        next(plan_iterator)


def test_heuristic_search_plan_generator_multiple_paths():
    """Test that generator can find multiple paths to goal."""

    def abstract_successor_function(s):
        x, y = s
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3:
                yield ((dx, dy), (nx, ny))

    def goal_check_abstract(s):
        return s == (1, 1)  # Goal in center

    goal = FunctionalGoal(
        check_state_fn=lambda x: False,
        check_abstract_state_fn=goal_check_abstract,
    )

    x0 = "start"
    s0 = (0, 0)
    bpg = BilevelPlanningGraph()

    # Get first two plans to test multiple paths.
    generator = HeuristicSearchAbstractPlanGenerator(
        heuristic_factory=lambda _1, _2: lambda _: 0,
        abstract_successor_function=abstract_successor_function,
        seed=123,
    )
    plan_iterator = generator(
        x0,
        s0,
        goal,
        100,
        bpg,
    )

    # Get first plan.
    s_plan1, a_plan1 = next(plan_iterator)
    assert s_plan1[-1] == (1, 1)
    assert len(s_plan1) == len(a_plan1) + 1

    # Get second plan.
    s_plan2, a_plan2 = next(plan_iterator)
    assert s_plan2[-1] == (1, 1)
    assert len(s_plan2) == len(a_plan2) + 1

    # The plans should be different.
    assert a_plan1 != a_plan2
