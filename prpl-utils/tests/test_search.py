"""Tests for search.py."""

from typing import Iterator, TypeAlias

import numpy as np
import pytest

from prpl_utils import search


def test_run_gbfs() -> None:
    """Tests for run_gbfs()."""
    S: TypeAlias = tuple[int, int]  # grid (row, col)
    A: TypeAlias = str  # up, down, left, right

    def _grid_successor_fn(state: S) -> Iterator[tuple[A, S, float]]:
        arrival_costs = np.array(
            [
                [1, 1, 8, 1, 1],
                [1, 8, 1, 1, 1],
                [1, 8, 1, 1, 1],
                [1, 1, 1, 8, 1],
                [1, 1, 2, 1, 1],
            ],
            dtype=float,
        )

        act_to_delta = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }

        r, c = state

        for act in sorted(act_to_delta):
            dr, dc = act_to_delta[act]
            new_r, new_c = r + dr, c + dc
            # Check if in bounds
            if not (
                0 <= new_r < arrival_costs.shape[0]
                and 0 <= new_c < arrival_costs.shape[1]
            ):
                continue
            # Valid action
            yield (act, (new_r, new_c), arrival_costs[new_r, new_c])

    def _grid_check_goal_fn(state: S) -> bool:
        # Bottom right corner of grid
        return state == (4, 4)

    def _grid_heuristic_fn(state: S) -> float:
        # Manhattan distance
        return float(abs(state[0] - 4) + abs(state[1] - 4))

    initial_state = (0, 0)
    state_sequence, action_sequence = search.run_gbfs(
        initial_state, _grid_check_goal_fn, _grid_successor_fn, _grid_heuristic_fn
    )
    assert state_sequence == [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (4, 1),
        (4, 2),
        (4, 3),
        (4, 4),
    ]
    assert action_sequence == [
        "down",
        "down",
        "down",
        "down",
        "right",
        "right",
        "right",
        "right",
    ]

    # Same, but actually reaching the goal is impossible.
    state_sequence, action_sequence = search.run_gbfs(
        initial_state, lambda s: False, _grid_successor_fn, _grid_heuristic_fn
    )
    assert state_sequence == [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (4, 1),
        (4, 2),
        (4, 3),
        (4, 4),
    ]
    assert action_sequence == [
        "down",
        "down",
        "down",
        "down",
        "right",
        "right",
        "right",
        "right",
    ]

    # Test with an infinite branching factor.
    def _inf_grid_successor_fn(state: S) -> Iterator[tuple[A, S, float]]:
        # Change all costs to 1.
        i = 0
        for a, ns, _ in _grid_successor_fn(state):
            yield (a, ns, 1.0)
        # Yield unnecessary and costly noops.
        # These lines should not be covered, and that's the point!
        while True:  # pragma: no cover
            action = f"noop{i}"
            yield (action, state, 100.0)
            i += 1

    state_sequence, action_sequence = search.run_gbfs(
        initial_state,
        _grid_check_goal_fn,
        _inf_grid_successor_fn,
        _grid_heuristic_fn,
        lazy_expansion=True,
    )
    assert state_sequence == [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (4, 1),
        (4, 2),
        (4, 3),
        (4, 4),
    ]
    assert action_sequence == [
        "down",
        "down",
        "down",
        "down",
        "right",
        "right",
        "right",
        "right",
    ]
    # Test limit on max evals.
    state_sequence, action_sequence = search.run_gbfs(
        initial_state,
        _grid_check_goal_fn,
        _inf_grid_successor_fn,
        _grid_heuristic_fn,
        max_evals=2,
    )  # note: need lazy_expansion to be False here
    assert state_sequence == [(0, 0), (1, 0)]
    assert action_sequence == ["down"]

    # Test timeout.
    # We don't care about the return value. Since the goal check always
    # returns False, the fact that this test doesn't hang means that
    # the timeout is working correctly.
    search.run_gbfs(
        initial_state,
        lambda s: False,
        _inf_grid_successor_fn,
        _grid_heuristic_fn,
        timeout=0.01,
    )


def test_run_astar() -> None:
    """Tests for run_astar()."""
    S: TypeAlias = tuple[int, int]  # grid (row, col)
    A: TypeAlias = str  # up, down, left, right

    def _grid_successor_fn(state: S) -> Iterator[tuple[A, S, float]]:
        arrival_costs = np.array(
            [
                [1, 1, 100, 1, 1],
                [1, 100, 1, 1, 1],
                [1, 100, 1, 1, 1],
                [1, 1, 1, 100, 1],
                [1, 1, 100, 1, 1],
            ],
            dtype=float,
        )

        act_to_delta = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }

        r, c = state

        for act in sorted(act_to_delta):
            dr, dc = act_to_delta[act]
            new_r, new_c = r + dr, c + dc
            # Check if in bounds
            if not (
                0 <= new_r < arrival_costs.shape[0]
                and 0 <= new_c < arrival_costs.shape[1]
            ):
                continue
            # Valid action
            yield (act, (new_r, new_c), arrival_costs[new_r, new_c])

    def _grid_check_goal_fn(state: S) -> bool:
        # Bottom right corner of grid
        return state == (4, 4)

    def _grid_heuristic_fn(state: S) -> float:
        # Manhattan distance
        return float(abs(state[0] - 4) + abs(state[1] - 4))

    initial_state = (0, 0)
    state_sequence, action_sequence = search.run_astar(
        initial_state, _grid_check_goal_fn, _grid_successor_fn, _grid_heuristic_fn
    )
    assert state_sequence == [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (3, 1),
        (3, 2),
        (2, 2),
        (2, 3),
        (2, 4),
        (3, 4),
        (4, 4),
    ]
    assert action_sequence == [
        "down",
        "down",
        "down",
        "right",
        "right",
        "up",
        "right",
        "right",
        "down",
        "down",
    ]


def test_run_hill_climbing() -> None:
    """Tests for run_hill_climbing()."""
    S: TypeAlias = tuple[int, int]  # grid (row, col)
    A: TypeAlias = str  # up, down, left, right

    def _grid_successor_fn(state: S) -> Iterator[tuple[A, S, float]]:
        arrival_costs = np.array(
            [
                [1, 1, 8, 1, 1],
                [1, 8, 1, 1, 1],
                [1, 8, 1, 1, 1],
                [1, 1, 1, 8, 1],
                [1, 1, 2, 1, 1],
            ],
            dtype=float,
        )

        act_to_delta = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }

        r, c = state

        for act in sorted(act_to_delta):
            dr, dc = act_to_delta[act]
            new_r, new_c = r + dr, c + dc
            # Check if in bounds
            if not (
                0 <= new_r < arrival_costs.shape[0]
                and 0 <= new_c < arrival_costs.shape[1]
            ):
                continue
            # Valid action
            yield (act, (new_r, new_c), arrival_costs[new_r, new_c])

    def _grid_check_goal_fn(state: S) -> bool:
        # Bottom right corner of grid
        return state == (4, 4)

    def _grid_heuristic_fn(state: S) -> float:
        # Manhattan distance
        return float(abs(state[0] - 4) + abs(state[1] - 4))

    initial_state = (0, 0)
    state_sequence, action_sequence, heuristics = search.run_hill_climbing(
        initial_state, _grid_check_goal_fn, _grid_successor_fn, _grid_heuristic_fn
    )
    assert state_sequence == [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (4, 1),
        (4, 2),
        (4, 3),
        (4, 4),
    ]
    assert action_sequence == [
        "down",
        "down",
        "down",
        "down",
        "right",
        "right",
        "right",
        "right",
    ]
    assert heuristics == [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]

    # Same, but actually reaching the goal is impossible.
    state_sequence, action_sequence, _ = search.run_hill_climbing(
        initial_state, lambda s: False, _grid_successor_fn, _grid_heuristic_fn
    )
    assert state_sequence == [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (4, 1),
        (4, 2),
        (4, 3),
        (4, 4),
    ]
    assert action_sequence == [
        "down",
        "down",
        "down",
        "down",
        "right",
        "right",
        "right",
        "right",
    ]

    # Search with no successors
    def _no_successor_fn(state: S) -> Iterator[tuple[A, S, float]]:
        if state == initial_state:
            yield "dummy_action", (2, 2), 1.0

    state_sequence, action_sequence, _ = search.run_hill_climbing(
        initial_state, lambda s: False, _no_successor_fn, _grid_heuristic_fn
    )
    assert state_sequence == [(0, 0), (2, 2)]
    assert action_sequence == ["dummy_action"]

    # Tests showing the benefit of enforced hill climbing.
    def _local_minimum_grid_heuristic_fn(state: S) -> float:
        # Manhattan distance
        if state in [(1, 0), (0, 1)]:
            return float("inf")
        return float(abs(state[0] - 4) + abs(state[1] - 4))

    # With enforced_depth 0, search fails.
    state_sequence, action_sequence, heuristics = search.run_hill_climbing(
        initial_state,
        _grid_check_goal_fn,
        _grid_successor_fn,
        _local_minimum_grid_heuristic_fn,
    )
    assert state_sequence == [(0, 0)]
    assert not action_sequence
    assert heuristics == [8.0]

    # With enforced_depth 1, search succeeds.
    state_sequence, action_sequence, heuristics = search.run_hill_climbing(
        initial_state,
        _grid_check_goal_fn,
        _grid_successor_fn,
        _local_minimum_grid_heuristic_fn,
        enforced_depth=1,
    )
    # Note that hill-climbing does not care about costs.
    assert state_sequence == [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (4, 1),
        (4, 2),
        (4, 3),
        (4, 4),
    ]
    assert action_sequence == [
        "down",
        "down",
        "down",
        "down",
        "right",
        "right",
        "right",
        "right",
    ]
    assert heuristics == [8.0, float("inf"), 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    # Test timeout.
    with pytest.raises(TimeoutError):
        search.run_hill_climbing(
            initial_state,
            _grid_check_goal_fn,
            _grid_successor_fn,
            _local_minimum_grid_heuristic_fn,
            enforced_depth=1,
            timeout=0.0,
        )

    # Test early_termination_heuristic_thresh with very high value.
    initial_state = (0, 0)
    state_sequence, action_sequence, heuristics = search.run_hill_climbing(
        initial_state,
        _grid_check_goal_fn,
        _grid_successor_fn,
        _grid_heuristic_fn,
        early_termination_heuristic_thresh=10000000,
    )
    assert state_sequence == [(0, 0)]
    assert not action_sequence


def test_run_policy_guided_astar() -> None:
    """Tests for run_policy_guided_astar()."""
    S: TypeAlias = tuple[int, int]  # grid (row, col)
    A: TypeAlias = str  # up, down, left, right

    arrival_costs = np.array(
        [
            [1, 1, 100, 1, 1],
            [1, 100, 1, 1, 1],
            [1, 100, 1, 1, 1],
            [1, 1, 1, 100, 1],
            [1, 1, 100, 1, 1],
        ],
        dtype=float,
    )

    act_to_delta = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }

    def _get_valid_actions(state: S) -> Iterator[tuple[A, float]]:
        r, c = state
        for act in sorted(act_to_delta):
            dr, dc = act_to_delta[act]
            new_r, new_c = r + dr, c + dc
            # Check if in bounds
            if not (
                0 <= new_r < arrival_costs.shape[0]
                and 0 <= new_c < arrival_costs.shape[1]
            ):
                continue
            yield (act, arrival_costs[new_r, new_c])

    def _get_next_state(state: S, action: A) -> S:
        r, c = state
        dr, dc = act_to_delta[action]
        return (r + dr, c + dc)

    goal = (4, 4)

    def _grid_check_goal_fn(state: S) -> bool:
        # Bottom right corner of grid
        return state == goal

    def _grid_heuristic_fn(state: S) -> float:
        # Manhattan distance
        return float(abs(state[0] - goal[0]) + abs(state[1] - goal[1]))

    def _policy(state: S) -> A:
        # Move right until we can't anymore.
        _, c = state
        if c >= arrival_costs.shape[1] - 1:
            return "left"
        return "right"

    initial_state = (0, 0)
    num_rollout_steps = 10

    # The policy should bias toward the path that moves all the way right, then
    # planning should move all the way down to reach the goal.
    state_sequence, action_sequence = search.run_policy_guided_astar(
        initial_state,
        _grid_check_goal_fn,
        _get_valid_actions,
        _get_next_state,
        _grid_heuristic_fn,
        _policy,
        num_rollout_steps=num_rollout_steps,
        rollout_step_cost=0,
    )

    assert state_sequence == [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 4),
        (2, 4),
        (3, 4),
        (4, 4),
    ]
    assert action_sequence == [
        "right",
        "right",
        "right",
        "right",
        "down",
        "down",
        "down",
        "down",
    ]

    # With a trivial policy, should find the optimal path.
    state_sequence, action_sequence = search.run_policy_guided_astar(
        initial_state,
        _grid_check_goal_fn,
        _get_valid_actions,
        _get_next_state,
        _grid_heuristic_fn,
        policy=lambda s: None,
        num_rollout_steps=num_rollout_steps,
        rollout_step_cost=0,
    )

    assert state_sequence == [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (3, 1),
        (3, 2),
        (2, 2),
        (2, 3),
        (2, 4),
        (3, 4),
        (4, 4),
    ]
    assert action_sequence == [
        "down",
        "down",
        "down",
        "right",
        "right",
        "up",
        "right",
        "right",
        "down",
        "down",
    ]

    # With a policy that outputs invalid actions, should ignore the policy
    # and find the optimal path.
    state_sequence, action_sequence = search.run_policy_guided_astar(
        initial_state,
        _grid_check_goal_fn,
        _get_valid_actions,
        _get_next_state,
        _grid_heuristic_fn,
        policy=lambda s: "garbage",
        num_rollout_steps=num_rollout_steps,
        rollout_step_cost=0,
    )

    assert state_sequence == [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (3, 1),
        (3, 2),
        (2, 2),
        (2, 3),
        (2, 4),
        (3, 4),
        (4, 4),
    ]
    assert action_sequence == [
        "down",
        "down",
        "down",
        "right",
        "right",
        "up",
        "right",
        "right",
        "down",
        "down",
    ]
