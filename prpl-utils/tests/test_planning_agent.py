"""Tests for the PlanningAgent subclass."""

from __future__ import annotations

from typing import Any

from prpl_utils.gym_agent import Agent
from prpl_utils.planning_agent import PlanningAgent


class _CountingPlanningAgent(PlanningAgent[int, int, int]):
    """Plans a fixed trajectory; step() pops one action like a vanilla Agent."""

    def __init__(
        self, seed: int, trajectory: list[tuple[int, int]] | None = None
    ) -> None:
        super().__init__(seed)
        self._trajectory = trajectory or [(0, 1), (1, 2), (2, 3)]
        self._popped: list[int] = []

    def plan(self) -> list[tuple[int, int]]:
        return list(self._trajectory)

    def _get_action(self) -> int:
        action = self._trajectory[len(self._popped)][1]
        self._popped.append(action)
        return action


def test_planning_agent_is_agent_subclass() -> None:
    """PlanningAgent inherits from Agent so existing Agent callers still work."""
    agent = _CountingPlanningAgent(seed=0)
    assert isinstance(agent, Agent)


def test_planning_agent_plan_returns_state_action_pairs() -> None:
    """Plan() returns the trajectory as (state, action) pairs."""
    agent = _CountingPlanningAgent(seed=0)
    agent.reset(obs=0, info={})
    trajectory = agent.plan()
    assert trajectory == [(0, 1), (1, 2), (2, 3)]


def test_planning_agent_step_still_works() -> None:
    """Agent.step contract is preserved on a planning agent."""
    agent = _CountingPlanningAgent(seed=0)
    agent.reset(obs=0, info={})
    assert agent.step() == 1
    assert agent.step() == 2
    info: dict[str, Any] = {}
    agent.update(obs=2, reward=0.0, done=False, info=info)
    assert agent.step() == 3
