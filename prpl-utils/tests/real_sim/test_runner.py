"""Tests for the real-to-sim Runner."""

from __future__ import annotations

from typing import Any

import gymnasium
import pytest
from gymnasium import spaces

from prpl_utils.gym_agent import Agent
from prpl_utils.real_sim import ActionGrounder, Perceiver, Runner


class _CountingRealEnv(gymnasium.Env[int, int]):
    """A toy env whose obs is a counter incremented on every step."""

    observation_space: spaces.Discrete = spaces.Discrete(1_000_000)
    action_space: spaces.Discrete = spaces.Discrete(1_000_000)

    def __init__(self, max_steps: int = 5) -> None:
        self._t = 0
        self._max_steps = max_steps
        self.last_real_action: int | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self._t = 0
        self.last_real_action = None
        return self._t, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        self.last_real_action = action
        self._t += 1
        terminated = self._t >= self._max_steps
        return self._t, 1.0, terminated, False, {"t": self._t}

    def render(self) -> None:
        """No-op renderer for this toy env."""
        return None


class _PassThroughPerceiver(Perceiver[int, int]):
    """Sim state is just the real obs (so we can trace it end-to-end)."""

    def reset(self, obs: int, info: dict[str, Any]) -> int:
        del info
        return obs

    def step(self, obs: int, info: dict[str, Any]) -> int:
        del info
        return obs


class _ConstantAgent(Agent[int, int]):
    """Always returns the same sim action."""

    def __init__(self, seed: int, action: int) -> None:
        super().__init__(seed)
        self._action = action

    def _get_action(self) -> int:
        return self._action


class _AddStateGrounder(ActionGrounder[int, int, int]):
    """Real action = sim action + current state, to verify state context flows."""

    def __call__(self, sim_action: int, sim_state: int) -> int:
        return sim_action + sim_state


def test_runner_step_before_reset_raises() -> None:
    """Step() before reset() must raise to surface programmer errors."""
    runner = Runner(
        _CountingRealEnv(),
        _PassThroughPerceiver(),
        _ConstantAgent(seed=0, action=1),
        _AddStateGrounder(),
    )
    with pytest.raises(RuntimeError):
        runner.step()


def test_runner_single_step_threads_state_through() -> None:
    """A single step routes the state through perceiver, agent, and grounder."""
    real_env = _CountingRealEnv(max_steps=10)
    runner = Runner(
        real_env,
        _PassThroughPerceiver(),
        _ConstantAgent(seed=0, action=3),
        _AddStateGrounder(),
    )
    initial_state = runner.reset(seed=0)
    assert initial_state == 0
    state, reward, terminated, truncated, info = runner.step()
    # The grounder added the state-at-decision (0) to the sim action (3).
    assert real_env.last_real_action == 3
    assert state == 1
    assert reward == 1.0
    assert terminated is False
    assert truncated is False
    assert info == {"t": 1}


def test_runner_run_stops_on_termination() -> None:
    """Run() exits early when the environment terminates."""
    real_env = _CountingRealEnv(max_steps=3)
    runner = Runner(
        real_env,
        _PassThroughPerceiver(),
        _ConstantAgent(seed=0, action=0),
        _AddStateGrounder(),
    )
    runner.reset(seed=0)
    total_reward = runner.run(max_steps=100)
    assert total_reward == 3.0
    assert real_env.last_real_action is not None


def test_runner_run_respects_max_steps() -> None:
    """Run() exits after max_steps even if the env hasn't terminated."""
    real_env = _CountingRealEnv(max_steps=1000)
    runner = Runner(
        real_env,
        _PassThroughPerceiver(),
        _ConstantAgent(seed=0, action=0),
        _AddStateGrounder(),
    )
    runner.reset(seed=0)
    total_reward = runner.run(max_steps=4)
    assert total_reward == 4.0


def test_runner_on_step_hook_is_called() -> None:
    """on_step is invoked once per step and receives the threaded action context."""
    real_env = _CountingRealEnv(max_steps=3)
    seen: list[tuple[int, int, int, float, bool, bool]] = []

    class _RecordingRunner(Runner[int, int, int, int]):
        """A Runner that records every on_step call for inspection."""

        def on_step(
            self,
            state: int,
            sim_action: int,
            real_action: int,
            reward: float,
            terminated: bool,
            truncated: bool,
            info: dict[str, Any],
        ) -> None:
            del info
            seen.append((state, sim_action, real_action, reward, terminated, truncated))

    runner = _RecordingRunner(
        real_env,
        _PassThroughPerceiver(),
        _ConstantAgent(seed=0, action=2),
        _AddStateGrounder(),
    )
    runner.reset(seed=0)
    runner.run(max_steps=10)
    assert len(seen) == 3
    # Grounder adds state-at-decision to sim_action (2): 0->2, 1->3, 2->4.
    assert seen[0][2] == 2
    assert seen[1][2] == 3
    assert seen[2][2] == 4
    assert seen[-1][4] is True
