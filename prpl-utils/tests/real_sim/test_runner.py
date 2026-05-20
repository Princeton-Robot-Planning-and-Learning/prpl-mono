"""Tests for the real-to-sim Runner."""

from __future__ import annotations

from typing import Any

import gymnasium
import pytest
from gymnasium import spaces

from prpl_utils.planning_agent import PlanningAgent
from prpl_utils.real_sim import Perceiver, PlanExecutor, Runner


class _CountingRealEnv(gymnasium.Env[int, int]):
    """A toy env whose obs is a counter incremented on every step."""

    observation_space: spaces.Discrete = spaces.Discrete(1_000_000)
    action_space: spaces.Discrete = spaces.Discrete(1_000_000)

    def __init__(self, max_steps: int = 5) -> None:
        self._t = 0
        self._max_steps = max_steps
        self.last_real_action: int | None = None
        self.all_real_actions: list[int] = []

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
        self.all_real_actions = []
        return self._t, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        self.last_real_action = action
        self.all_real_actions.append(action)
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


class _FixedPlanAgent(PlanningAgent[int, int, int]):
    """Returns the same fixed trajectory on every plan() call."""

    def __init__(self, seed: int, trajectory: list[tuple[int, int]]) -> None:
        super().__init__(seed)
        self._trajectory = trajectory
        self.update_calls: list[tuple[int, float, bool]] = []

    def plan(self) -> list[tuple[int, int]]:
        return list(self._trajectory)

    def _get_action(self) -> int:  # pragma: no cover - not used in trajectory mode
        return 0

    def update(self, obs: int, reward: float, done: bool, info: dict[str, Any]) -> None:
        super().update(obs, reward, done, info)
        self.update_calls.append((obs, reward, done))


class _OnePerTickExecutor(PlanExecutor[int, int, int]):
    """One real-env tick per trajectory entry; real action = state + sim_action."""

    def __init__(self) -> None:
        self._trajectory: list[tuple[int, int]] = []
        self._index = 0

    def set_trajectory(self, trajectory: list[tuple[int, int]]) -> None:
        self._trajectory = list(trajectory)
        self._index = 0

    def step(self, sim_state: int) -> tuple[int, int]:
        _, sim_action = self._trajectory[self._index]
        real_action = sim_state + sim_action
        self._index += 1
        return real_action, sim_action

    def done(self, sim_state: int) -> bool:
        del sim_state
        return self._index >= len(self._trajectory)


class _SettleExecutor(PlanExecutor[int, int, int]):
    """Two ticks per trajectory entry, to exercise repeated current_sim_action."""

    def __init__(self) -> None:
        self._trajectory: list[tuple[int, int]] = []
        self._index = 0
        self._ticks_on_index = 0

    def set_trajectory(self, trajectory: list[tuple[int, int]]) -> None:
        self._trajectory = list(trajectory)
        self._index = 0
        self._ticks_on_index = 0

    def step(self, sim_state: int) -> tuple[int, int]:
        _, sim_action = self._trajectory[self._index]
        self._ticks_on_index += 1
        if self._ticks_on_index >= 2:
            self._index += 1
            self._ticks_on_index = 0
        return sim_state + sim_action, sim_action

    def done(self, sim_state: int) -> bool:
        del sim_state
        return self._index >= len(self._trajectory)


def test_runner_step_before_reset_raises() -> None:
    """Step() before reset() must raise to surface programmer errors."""
    runner = Runner(
        _CountingRealEnv(),
        _PassThroughPerceiver(),
        _FixedPlanAgent(seed=0, trajectory=[(0, 1)]),
        _OnePerTickExecutor(),
    )
    with pytest.raises(RuntimeError):
        runner.step()


def test_runner_step_executes_whole_trajectory() -> None:
    """One outer step ticks the executor until the trajectory is exhausted."""
    real_env = _CountingRealEnv(max_steps=10)
    agent = _FixedPlanAgent(seed=0, trajectory=[(0, 3), (0, 5), (0, 7)])
    runner = Runner(real_env, _PassThroughPerceiver(), agent, _OnePerTickExecutor())
    state, reward, terminated, truncated, info = (
        runner.reset(seed=0),
        None,
        None,
        None,
        None,
    )
    assert state == 0
    state, reward, terminated, truncated, info = runner.step()
    # The executor's real action was state + sim_action at decision time:
    # state 0 + 3 -> obs 1; state 1 + 5 -> obs 2; state 2 + 7 -> obs 3.
    assert real_env.all_real_actions == [3, 6, 9]
    assert state == 3
    assert reward == 3.0
    assert terminated is False
    assert truncated is False
    assert info == {"t": 3}


def test_runner_step_calls_agent_update_per_tick() -> None:
    """agent.update fires once per inner real-env tick, not once per trajectory."""
    real_env = _CountingRealEnv(max_steps=10)
    agent = _FixedPlanAgent(seed=0, trajectory=[(0, 1), (0, 1), (0, 1)])
    runner = Runner(real_env, _PassThroughPerceiver(), agent, _OnePerTickExecutor())
    runner.reset(seed=0)
    runner.step()
    assert len(agent.update_calls) == 3
    assert [c[0] for c in agent.update_calls] == [1, 2, 3]
    assert [c[1] for c in agent.update_calls] == [1.0, 1.0, 1.0]
    assert agent.update_calls[-1][2] is False


def test_runner_step_records_planned_sim_action_in_last_action() -> None:
    """The agent's _last_action reflects the planned sim_action at each tick."""
    real_env = _CountingRealEnv(max_steps=10)
    agent = _FixedPlanAgent(seed=0, trajectory=[(0, 42)])
    runner = Runner(real_env, _PassThroughPerceiver(), agent, _OnePerTickExecutor())
    runner.reset(seed=0)
    runner.step()
    # pylint: disable=protected-access
    assert agent._last_action == 42


def test_runner_step_handles_settle_executor_repeated_sim_action() -> None:
    """A settle-style executor emits the same sim_action across multiple ticks."""
    real_env = _CountingRealEnv(max_steps=10)
    agent = _FixedPlanAgent(seed=0, trajectory=[(0, 5), (0, 9)])
    runner = Runner(real_env, _PassThroughPerceiver(), agent, _SettleExecutor())
    runner.reset(seed=0)
    state, reward, _, _, _ = runner.step()
    # Two ticks per waypoint -> four ticks total.
    assert len(real_env.all_real_actions) == 4
    assert reward == 4.0
    assert state == 4
    # Each update_call carries the sim_action the executor was tracking.
    assert len(agent.update_calls) == 4


def test_runner_step_breaks_on_termination_mid_trajectory() -> None:
    """If the env terminates mid-trajectory, step() returns immediately."""
    real_env = _CountingRealEnv(max_steps=2)
    agent = _FixedPlanAgent(seed=0, trajectory=[(0, 1), (0, 1), (0, 1), (0, 1)])
    runner = Runner(real_env, _PassThroughPerceiver(), agent, _OnePerTickExecutor())
    runner.reset(seed=0)
    state, reward, terminated, truncated, _ = runner.step()
    assert state == 2
    assert reward == 2.0
    assert terminated is True
    assert truncated is False


def test_runner_run_stops_on_termination() -> None:
    """Run() exits early when the environment terminates."""
    real_env = _CountingRealEnv(max_steps=3)
    agent = _FixedPlanAgent(seed=0, trajectory=[(0, 0), (0, 0), (0, 0), (0, 0)])
    runner = Runner(real_env, _PassThroughPerceiver(), agent, _OnePerTickExecutor())
    runner.reset(seed=0)
    total_reward = runner.run(max_steps=100)
    assert total_reward == 3.0


def test_runner_run_respects_max_outer_steps() -> None:
    """Run() exits after max_steps outer trajectories even if env hasn't terminated."""
    real_env = _CountingRealEnv(max_steps=1000)
    # One-tick trajectory means one outer step == one inner tick, making this
    # easy to count.
    agent = _FixedPlanAgent(seed=0, trajectory=[(0, 0)])
    runner = Runner(real_env, _PassThroughPerceiver(), agent, _OnePerTickExecutor())
    runner.reset(seed=0)
    total_reward = runner.run(max_steps=4)
    assert total_reward == 4.0
    assert len(real_env.all_real_actions) == 4


def test_runner_empty_trajectory_is_noop() -> None:
    """An empty plan results in zero inner ticks but doesn't crash."""
    real_env = _CountingRealEnv(max_steps=10)
    agent = _FixedPlanAgent(seed=0, trajectory=[])
    runner = Runner(real_env, _PassThroughPerceiver(), agent, _OnePerTickExecutor())
    runner.reset(seed=0)
    state, reward, terminated, truncated, info = runner.step()
    assert state == 0
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info == {}
    assert not real_env.all_real_actions


def test_runner_on_step_hook_fires_per_inner_tick() -> None:
    """on_step is invoked once per real-env tick with the threaded action context."""
    real_env = _CountingRealEnv(max_steps=10)
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

    agent = _FixedPlanAgent(seed=0, trajectory=[(0, 2), (0, 2), (0, 2)])
    runner = _RecordingRunner(
        real_env, _PassThroughPerceiver(), agent, _OnePerTickExecutor()
    )
    runner.reset(seed=0)
    runner.run(max_steps=10)
    # Three inner ticks per outer step; the env terminates at step 10, so the
    # rollout completes a few outer steps. Just check that each on_step call is
    # one inner tick with the right sim_action.
    assert all(entry[1] == 2 for entry in seen)
    # Real action = state at decision + sim_action.
    assert seen[0][2] == 0 + 2
    assert seen[1][2] == 1 + 2
    assert seen[2][2] == 2 + 2
