"""Tests for the SimEnv protocol."""

from __future__ import annotations

from prpl_utils.sim_env import SimEnv


class _CompleteSim:
    """A class that implements every SimEnv method."""

    def __init__(self) -> None:
        self._state = 0

    def get_state(self) -> int:
        """Return the current state."""
        return self._state

    def set_state(self, state: int) -> None:
        """Overwrite the current state."""
        self._state = state

    def get_transition(self, state: int, action: int) -> tuple[int, float, bool]:
        """Return next state, reward, and terminated for a hypothetical transition."""
        next_state = state + action
        return next_state, float(action), next_state >= 10

    def get_next_state(self, state: int, action: int) -> int:
        """Return only the next state."""
        return self.get_transition(state, action)[0]

    def get_reward_and_done(self, state: int, action: int) -> tuple[float, bool]:
        """Return only reward and terminated."""
        _, reward, done = self.get_transition(state, action)
        return reward, done


class _PartialSim:
    """Missing get_transition and friends, so should not satisfy SimEnv."""

    def get_state(self) -> int:
        """Return a dummy state."""
        return 0

    def set_state(self, state: int) -> None:
        """Ignore the state for this stub."""


def test_complete_sim_satisfies_protocol() -> None:
    """A class with every SimEnv method passes isinstance(env, SimEnv)."""
    env = _CompleteSim()
    assert isinstance(env, SimEnv)


def test_partial_sim_does_not_satisfy_protocol() -> None:
    """Missing planning methods fail the runtime protocol check."""
    env = _PartialSim()
    assert not isinstance(env, SimEnv)


def test_planning_methods_round_trip() -> None:
    """get_state/set_state and the planning helpers behave consistently."""
    env: SimEnv[int, int] = _CompleteSim()
    env.set_state(3)
    assert env.get_state() == 3
    next_state, reward, terminated = env.get_transition(3, 4)
    assert next_state == 7
    assert reward == 4.0
    assert terminated is False
    assert env.get_next_state(3, 4) == 7
    assert env.get_reward_and_done(3, 8) == (8.0, True)
