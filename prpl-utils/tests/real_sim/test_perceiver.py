"""Tests for the Perceiver base class."""

from __future__ import annotations

from typing import Any

from prpl_utils.real_sim import Perceiver


class _CountingPerceiver(Perceiver[int, tuple[int, int]]):
    """Returns (obs, steps_since_reset)."""

    def __init__(self) -> None:
        self._steps = 0

    def reset(self, obs: int, info: dict[str, Any]) -> tuple[int, int]:
        del info
        self._steps = 0
        return obs, self._steps

    def step(self, obs: int, info: dict[str, Any]) -> tuple[int, int]:
        del info
        self._steps += 1
        return obs, self._steps


def test_perceiver_reset_and_step() -> None:
    """Reset() returns an initial estimate and step() advances internal state."""
    perceiver = _CountingPerceiver()
    assert perceiver.reset(7, {}) == (7, 0)
    assert perceiver.step(8, {}) == (8, 1)
    assert perceiver.step(9, {}) == (9, 2)
    assert perceiver.reset(0, {}) == (0, 0)
