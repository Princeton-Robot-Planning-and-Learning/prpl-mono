"""Tests for the PlanExecutor base class."""

from __future__ import annotations

from prpl_utils.real_sim import PlanExecutor


class _LinearExecutor(PlanExecutor[int, int, int]):
    """One real-env tick per trajectory entry; real action = sim_state + sim_action."""

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


def test_plan_executor_walks_trajectory() -> None:
    """Set, step, step, step, done."""
    executor = _LinearExecutor()
    executor.set_trajectory([(0, 10), (0, 20), (0, 30)])
    assert executor.done(0) is False
    assert executor.step(1) == (11, 10)
    assert executor.step(2) == (22, 20)
    assert executor.done(0) is False
    assert executor.step(3) == (33, 30)
    assert executor.done(0) is True


def test_plan_executor_set_trajectory_resets_progress() -> None:
    """set_trajectory restarts the executor from the beginning."""
    executor = _LinearExecutor()
    executor.set_trajectory([(0, 1), (0, 2)])
    executor.step(0)
    executor.step(0)
    assert executor.done(0) is True
    executor.set_trajectory([(0, 5), (0, 6), (0, 7)])
    assert executor.done(0) is False
    assert executor.step(0) == (5, 5)
