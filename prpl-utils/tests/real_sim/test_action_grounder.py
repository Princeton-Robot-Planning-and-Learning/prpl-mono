"""Tests for the ActionGrounder base class."""

from __future__ import annotations

from prpl_utils.real_sim import ActionGrounder


class _OffsetGrounder(ActionGrounder[int, int, int]):
    """Real action is the sim action shifted by the current state."""

    def __call__(self, sim_action: int, sim_state: int) -> int:
        return sim_action + sim_state


def test_action_grounder_uses_state_context() -> None:
    """The grounder receives the current state and can use it to map actions."""
    grounder = _OffsetGrounder()
    assert grounder(2, 10) == 12
    assert grounder(-3, 5) == 2
