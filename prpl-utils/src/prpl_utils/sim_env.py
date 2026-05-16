"""Protocol for simulator environments that support state inspection and setting.

A :class:`SimEnv` extends the usual ``gymnasium.Env`` API with methods for reading,
writing, and rolling out from arbitrary states. Any class that provides these methods
will satisfy the protocol structurally; no inheritance is required.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

_StateType = TypeVar("_StateType")
_ActType_contra = TypeVar("_ActType_contra", contravariant=True)


@runtime_checkable
class SimEnv(Protocol[_StateType, _ActType_contra]):
    """A simulator environment with state-setting and one-step planning helpers."""

    def get_state(self) -> _StateType:
        """Return the current environment state."""

    def set_state(self, state: _StateType) -> None:
        """Overwrite the environment state."""

    def get_transition(
        self, state: _StateType, action: _ActType_contra
    ) -> tuple[_StateType, float, bool]:
        """Return ``(next_state, reward, terminated)`` from ``state`` taking
        ``action``."""

    def get_next_state(self, state: _StateType, action: _ActType_contra) -> _StateType:
        """Return the next state from ``state`` taking ``action``."""

    def get_reward_and_done(
        self, state: _StateType, action: _ActType_contra
    ) -> tuple[float, bool]:
        """Return ``(reward, terminated)`` from ``state`` taking ``action``."""


__all__ = ["SimEnv"]
