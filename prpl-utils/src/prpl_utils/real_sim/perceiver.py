"""Perceivers map real-environment observations to simulator states.

A :class:`Perceiver` is stateful so that it can track objects, smooth estimates, or
integrate proprioceptive information across timesteps.
"""

from __future__ import annotations

import abc
from typing import Any, Generic, TypeVar

_RealObsType = TypeVar("_RealObsType")
_StateType = TypeVar("_StateType")


class Perceiver(Generic[_RealObsType, _StateType], abc.ABC):
    """Stateful mapping from real observations to simulator states."""

    @abc.abstractmethod
    def reset(self, obs: _RealObsType, info: dict[str, Any]) -> _StateType:
        """Reset internal state and return an initial state estimate."""

    @abc.abstractmethod
    def step(self, obs: _RealObsType, info: dict[str, Any]) -> _StateType:
        """Update internal state with a new observation and return the state
        estimate."""


__all__ = ["Perceiver"]
