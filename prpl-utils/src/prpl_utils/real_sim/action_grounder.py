"""Action grounders map simulator actions to real-environment actions.

An :class:`ActionGrounder` is the dual of a :class:`prpl_utils.perceiver.Perceiver`:
the perceiver lifts real observations into simulator states, and the grounder lowers
simulator actions into actions that can be executed in the real environment.
"""

from __future__ import annotations

import abc
from typing import Generic, TypeVar

_SimActType = TypeVar("_SimActType")
_RealActType = TypeVar("_RealActType")
_StateType = TypeVar("_StateType")


class ActionGrounder(Generic[_SimActType, _RealActType, _StateType], abc.ABC):
    """Mapping from a simulator action (in context of a state) to a real action."""

    @abc.abstractmethod
    def __call__(self, sim_action: _SimActType, sim_state: _StateType) -> _RealActType:
        """Return the real action that executes ``sim_action`` from ``sim_state``."""


__all__ = ["ActionGrounder"]
