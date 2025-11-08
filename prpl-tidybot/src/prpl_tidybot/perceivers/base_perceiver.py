"""Base class for perceivers, which are real-to-sim interfaces.

A perceiver is responsible for producing a "state" of some kind that can be used to set
a simulator of some kind.
"""

import abc
from typing import Generic, TypeVar

_S = TypeVar("_S")  # a state


class Perceiver(abc.ABC, Generic[_S]):
    """Base class for perceivers, which are real-to-sim interfaces."""

    @abc.abstractmethod
    def get_state(self) -> _S:
        """Get the current state."""
