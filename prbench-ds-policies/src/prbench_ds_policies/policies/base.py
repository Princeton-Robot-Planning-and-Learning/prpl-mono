"""Base class for domain-specific policies."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class StatefulPolicy(ABC):
    """Base class for policies that maintain internal state across steps.

    Policies that maintain state (e.g., tracking progress through a skill sequence) must
    inherit from this class and implement the reset() method. The reset method is called
    by demo collection scripts between episodes to ensure the policy starts fresh for
    each episode.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset the policy state for a new episode.

        This method must clear all internal state that persists between steps, returning
        the policy to its initial state as if freshly constructed.
        """

    @abstractmethod
    def __call__(self, observation: NDArray[np.float32]) -> NDArray[np.float32]:
        """Compute an action given an observation.

        Args:
            observation: The current observation from the environment.

        Returns:
            The action to take.
        """


class PolicyFailure(BaseException):
    """Raised when a policy fails to produce an action."""
