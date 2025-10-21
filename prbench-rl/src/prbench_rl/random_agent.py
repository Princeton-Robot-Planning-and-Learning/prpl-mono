"""Random action baseline agent for PRBench environments."""

from typing import TypeVar

from prbench_rl.agent import BaseRLAgent

_O = TypeVar("_O")
_U = TypeVar("_U")


class RandomAgent(BaseRLAgent[_O, _U]):
    """Random action baseline agent."""

    def _get_action(self) -> _U:
        """Sample a random action from the action space."""
        return self.action_space.sample()
