"""A very naive baseline approach that samples random actions."""

from typing import TypeVar

from programmatic_policy_learning.approaches.base_approach import BaseApproach

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")


class RandomActionsApproach(BaseApproach[_ObsType, _ActType]):
    """A very naive baseline approach that samples random actions."""

    def _get_action(self) -> _ActType:
        return self._action_space.sample()
