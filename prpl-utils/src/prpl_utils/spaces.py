"""Utilities for gymnasium-like spaces."""

from typing import Any, Callable, Sequence, TypeVar

import gymnasium as gym
import numpy as np

_Element = TypeVar("_Element")


class EnumSpace(gym.spaces.Space[_Element]):
    """A space defined by an arbitrary finite list."""

    def __init__(
        self,
        elements: Sequence[_Element],
        seed: int | np.random.Generator | None = None,
    ) -> None:
        super().__init__(shape=None, dtype=None, seed=seed)
        self.elements = list(elements)
        self.num_elements = len(self.elements)

    def sample(
        self, mask: Any | None = None, probability: Any | None = None
    ) -> _Element:
        del mask  # Unused
        del probability  # Unused
        return self.elements[self.np_random.choice(self.num_elements)]

    def contains(self, x: Any) -> bool:
        return x in self.elements

    @property
    def is_np_flattenable(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"EnumSpace({self.elements})"


class FunctionalSpace(gym.spaces.Space[_Element]):
    """A space defined with explicit sampling and contains functions."""

    def __init__(
        self,
        contains_fn: Callable[[Any], bool],
        sample_fn: Callable[[np.random.Generator], _Element] | None = None,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        super().__init__(shape=None, dtype=None, seed=seed)
        self._sample_fn = sample_fn
        self._contains_fn = contains_fn

    def sample(
        self, mask: Any | None = None, probability: Any | None = None
    ) -> _Element:
        del mask  # Unused
        del probability  # Unused
        if self._sample_fn is None:
            raise NotImplementedError("Sampling not implemented for space")
        return self._sample_fn(self.np_random)

    def contains(self, x: Any) -> bool:
        return self._contains_fn(x)

    @property
    def is_np_flattenable(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "FunctionalSpace()"
