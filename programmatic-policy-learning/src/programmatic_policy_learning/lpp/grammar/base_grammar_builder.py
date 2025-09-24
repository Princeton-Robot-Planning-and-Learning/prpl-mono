"""Abstract base class for grammar builders."""

from abc import ABC, abstractmethod
from typing import Any


class BaseGrammarBuilder(ABC):
    """Abstract base class for environment provider grammar builders."""

    @abstractmethod
    def create_grammar(
        self, object_types: tuple[str, ...]
    ) -> dict[int, tuple[Any, list[float]]]:
        """Create grammar specific to the environment provider.

        Args:
            object_types: Available object types/values in the environment.

        Returns:
            Grammar dictionary with production rules and probabilities.
        """

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of the provider this grammar builder supports.

        Returns:
            Provider name (e.g., 'ggg', 'prbench', 'lunar_lander').
        """
