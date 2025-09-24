"""Grammar registry system for managing provider-specific grammar builders."""

from typing import Optional

from programmatic_policy_learning.lpp.grammar.base_grammar_builder import (
    BaseGrammarBuilder,
)


class GrammarRegistry:
    """Registry for managing environment provider grammar builders."""

    def __init__(self) -> None:
        self._builders: dict[str, BaseGrammarBuilder] = {}

    def register(self, provider_name: str, builder: BaseGrammarBuilder) -> None:
        """Register a grammar builder for a provider.

        Args:
            provider_name: Name of the provider (e.g., 'ggg', 'prbench').
            builder: Instance of the grammar builder.
        """
        self._builders[provider_name.lower()] = builder

    def get_builder(self, provider_name: str) -> Optional[BaseGrammarBuilder]:
        """Get grammar builder for a specific provider.

        Args:
            provider_name: Name of the provider.

        Returns:
            Grammar builder instance or None if not found.
        """
        return self._builders.get(provider_name.lower())

    def create_grammar(
        self, provider_name: str, object_types: tuple[str, ...]
    ) -> dict[int, tuple[object, list[float]]]:
        """Create grammar for a specific provider.

        Args:
            provider_name: Name of the provider.
            object_types: Available object types/values in the environment.

        Returns:
            Grammar dictionary with production rules and probabilities.

        Raises:
            ValueError: If no grammar builder is registered for the provider.
        """
        builder = self.get_builder(provider_name)
        if builder is None:
            raise ValueError(
                f"No grammar builder registered for provider '{provider_name}'. "
                f"Available providers: {list(self._builders.keys())}"
            )
        return builder.create_grammar(object_types)

    def list_providers(self) -> list[str]:
        """List all registered providers.

        Returns:
            List of registered provider names.
        """
        return list(self._builders.keys())

    def clear(self) -> None:
        """Clear all registered builders.

        Mainly for testing.
        """
        self._builders.clear()
