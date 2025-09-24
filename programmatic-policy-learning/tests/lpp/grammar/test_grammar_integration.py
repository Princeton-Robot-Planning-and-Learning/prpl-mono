"""Integration tests for grammar system."""

import pytest

from programmatic_policy_learning.lpp.grammar.provider.ggg_grammar_builder import (
    GGGGrammarBuilder,
)
from programmatic_policy_learning.lpp.grammar.registry import GrammarRegistry


def test_grammar_system_integration() -> None:
    """Test the complete grammar system integration."""
    # Create registry instance and builder
    registry = GrammarRegistry()
    builder = GGGGrammarBuilder()

    # Register builder and test registry
    registry.register("ggg", builder)
    assert registry.get_builder("ggg") == builder
    assert registry.get_builder("GGG") == builder  # case insensitive

    # Test grammar creation through registry
    object_types = ("red", "blue")
    grammar = registry.create_grammar("ggg", object_types)
    assert isinstance(grammar, dict)
    assert len(grammar) > 0

    # Test error for unregistered provider
    with pytest.raises(ValueError, match="No grammar builder registered"):
        registry.create_grammar("unknown", object_types)
