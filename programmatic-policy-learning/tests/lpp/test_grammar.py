"""Tests for grammar creation functionality."""

from programmatic_policy_learning.lpp.environment.setting import get_object_types
from programmatic_policy_learning.lpp.grammar.constants import (
    CONDITION,
    DIRECTION,
    LOCAL_PROGRAM,
    NEGATIVE_NUM,
    POSITIVE_NUM,
    START,
    VALUE,
)
from programmatic_policy_learning.lpp.grammar.grammar_builder import create_grammar


def test_create_grammar_basic_structure() -> None:
    """Test that create_grammar returns the correct basic structure."""
    object_types = ("obj1", "obj2", "obj3")
    grammar = create_grammar(object_types)

    # Check that grammar is a dictionary
    assert isinstance(grammar, dict)

    # Check that all expected grammar symbols are present
    expected_symbols = {
        START,
        LOCAL_PROGRAM,
        CONDITION,
        DIRECTION,
        POSITIVE_NUM,
        NEGATIVE_NUM,
        VALUE,
    }
    assert set(grammar.keys()) == expected_symbols

    # Check that each grammar entry is a tuple of (productions, probabilities)
    for symbol, entry in grammar.items():  # pylint: disable=unused-variable
        assert isinstance(entry, tuple)
        assert len(entry) == 2
        productions, probabilities = entry
        assert isinstance(productions, (list, tuple))
        assert isinstance(probabilities, list)
        assert len(productions) == len(probabilities)


def test_create_grammar_probability_sums() -> None:
    """Test that probabilities for each symbol sum to 1.0."""
    object_types = ("a", "b", "c", "d")
    grammar = create_grammar(object_types)

    for symbol, (
        productions,  # pylint: disable=unused-variable
        probabilities,
    ) in grammar.items():
        prob_sum = sum(probabilities)
        assert (
            abs(prob_sum - 1.0) < 1e-10
        ), f"Probabilities for symbol {symbol} sum to {prob_sum}, not 1.0"


def test_create_grammar_with_real_environment() -> None:
    """Test create_grammar with real environment object types."""
    object_types = get_object_types("TwoPileNim")
    grammar = create_grammar(object_types)

    # Basic structure checks
    assert len(grammar) == 7  # Should have 7 grammar symbols

    # Check VALUE uses the real object types
    value_productions, value_probs = grammar[VALUE]
    assert value_productions == object_types
    assert len(value_probs) == len(object_types)

    # Verify probability sum
    assert abs(sum(value_probs) - 1.0) < 1e-10
