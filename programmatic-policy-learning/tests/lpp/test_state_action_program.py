"""Tests for LPP StateActionProgram."""

import numpy as np
import pytest

from programmatic_policy_learning.lpp.state_action_program import StateActionProgram


def test_state_action_program_primitive_function() -> None:
    """Test program that uses a primitive function."""
    state = np.array([[1, 0], [0, 1]])
    action = (0, 0)

    # Program that uses cell_is_value primitive
    program: StateActionProgram = StateActionProgram("cell_is_value(1, a, s)")

    # Test basic execution
    result = program(state, action)
    assert result


def test_state_action_program_primitive_function_false() -> None:
    """Test program returns False when condition not met."""
    state = np.array([[1, 0], [0, 1]])
    action = (0, 1)  # This position has value 0, not 1

    program: StateActionProgram = StateActionProgram("cell_is_value(1, a, s)")

    # Test complex program
    result = program(state, action)
    assert not result


def test_state_action_program_direct_state_access() -> None:
    """Test program with direct state array access."""
    state = np.array([[1, 2], [3, 4]])
    action = (0, 0)

    # Program that directly accesses state
    program: StateActionProgram = StateActionProgram("s[0, 0] == 1")

    # Test with different parameters
    assert program(state, action)


def test_state_action_program_action_coordinate_access() -> None:
    """Test program that uses action coordinates."""
    state = np.array([[1, 2], [3, 4]])

    # Program that checks if action is in top row
    program: StateActionProgram = StateActionProgram("a[0] == 0")

    # Test first call
    assert program(state, (0, 0))
    assert not program(state, (1, 0))


def test_state_action_program_caching() -> None:
    """Test that programs are compiled and cached."""
    program: StateActionProgram = StateActionProgram("True")
    state = np.array([[1]])
    action = (0, 0)

    # First call should compile
    assert program.compiled_func is None
    result1 = program(state, action)
    assert program.compiled_func is not None

    # Second call should use cached version
    result2 = program(state, action)  # type: ignore[unreachable]
    # Verify results are the same
    assert result1 == result2


def test_state_action_program_string_representation() -> None:
    """Test string representations of programs."""
    program: StateActionProgram = StateActionProgram("cell_is_value(1, a, s)")

    assert str(program) == "cell_is_value(1, a, s)"
    assert repr(program) == "StateActionProgram('cell_is_value(1, a, s)')"


def test_state_action_program_invalid_program() -> None:
    """Test that invalid programs raise appropriate errors."""
    program: StateActionProgram = StateActionProgram("invalid_syntax[")
    state = np.array([[1]])
    action = (0, 0)

    with pytest.raises(SyntaxError):
        program(state, action)


def test_state_action_program_custom_eval_context() -> None:
    """Test program with custom evaluation context."""

    # Define custom functions for a different environment
    def is_even(x: int) -> bool:
        return x % 2 == 0

    def sum_coordinates(coord: tuple[int, int]) -> int:
        return coord[0] + coord[1]

    # Custom evaluation context
    custom_context = {
        "is_even": is_even,
        "sum_coordinates": sum_coordinates,
    }

    # Create program with custom context
    program: StateActionProgram = StateActionProgram(
        "is_even(sum_coordinates(a))", eval_context=custom_context
    )

    # Test with different action types (using int instead of np arrays for this example)
    state = 42  # Simple state for this test

    # Action (1, 1) -> sum = 2 -> is_even(2) = True
    assert program(state, (1, 1))

    # Action (1, 2) -> sum = 3 -> is_even(3) = False
    assert not program(state, (1, 2))


def test_state_action_program_default_vs_custom_context() -> None:
    """Test that default context works when eval_context is None."""

    # Program using default context (should have grid primitives)
    default_program: StateActionProgram = StateActionProgram("cell_is_value(1, a, s)")
    assert default_program.eval_context is not None
    assert "cell_is_value" in default_program.eval_context
    assert "out_of_bounds" in default_program.eval_context

    # Program with empty custom context
    custom_program: StateActionProgram = StateActionProgram("True", eval_context={})
    assert custom_program.eval_context == {}

    # Verify they have different contexts
    assert default_program.eval_context != custom_program.eval_context
