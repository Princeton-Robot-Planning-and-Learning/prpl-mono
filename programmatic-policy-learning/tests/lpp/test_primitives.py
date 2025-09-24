"""Tests for LPP DSL primitives."""

from typing import Any

import numpy as np

from programmatic_policy_learning.lpp.dsl import (
    at_action_cell,
    at_cell_with_value,
    cell_is_value,
    out_of_bounds,
    scanning,
    shifted,
)


def test_out_of_bounds() -> None:
    """Test out_of_bounds function."""
    shape = (3, 3)

    # Valid positions
    assert out_of_bounds(0, 0, shape) is False
    assert out_of_bounds(2, 2, shape) is False
    assert out_of_bounds(1, 1, shape) is False

    # Invalid positions
    assert out_of_bounds(-1, 0, shape) is True
    assert out_of_bounds(0, -1, shape) is True
    assert out_of_bounds(3, 0, shape) is True
    assert out_of_bounds(0, 3, shape) is True


def test_cell_is_value_valid_position() -> None:
    """Test cell_is_value with valid positions."""
    obs = np.array([[1, 2], [3, 4]])

    assert cell_is_value(1, (0, 0), obs)
    assert cell_is_value(2, (0, 1), obs)
    assert cell_is_value(3, (1, 0), obs)
    assert cell_is_value(4, (1, 1), obs)

    # Wrong values
    assert not cell_is_value(5, (0, 0), obs)
    assert not cell_is_value(1, (0, 1), obs)


def test_cell_is_value_out_of_bounds() -> None:
    """Test cell_is_value with out of bounds positions."""
    obs = np.array([[1, 2], [3, 4]])

    # Out of bounds positions should return False
    assert cell_is_value(1, (-1, 0), obs) is False
    assert cell_is_value(1, (0, -1), obs) is False
    assert cell_is_value(1, (2, 0), obs) is False
    assert cell_is_value(1, (0, 2), obs) is False


def test_cell_is_value_none_position() -> None:
    """Test cell_is_value with None position."""
    obs = np.array([[1, 2], [3, 4]])

    # Test with invalid position
    assert not cell_is_value(1, None, obs)
    assert not cell_is_value(None, None, obs)


def test_shifted_basic() -> None:
    """Test shifted applies program to cell moved by direction vector."""
    obs = np.array([[1, 2], [3, 4]])

    # Simple local program that checks if cell value equals 4
    def local_program(cell: tuple[int, int] | None, obs: np.ndarray) -> bool:
        if cell is None:
            return False
        return obs[cell[0], cell[1]] == 4

    # Start at (0, 0) and shift right+down by (1, 1) to reach (1, 1) which has value 4
    result = shifted((1, 1), local_program, (0, 0), obs)
    assert result  # Should be True because obs[1,1] == 4


def test_shifted_none_cell() -> None:
    """Test shifted handles None cell correctly."""
    obs = np.array([[1, 2], [3, 4]])

    def local_program(  # pylint: disable=unused-argument
        cell: tuple[int, int] | None, obs: np.ndarray
    ) -> bool:
        return cell is not None

    # Shifting None should result in None being passed to local_program
    result = shifted((1, 0), local_program, None, obs)
    assert not result  # Should be False because cell is None


def test_shifted_directions() -> None:
    """Test shifted works with different direction vectors."""
    obs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def get_value_program(cell: tuple[int, int] | None, obs: np.ndarray) -> int:
        if cell is None or out_of_bounds(cell[0], cell[1], obs.shape):
            return -1
        return obs[cell[0], cell[1]]

    start_cell = (1, 1)  # Center cell with value 5

    # Test different directions
    assert shifted((0, 1), get_value_program, start_cell, obs) == 6  # Right: (1,2)
    assert shifted((0, -1), get_value_program, start_cell, obs) == 4  # Left: (1,0)
    assert shifted((1, 0), get_value_program, start_cell, obs) == 8  # Down: (2,1)
    assert shifted((-1, 0), get_value_program, start_cell, obs) == 2  # Up: (0,1)


def test_at_cell_with_value_found() -> None:
    """Test at_cell_with_value finds and applies program to cell with value."""
    obs = np.array([[1, 2], [3, 4]])

    # Local program that returns the cell coordinates as a tuple
    def get_coordinates_program(  # pylint: disable=unused-argument
        cell: tuple[int, int] | None, obs: np.ndarray
    ) -> Any:
        return cell

    # Look for value 4, should find it at (1, 1)
    result = at_cell_with_value(4, get_coordinates_program, obs)
    assert np.array_equal(result, [1, 1])  # np.argwhere returns numpy array


def test_at_cell_with_value_not_found() -> None:
    """Test at_cell_with_value when value doesn't exist in grid."""
    obs = np.array([[1, 2], [3, 4]])

    def check_cell_program(  # pylint: disable=unused-argument
        cell: tuple[int, int] | None, obs: np.ndarray
    ) -> bool:
        return cell is not None

    # Look for value 99 which doesn't exist
    result = at_cell_with_value(99, check_cell_program, obs)
    assert not result  # Should be False because cell is None


def test_at_cell_with_value_multiple_occurrences() -> None:
    """Test at_cell_with_value finds first occurrence when value appears
    multiple times."""
    obs = np.array([[1, 2, 1], [3, 1, 4]])

    def get_coordinates_program(  # pylint: disable=unused-argument
        cell: tuple[int, int] | None, obs: np.ndarray
    ) -> tuple[int, int] | None:
        return cell

    # Look for value 1, should find first occurrence at (0, 0)
    result = at_cell_with_value(1, get_coordinates_program, obs)
    assert np.array_equal(result, [0, 0])  # First occurrence


def test_at_cell_with_value_with_logic() -> None:
    """Test at_cell_with_value applies meaningful logic to found cell."""
    obs = np.array([[5, 2], [3, 8]])

    def check_neighbors_program(cell: tuple[int, int] | None, obs: np.ndarray) -> bool:
        """Check if cell has any neighbors with value > 5."""
        if cell is None:
            return False

        r, c = cell[0], cell[1]
        # Check all 4 directions
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if not out_of_bounds(nr, nc, obs.shape) and obs[nr, nc] > 5:
                return True
        return False

    # Find cell with value 3 at (1, 0), check if it has neighbors > 5
    # (1, 0) has neighbor (0, 0) with value 5, and (1, 1) with value 8
    result = at_cell_with_value(3, check_neighbors_program, obs)
    assert result  # Should be True because neighbor (1, 1) has value 8 > 5


def test_at_action_cell_basic() -> None:
    """Test at_action_cell passes through parameters correctly."""
    obs = np.array([[1, 2], [3, 4]])
    cell = (1, 1)

    def get_value_program(cell: tuple[int, int] | None, obs: np.ndarray) -> int:
        if cell is None:
            return -1  # Default value for None case
        return int(obs[cell[0], cell[1]])

    result = at_action_cell(get_value_program, cell, obs)
    assert result == 4  # Value at (1, 1)


def test_at_action_cell_none() -> None:
    """Test at_action_cell handles None cell correctly."""
    obs = np.array([[1, 2], [3, 4]])

    def check_none_program(  # pylint: disable=unused-argument
        cell: tuple[int, int] | None, obs: np.ndarray
    ) -> bool:
        return cell is None

    result = at_action_cell(check_none_program, None, obs)
    assert result  # Should be True because cell is None


def test_scanning_finds_true_condition() -> None:
    """Test scanning stops when true condition is met."""
    obs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def true_condition(cell: tuple[int, int] | None, obs: np.ndarray) -> bool:
        if cell is None or out_of_bounds(cell[0], cell[1], obs.shape):
            return False
        return obs[cell[0], cell[1]] == 6  # Looking for value 6

    def false_condition(  # pylint: disable=unused-argument
        cell: tuple[int, int] | None, obs: np.ndarray
    ) -> bool:
        return False  # Never stop early

    # Start at (1, 0) and scan right, should find 6 at (1, 2)
    result = scanning((0, 1), true_condition, false_condition, (1, 0), obs)
    assert result  # Should find 6 and return True


def test_scanning_hits_false_condition() -> None:
    """Test scanning stops when false condition is met."""
    obs = np.array([[1, 2, 3], [4, 0, 6], [7, 8, 9]])

    def true_condition(  # pylint: disable=unused-argument
        cell: tuple[int, int] | None, obs: np.ndarray
    ) -> bool:
        return False  # Never find true condition

    def false_condition(cell: tuple[int, int] | None, obs: np.ndarray) -> bool:
        if cell is None or out_of_bounds(cell[0], cell[1], obs.shape):
            return False
        return obs[cell[0], cell[1]] == 0  # Stop at value 0

    # Start at (1, 0) and scan right, should hit 0 at (1, 1) and stop
    result = scanning((0, 1), true_condition, false_condition, (1, 0), obs)
    assert not result  # Should stop at false condition and return False


def test_scanning_hits_boundary() -> None:
    """Test scanning stops when reaching grid boundary."""
    obs = np.array([[1, 2], [3, 4]])

    def true_condition(  # pylint: disable=unused-argument
        cell: tuple[int, int] | None, obs: np.ndarray
    ) -> bool:
        return False  # Never find true

    def false_condition(  # pylint: disable=unused-argument
        cell: tuple[int, int] | None, obs: np.ndarray
    ) -> bool:
        return False  # Never stop early

    # Start at (0, 0) and scan left, should hit boundary immediately
    result = scanning((0, -1), true_condition, false_condition, (0, 0), obs)
    assert not result  # Should return False when hitting boundary


def test_scanning_none_cell() -> None:
    """Test scanning returns False when starting cell is None."""
    obs = np.array([[1, 2], [3, 4]])

    def true_condition(  # pylint: disable=unused-argument
        cell: tuple[int, int] | None, obs: np.ndarray
    ) -> bool:
        return True  # Would return True if called

    def false_condition(  # pylint: disable=unused-argument
        cell: tuple[int, int] | None, obs: np.ndarray
    ) -> bool:
        return False

    result = scanning((1, 0), true_condition, false_condition, None, obs)
    assert not result  # Should return False immediately for None cell


def test_scanning_timeout() -> None:
    """Test scanning respects max_timeout parameter."""
    obs = np.array([[1] * 100])  # Very wide grid

    def true_condition(  # pylint: disable=unused-argument
        cell: tuple[int, int] | None, obs: np.ndarray
    ) -> bool:
        return False  # Never find true

    def false_condition(  # pylint: disable=unused-argument
        cell: tuple[int, int] | None, obs: np.ndarray
    ) -> bool:
        return False  # Never stop early

    # Start at (0, 0) and scan right with small timeout
    result = scanning(
        (0, 1), true_condition, false_condition, (0, 0), obs, max_timeout=5
    )
    assert not result  # Should timeout and return False
