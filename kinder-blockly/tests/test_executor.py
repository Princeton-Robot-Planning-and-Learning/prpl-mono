"""Tests for the Blockly program executor."""

import numpy as np

from kinder_blockly.executor import execute_program


def test_move_base_to_target_solves_task():
    """Executing move_base_to_target should reach the goal."""
    program = {"blocks": [{"type": "move_base_to_target"}]}
    frames = list(execute_program(program, seed=123))
    assert len(frames) > 0
    assert isinstance(frames[0], np.ndarray)
    assert frames[0].ndim == 3
