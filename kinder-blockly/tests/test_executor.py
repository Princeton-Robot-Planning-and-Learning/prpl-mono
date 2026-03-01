"""Tests for the Blockly program executor."""

import numpy as np

from kinder_blockly.executor import execute_program


def test_move_base_to_single_target():
    """Executing a single move_base_to_target block yields frames."""
    program = {"blocks": [{"type": "move_base_to_target", "x": 0.5, "y": 0.5}]}
    frames = list(execute_program(program, seed=123))
    assert len(frames) > 1
    assert isinstance(frames[0], np.ndarray)
    assert frames[0].ndim == 3


def test_move_base_to_multiple_targets():
    """Multiple move blocks execute in sequence."""
    program = {
        "blocks": [
            {"type": "move_base_to_target", "x": 0.5, "y": 0.0},
            {"type": "move_base_to_target", "x": 0.5, "y": 0.5},
            {"type": "move_base_to_target", "x": 0.0, "y": 0.0},
        ]
    }
    frames = list(execute_program(program, seed=42))
    assert len(frames) > 3
