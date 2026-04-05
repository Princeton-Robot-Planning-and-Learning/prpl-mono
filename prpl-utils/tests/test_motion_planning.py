"""Tests for motion_planning.py."""

import numpy as np
import pytest

from prpl_utils.motion_planning import BiRRT


def test_motion_planning():
    """Basic assertion test for BiRRT."""
    # Create dummy functions to pass into BiRRT.
    dummy_sample_fn = lambda x: x
    dummy_extend_fn = lambda x, y: [x, y]
    dummy_collision_fn = lambda x: False
    dummy_distance_fn = lambda x, y: 0.0

    birrt = BiRRT(
        dummy_sample_fn,
        dummy_extend_fn,
        dummy_collision_fn,
        dummy_distance_fn,
        np.random.default_rng(0),
        num_attempts=1,
        num_iters=1,
        smooth_amt=0,
    )

    # Test that query_to_goal_fn for BiRRT raises a NotImplementedError
    with pytest.raises(NotImplementedError):
        birrt.query_to_goal_fn(0, lambda x: False, lambda: 1)
