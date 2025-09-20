"""Tests for spaces.py."""

import numpy as np

from pybullet_helpers.spaces import PoseSpace


def test_pose_space():
    """Tests for PoseSpace()."""
    pose_space = PoseSpace(
        -1, 1, -2, 2, -3, 3, -np.pi, -np.pi + 0.1, 0, 0.1, np.pi - 0.1, np.pi
    )
    pose_space.seed(123)
    for _ in range(100):
        sample = pose_space.sample()
        assert pose_space.contains(sample)
