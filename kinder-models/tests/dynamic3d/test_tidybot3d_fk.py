"""Tests for the TidyBot3D forward kinematics solver (FKSolver)."""

import numpy as np

from kinder_models.dynamic3d.fk_solver import TidybotFKSolver


def test_forward_kinematics():
    """Test that the TidybotFKSolver returns the correct end-effector pose for the home
    position."""
    fk = TidybotFKSolver(ee_offset=0.12)
    home_qpos = np.deg2rad([0, 15, 180, -130, 0, 55, 90])
    expected_home_pos = np.array([0.576, 0.0, 0.434])
    expected_home_quat = np.array([0.5, 0.5, 0.5, 0.5])
    pos, quat = fk.forward_kinematics(home_qpos)
    assert np.allclose(pos, expected_home_pos, atol=1e-2)
    assert np.allclose(quat, expected_home_quat, atol=5 * 1e-2)
