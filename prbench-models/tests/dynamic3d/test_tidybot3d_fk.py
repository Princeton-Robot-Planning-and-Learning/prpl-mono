"""Tests for the TidyBot3D forward kinematics solver (FKSolver)."""

import numpy as np

from prbench_models.dynamic3d.fk_solver import TidybotFKSolver


def test_forward_kinematics():
    """Test that the TidybotFKSolver returns the correct end-effector pose for the home
    position."""
    fk = TidybotFKSolver(ee_offset=0.0)
    home_qpos = np.deg2rad([0, 15, 180, -130, 0, 55, 90])
    expected_home_pos = np.array([0.456, 0.0, 0.314])
    expected_home_quat = np.array([0.5, 0.5, 0.5, 0.5])
    pos, quat = fk.forward_kinematics(home_qpos)
    assert np.allclose(pos, expected_home_pos, atol=1e-2)
    assert np.allclose(quat, expected_home_quat, atol=5*1e-2)

    retract_qpos = np.deg2rad([0, -20, 180, -146, 0, -50, 90])
    expected_retract_pos = np.array([0.12, 0.0, 0.209])
    expected_retract_quat = np.array([0.707, 0.707, 0.0, 0.0])
    pos, quat = fk.forward_kinematics(retract_qpos)
    assert np.allclose(pos, expected_retract_pos, atol=1e-2)
    assert np.allclose(quat, expected_retract_quat, atol=5*1e-2)
