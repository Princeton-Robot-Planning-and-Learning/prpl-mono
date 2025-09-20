"""Tests for the TidyBot3D inverse kinematics solver (IKSolver)."""

import numpy as np

from prbench_models.tidybot.ik_solver import TidybotIKSolver


def test_ik_solver_basic():
    """Test that the TidybotIKSolver returns a valid joint configuration for a simple
    target pose."""
    ik = TidybotIKSolver(ee_offset=0.12)
    target_pos = ik.site_pos.copy()
    target_quat = np.array([0, 0, 0, 1])  # Identity quaternion (x, y, z, w)
    curr_qpos = ik.qpos0.copy()
    result_qpos = ik.solve(target_pos, target_quat, curr_qpos)
    assert result_qpos.shape == curr_qpos.shape
    assert np.all(np.isfinite(result_qpos))


def test_ik_solver_performance_and_accuracy():
    """Test the performance and accuracy of the TidybotIKSolver for a known home pose
    over 1000 iterations."""
    ik_solver = TidybotIKSolver()
    home_pos = np.array([0.456, 0.0, 0.434])
    home_quat = np.array([0.5, 0.5, 0.5, 0.5])
    retract_qpos = np.deg2rad([0, -20, 180, -146, 0, -50, 90])

    for _ in range(1000):
        qpos = ik_solver.solve(home_pos, home_quat, retract_qpos)

    assert qpos.shape == retract_qpos.shape
    assert np.all(np.isfinite(qpos))

    expected_home_deg = np.array([0, 15, 180, -130, 0, 55, 90])
    qpos_deg = np.rad2deg(qpos)
    assert np.allclose(
        qpos_deg, expected_home_deg, atol=5
    ), f"IK solution deviates from expected: {qpos_deg} vs {expected_home_deg}"
