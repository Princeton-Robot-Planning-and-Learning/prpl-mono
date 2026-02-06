"""Tests for the TidyBot3D FK and IK solvers."""

import numpy as np

from kinder_models.dynamic3d.fk_solver import TidybotFKSolver
from kinder_models.dynamic3d.ik_solver import TidybotIKSolver


def test_ik_fk_roundtrip():
    """Test that IK followed by FK returns the original target pose."""
    ee_offset = 0.12  # Use same offset for both!

    fk = TidybotFKSolver(ee_offset=ee_offset)
    ik = TidybotIKSolver(ee_offset=ee_offset)

    # Target pose
    target_pos = np.array([0.576, 0.0, 0.314])
    target_quat = np.array([0.5, 0.5, 0.5, 0.5])  # (x, y, z, w)

    # Initial joint configuration
    init_qpos = np.deg2rad([0, -20, 180, -146, 0, -50, 90])

    # IK: pose -> joints
    result_qpos = ik.solve(target_pos, target_quat, init_qpos)

    # FK: joints -> pose
    fk_pos, fk_quat = fk.forward_kinematics(result_qpos)

    # Should match original target
    assert np.allclose(
        fk_pos, target_pos, atol=1e-3
    ), f"Position mismatch: {fk_pos} vs {target_pos}"
    assert np.allclose(
        fk_quat, target_quat, atol=1e-2
    ), f"Quaternion mismatch: {fk_quat} vs {target_quat}"

    retract_pos = np.array([0.12, 0.0, 0.209])
    retract_quat = np.array([0.707, 0.707, 0.0, 0.0])
    result_qpos = ik.solve(retract_pos, retract_quat, init_qpos)
    assert result_qpos.shape == init_qpos.shape
    assert np.all(np.isfinite(result_qpos))

    fk_pos, fk_quat = fk.forward_kinematics(result_qpos)
    assert np.allclose(
        fk_pos, retract_pos, atol=1e-3
    ), f"Position mismatch: {fk_pos} vs {retract_pos}"
    assert np.allclose(
        fk_quat, retract_quat, atol=1e-2
    ), f"Quaternion mismatch: {fk_quat} vs {retract_quat}"
