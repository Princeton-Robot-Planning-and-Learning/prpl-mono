"""Unit tests for spatialmath <-> PyBullet pose conversions."""

import numpy as np
from spatialmath import SE3

from prpl_kinematics.geometry import pose_from_pybullet, pose_to_pybullet


def test_identity_orientation_round_trip():
    """A pure translation with identity orientation round-trips exactly."""
    position = (1.0, 2.0, 3.0)
    orientation = (0.0, 0.0, 0.0, 1.0)
    pose = pose_from_pybullet(position, orientation)
    assert np.allclose(pose.t, position)
    assert np.allclose(pose.R, np.eye(3))
    out_position, out_orientation = pose_to_pybullet(pose)
    assert np.allclose(out_position, position)
    assert np.allclose(out_orientation, orientation)


def test_z_rotation_maps_x_to_y():
    """A 90-degree rotation about +z sends the +x axis to +y."""
    half_sqrt2 = np.sqrt(2) / 2
    pose = pose_from_pybullet((0.0, 0.0, 0.0), (0.0, 0.0, half_sqrt2, half_sqrt2))
    rotated = pose.R @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(rotated, [0.0, 1.0, 0.0], atol=1e-6)


def test_arbitrary_pose_round_trip():
    """An arbitrary SE3 survives a round-trip through PyBullet conventions."""
    pose = SE3.Rt(SE3.RPY([0.3, -0.7, 1.1]).R, [0.5, -1.0, 2.0])
    position, orientation = pose_to_pybullet(pose)
    recovered = pose_from_pybullet(position, orientation)
    assert np.allclose(recovered.A, pose.A, atol=1e-6)
