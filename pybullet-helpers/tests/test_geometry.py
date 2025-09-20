"""Tests for geometry PyBullet helper utilities."""

import numpy as np
import pybullet as p

from pybullet_helpers.geometry import (
    Pose,
    get_pose,
    interpolate_poses,
    iter_between_poses,
    matrix_from_quat,
    rotate_pose,
)


def test_pose():
    """Tests for Pose()."""
    position = (5.0, 0.5, 1.0)
    orientation = (0.0, 0.0, 0.0, 1.0)
    pose = Pose(position, orientation)
    rpy = pose.rpy
    reconstructed_pose = Pose.from_rpy(position, rpy)
    assert pose.allclose(reconstructed_pose)
    unit_pose = Pose.identity()
    assert not pose.allclose(unit_pose)
    multiplied_pose = pose.multiply(unit_pose, unit_pose, unit_pose)
    assert pose.allclose(multiplied_pose)
    inverted_pose = pose.invert()
    assert not pose.allclose(inverted_pose)
    assert pose.allclose(inverted_pose.invert())
    matrix = pose.to_matrix()
    assert matrix.shape == (4, 4) and np.allclose(matrix[3, :], [0.0, 0.0, 0.0, 1.0])
    reconstructed_pose = Pose.from_matrix(matrix)
    assert pose.allclose(reconstructed_pose)


def test_matrix_from_quat():
    """Tests for matrix_from_quat()."""
    mat = matrix_from_quat((0.0, 0.0, 0.0, 1.0))
    assert np.allclose(mat, np.eye(3))
    mat = matrix_from_quat((0.0, 0.0, 0.0, -1.0))
    assert np.allclose(mat, np.eye(3))
    mat = matrix_from_quat((1.0, 0.0, 0.0, 1.0))
    expected_mat = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    assert np.allclose(mat, expected_mat)


def test_rotate_pose():
    """Tests for rotate_pose()."""
    pose = Pose.identity()
    new_pose = rotate_pose(pose, roll=np.pi / 2, pitch=np.pi / 3, yaw=np.pi / 4)
    assert np.allclose(new_pose.rpy, [np.pi / 2, np.pi / 3, np.pi / 4])
    inv_pose = new_pose.invert()
    old_pose = rotate_pose(
        new_pose, roll=inv_pose.rpy[0], pitch=inv_pose.rpy[1], yaw=inv_pose.rpy[2]
    )
    assert pose.allclose(old_pose)


def test_get_pose(physics_client_id):
    """Tests for get_pose()."""
    collision_id = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[1, 1, 1], physicsClientId=physics_client_id
    )
    mass = 0
    position = (1.0, 0.0, 3.0)
    orientation = (0.0, 1.0, 0.0, 0.0)
    expected_pose = Pose(position, orientation)
    body = p.createMultiBody(
        mass,
        collision_id,
        basePosition=position,
        baseOrientation=orientation,
        physicsClientId=physics_client_id,
    )
    pose = get_pose(body, physics_client_id)
    assert pose.allclose(expected_pose)


def test_iter_between_poses():
    """Tests for iter_between_poses()."""
    start = Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    end = Pose((0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 0.0))
    poses = list(iter_between_poses(start, end, num_interp=10, include_start=True))
    assert len(poses) == 11
    assert poses[0].allclose(start)
    assert poses[-1].allclose(end)
    poses = list(iter_between_poses(start, end, num_interp=10, include_start=False))
    assert len(poses) == 10
    assert not poses[0].allclose(start)
    assert poses[-1].allclose(end)


def test_interpolate_poses():
    """Tests for interpolate_poses()."""
    start = Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    end = Pose((0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 0.0))
    pose0 = interpolate_poses(start, end, 0.0)
    assert pose0.allclose(start)
    pose1 = interpolate_poses(start, end, 1.0)
    assert pose1.allclose(end)
    pose_middle = interpolate_poses(start, end, 0.5)
    expected = Pose(
        position=(0.0, 0.0, 0.5), orientation=(0.0, 0.0, np.sqrt(2) / 2, np.sqrt(2) / 2)
    )
    assert pose_middle.allclose(expected)
