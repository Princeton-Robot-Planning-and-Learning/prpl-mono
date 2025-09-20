"""Tests for math_utils.py."""

import numpy as np
import pybullet as p

from pybullet_helpers.geometry import Pose
from pybullet_helpers.math_utils import (
    geometric_sequence,
    get_poses_facing_line,
    rotate_about_point,
)


def test_get_poses_facing_line():
    """Tests for get_poses_facing_line()."""

    axis = (0.0, 0.0, 1.0)
    point_on_line = (0.0, 0.0, 0.0)
    radius = 1.0
    poses = get_poses_facing_line(axis, point_on_line, radius, num_points=4)
    expected = [
        Pose(position=(1.0, 0.0, 0.0), orientation=(0.5, 0.5, -0.5, -0.5)),
        Pose(position=(0.0, 1.0, 0.0), orientation=(0.0, 0.7071, -0.7071, 0.0)),
        Pose(position=(-1.0, 0.0, 0.0), orientation=(-0.5, 0.5, -0.5, 0.5)),
        Pose(position=(0.0, -1.0, -0.0), orientation=(-0.7071, 0.0, 0.0, 0.7071)),
    ]
    for pose, exp in zip(poses, expected, strict=True):
        assert pose.allclose(exp)

    axis = (0.0, 0.0, 1.0)
    point_on_line = (10.0, 0.0, 0.0)
    radius = 2.0
    poses = get_poses_facing_line(axis, point_on_line, radius, num_points=4)
    expected = [
        Pose(position=(12.0, 0.0, 0.0), orientation=(0.5, 0.5, -0.5, -0.5)),
        Pose(position=(10.0, 2.0, 0.0), orientation=(0.0, 0.7071, -0.7071, 0.0)),
        Pose(position=(8.0, 0.0, 0.0), orientation=(-0.5, 0.5, -0.5, 0.5)),
        Pose(position=(10.0, -2.0, -0.0), orientation=(-0.7071, 0.0, 0.0, 0.7071)),
    ]
    for pose, exp in zip(poses, expected, strict=True):
        assert pose.allclose(exp)

    axis = (1.0, 0.0, 0.0)
    point_on_line = (0.0, 0.0, 0.0)
    radius = 1.0
    poses = get_poses_facing_line(axis, point_on_line, radius, num_points=4)
    expected = [
        Pose(position=(0.0, -1.0, 0.0), orientation=(0.5, -0.5, -0.5, -0.5)),
        Pose(position=(0.0, 0.0, -1.0), orientation=(0.0, 0.0, 0.707107, 0.707107)),
        Pose(position=(0.0, 1.0, 0.0), orientation=(0.5, -0.5, 0.5, 0.5)),
        Pose(position=(0.0, 0.0, 1.0), orientation=(0.707107, -0.707107, 0.0, 0.0)),
    ]
    for pose, exp in zip(poses, expected, strict=True):
        assert pose.allclose(exp)

    # Test angle_offset.
    axis = (0.0, 0.0, 1.0)
    point_on_line = (0.0, 0.0, 0.0)
    radius = 1.0
    poses = get_poses_facing_line(
        axis, point_on_line, radius, num_points=4, angle_offset=np.pi / 2
    )
    expected = [
        Pose(position=(0.0, 1.0, 0.0), orientation=(0.0, 0.707107, -0.707107, 0.0)),
        Pose(position=(-1.0, 0.0, 0.0), orientation=(-0.5, 0.5, -0.5, 0.5)),
        Pose(position=(0.0, -1.0, 0.0), orientation=(-0.707107, 0.0, 0.0, 0.707107)),
        Pose(position=(1.0, 0.0, 0.0), orientation=(-0.5, -0.5, 0.5, 0.5)),
    ]
    for pose, exp in zip(poses, expected, strict=True):
        assert pose.allclose(exp)

    # Uncomment to debug.
    # from pybullet_helpers.gui import create_gui_connection, visualize_pose

    # physics_client_id = create_gui_connection(camera_target=point_on_line)
    # p.addUserDebugLine(
    #     lineFromXYZ=point_on_line,
    #     lineToXYZ=np.add(point_on_line, axis),
    #     lineColorRGB=(1.0, 1.0, 1.0),
    #     lifeTime=0,
    #     physicsClientId=physics_client_id,
    # )
    # for pose in poses:
    #     visualize_pose(pose, physics_client_id)
    # while True:
    #     p.stepSimulation(physicsClientId=physics_client_id)


def test_rotate_about_point():
    """Tests for rotate_about_point()."""

    point = (0.0, 0.0, 0.0)
    current_pose = Pose((1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    rotation = p.getQuaternionFromEuler((0.0, 0.0, np.pi))
    rotated_pose = rotate_about_point(point, rotation, current_pose)
    assert np.allclose(rotated_pose.position, (-1.0, 0.0, 0.0))
    assert np.allclose(rotated_pose.orientation, (0.0, 0.0, 1.0, 0.0))

    # Uncomment to debug.
    # from pybullet_helpers.gui import create_gui_connection, visualize_pose

    # physics_client_id = create_gui_connection(camera_target=point)
    # p.addUserDebugPoints(
    #     pointPositions=[point],
    #     pointColorsRGB=[(1.0, 1.0, 1.0)],
    #     pointSize=10.0,
    #     lifeTime=0,
    #     physicsClientId=physics_client_id,
    # )
    # visualize_pose(current_pose, physics_client_id)
    # visualize_pose(rotated_pose, physics_client_id)
    # while True:
    #     p.stepSimulation(physicsClientId=physics_client_id)


def test_geometric_sequence():
    """Tests for geometric_sequence()."""
    assert np.allclose(geometric_sequence(0.5, 4), [1.0, 0.5, 0.25, 0.125])
    assert np.allclose(geometric_sequence(0.5, 3, start_value=0.5), [0.5, 0.25, 0.125])
