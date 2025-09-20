"""Math utilities."""

import numpy as np
from numpy.typing import ArrayLike

from pybullet_helpers.geometry import (
    Pose,
    Pose3D,
    Quaternion,
    matrix_from_quat,
    quat_from_matrix,
)


def get_poses_facing_line(
    axis: tuple[float, float, float],
    point_on_line: tuple[float, float, float],
    radius: float,
    num_points: int,
    angle_offset: float = 0.0,
) -> list[Pose]:
    """Generate poses that are rotated around a given line at a given radius,
    facing towards the line.

    "Facing" means that the z dim of the pose is pointing toward the
    line. The x dim is pointing right and the y dim is pointing down.

    A typical use case is generating multiple candidate grasps of an
    object.

    angle_offset is added to each angle around the circle. A typical use
    would be in sampling random poses (with num_points = 1).
    """
    assert np.isclose(np.linalg.norm(axis), 1.0), "axis should have unit norm"

    # Create a vector orthogonal to the axis to define a circle plane.
    if np.allclose(axis, [0, 0, 1]):
        ortho_vector = np.array([1, 0, 0])
    else:
        ortho_vector = np.cross(axis, [0, 0, 1])
        ortho_vector = ortho_vector / np.linalg.norm(ortho_vector)

    # Compute points on the circle.
    poses = []
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points + angle_offset
        position_offset = np.cos(angle) * ortho_vector + np.sin(angle) * np.cross(
            axis, ortho_vector
        )
        position = point_on_line + radius * position_offset

        # Compute the orientation (rotation) quaternion
        z_axis = point_on_line - position  # vector from position to point_on_line
        z_axis /= np.linalg.norm(z_axis)
        x_axis = np.cross(z_axis, axis)
        y_axis = np.cross(z_axis, x_axis)

        rotation_matrix = np.vstack((x_axis, y_axis, z_axis)).T
        orientation = quat_from_matrix(rotation_matrix)

        poses.append(Pose(position=tuple(position), orientation=orientation))

    return poses


def rotate_about_point(point: Pose3D, rotation: Quaternion, current_pose: Pose) -> Pose:
    """Rotate a current pose by a given rotation around a point."""

    # First rotate the current position.
    current_position = current_pose.position

    # Translate the current_position.
    translated_position = np.subtract(current_position, point)

    # Apply the rotation.
    rotation_matrix = matrix_from_quat(rotation)
    rotated_position = rotation_matrix @ translated_position

    # Translate back to original position.
    new_position = tuple(rotated_position + point)

    # Now rotate the current orientation.
    current_orientation_matrix = matrix_from_quat(current_pose.orientation)
    new_orientation_matrix = rotation_matrix @ current_orientation_matrix
    new_orientation = quat_from_matrix(new_orientation_matrix)

    return Pose(new_position, new_orientation)


def geometric_sequence(base: float, length: int, start_value: float = 1.0) -> ArrayLike:
    """E.g., if base = 0.5, then this outputs 1.0, 0.5, 0.25, 0.125, ..."""
    return start_value * np.power(base, np.arange(length))


def sample_within_sphere(
    center: Pose3D, min_radius: float, max_radius: float, rng: np.random.Generator
) -> Pose3D:
    """Sample a random point within a sphere of given radius and center."""
    return sample_on_sphere(center, rng.uniform(min_radius, max_radius), rng)


def sample_on_sphere(center: Pose3D, radius: float, rng: np.random.Generator) -> Pose3D:
    """Sample a random point on a sphere of given radius and center."""
    # Sample a random point on the unit sphere.
    vec = rng.normal(size=(3,))
    vec /= np.linalg.norm(vec, axis=0)

    vec = radius * vec

    # Translate to the center.
    vec = np.add(center, vec)
    return vec.tolist()
