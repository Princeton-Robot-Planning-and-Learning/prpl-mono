"""Utility functions for TidyBot environments."""

import numpy as np
import transforms3d.euler as t3d_euler  # type: ignore[import-untyped]
from numpy.typing import NDArray


def euler2mat_rzxy(
    angle_z: float, angle_x: float, angle_y: float
) -> NDArray[np.float64]:
    """Convert Euler angles to rotation matrix using RZXY convention.

    Args:
        angle_z: Z-rotation angle in radians
        angle_x: X-rotation angle in radians
        angle_y: Y-rotation angle in radians

    Returns:
        3x3 rotation matrix
    """
    # Create rotation matrices inline
    cz, sz = np.cos(angle_z), np.sin(angle_z)
    cx, sx = np.cos(angle_x), np.sin(angle_x)
    cy, sy = np.cos(angle_y), np.sin(angle_y)

    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)

    return rz @ rx @ ry  # type: ignore[operator]


def mat2euler_rxyz(
    rotation_matrix: NDArray[np.float64],
) -> tuple[float, float, float]:
    """Convert rotation matrix to Euler angles using RXYZ convention.

    Args:
        rotation_matrix: 3x3 rotation matrix

    Returns:
        Tuple of (roll, pitch, yaw) angles in radians
    """
    r = rotation_matrix

    # Extract angles from rotation matrix for RXYZ convention
    # R = Rz(yaw) * Ry(pitch) * Rx(roll)
    # This is the standard aerospace convention

    # Check for gimbal lock
    sin_pitch = -r[2, 0]
    sin_pitch = np.clip(sin_pitch, -1.0, 1.0)
    pitch = float(np.arcsin(sin_pitch))

    cos_pitch = np.cos(pitch)
    if abs(cos_pitch) > 1e-6:  # Not in gimbal lock
        roll = float(np.arctan2(r[2, 1], r[2, 2]))
        yaw = float(np.arctan2(r[1, 0], r[0, 0]))
    else:  # Gimbal lock case
        roll = 0.0
        yaw = float(np.arctan2(-r[0, 1], r[1, 1]))

    return (roll, pitch, yaw)


def convert_yaw_to_quaternion(yaw: float) -> list[float]:
    """Convert yaw angle (in radians) to quaternion representation.

    Args:
        yaw: Yaw angle in radians

    Returns:
        Quaternion as a list [w, x, y, z]
    """
    half_yaw = yaw / 2
    return [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)]  # w, x, y, z


def compute_camera_euler(
    position: list[float], lookat: list[float]
) -> tuple[float, float, float]:
    """Compute euler angles for camera to look at target.

    Args:
        position: Camera position [x, y, z]
        lookat: Target position to look at [x, y, z]

    Returns:
        Euler angles (roll, pitch, yaw) in radians for MuJoCo's XYZ convention.
        After rotation, the camera's -Z axis will point from position to lookat,
        and the +X axis will have a convex angle with the global -Z direction.
    """
    pos_array: NDArray[np.float64] = np.array(position, dtype=np.float64)
    lookat_array: NDArray[np.float64] = np.array(lookat, dtype=np.float64)

    # Direction vector from target to camera (where +Z of camera should point)
    direction: NDArray[np.float64] = pos_array - lookat_array
    direction_norm = np.linalg.norm(direction)
    if direction_norm < 1e-6:
        # Camera and target are at the same position, default to looking forward
        return (0.0, 0.0, 0.0)

    # Step 1: Convert direction to spherical coordinates
    # Spherical coordinates: (r, theta, phi)
    # - r: radial distance
    # - theta (polar angle): angle from positive z-axis [0, π]
    # - phi (azimuthal angle): angle in xy-plane from positive x-axis [0, 2π)

    r = direction_norm
    theta = float(np.arccos(np.clip(direction[2] / r, -1.0, 1.0)))  # polar angle
    phi = float(np.arctan2(direction[1], direction[0]))  # azimuthal angle

    # Step 2: Convert spherical coordinates to Euler angles
    # euler_zxy represents (Z-rotation, X-rotation, Y-rotation)
    euler_zxy = (np.pi / 2 + phi, theta, 0)
    rot_mat_zxy = euler2mat_rzxy(euler_zxy[0], euler_zxy[1], euler_zxy[2])
    euler_xyz = mat2euler_rxyz(rot_mat_zxy)

    # Convert to XYZ (roll, pitch, yaw)
    # roll = X-rotation, pitch = Y-rotation, yaw = Z-rotation
    roll = euler_xyz[0]  # X-rotation = theta
    pitch = euler_xyz[1]  # Y-rotation = 0
    yaw = euler_xyz[2]  # Z-rotation = π/2 + phi

    return (roll, pitch, yaw)


def point_in_bbox_3d(
    position: NDArray[np.float32],
    bbox: list[float],
) -> bool:
    """Check if a 3D position is inside a 3D bounding box.

    Args:
        position: Position as [x, y, z] array
        bbox: Bounding box as [x_min, y_min, z_min, x_max, y_max, z_max]

    Returns:
        True if position is inside the bounding box, False otherwise
    """
    x, y, z = position
    x_min, y_min, z_min, x_max, y_max, z_max = bbox
    return x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max


def sample_pose_in_bbox_3d(
    bbox: list[float],
    np_random: np.random.Generator,
    yaw_range_deg: tuple[float, float] = (0.0, 360.0),
) -> tuple[float, float, float, float]:
    """Sample a pose uniformly from a 3D bounding box.

    Args:
        bbox: Bounding box as [x_min, y_min, z_min, x_max, y_max, z_max]
        np_random: Random number generator
        yaw_range_deg: Yaw range in degrees (min, max)

    Returns:
        Tuple of (x, y, z, yaw) where yaw is in radians
    """
    x_min, y_min, z_min, x_max, y_max, z_max = bbox

    # Sample position uniformly within the bounding box
    x = np_random.uniform(x_min, x_max)
    y = np_random.uniform(y_min, y_max)
    z = np_random.uniform(z_min, z_max)

    # Sample yaw
    yaw_deg = np_random.uniform(yaw_range_deg[0], yaw_range_deg[1])
    yaw = np.radians(yaw_deg)

    return (x, y, z, yaw)


def bboxes_overlap(
    bbox1: list[float], bbox2: list[float], margin: float = 0.001
) -> bool:
    """Check if two bounding boxes overlap with a safety margin.

    Args:
        bbox1: First bounding box as [x_min, y_min, x_max, y_max]
        bbox2: Second bounding box as [x_min, y_min, x_max, y_max]
        margin: Safety margin in meters to add between bounding boxes

    Returns:
        True if bounding boxes overlap (including margin), False otherwise
    """
    if len(bbox1) == 4:
        assert len(bbox2) == 4
        return not (
            bbox1[2] + margin <= bbox2[0]  # bbox1 right + margin <= bbox2 left
            or bbox2[2] + margin <= bbox1[0]  # bbox2 right + margin <= bbox1 left
            or bbox1[3] + margin <= bbox2[1]  # bbox1 top + margin <= bbox2 bottom
            or bbox2[3] + margin <= bbox1[1]
        )  # bbox2 top + margin <= bbox1 bottom
    if len(bbox1) == 6:
        assert len(bbox2) == 6
        return not (
            bbox1[3] + margin <= bbox2[0]  # bbox1 x_max + margin <= bbox2 x_min
            or bbox2[3] + margin <= bbox1[0]  # bbox2 x_max + margin <= bbox1 x_min
            or bbox1[4] + margin <= bbox2[1]  # bbox1 y_max + margin <= bbox2 y_min
            or bbox2[4] + margin <= bbox1[1]  # bbox2 y_max + margin <= bbox1 y_min
            or bbox1[5] + margin <= bbox2[2]  # bbox1 z_max + margin <= bbox2 z_min
            or bbox2[5] + margin <= bbox1[2]
        )  # bbox2 z_max + margin <= bbox1 z_min
    raise ValueError("Bounding boxes must be of length 4 or 6.")


def translate_bounding_box(
    bbox: list[float], translation: NDArray[np.float32]
) -> list[float]:
    """Translate a bounding box by a given translation vector.

    Args:
        bbox: Bounding box as [x_min, y_min, z_min, x_max, y_max, z_max]
        translation: Translation vector as [dx, dy, dz] array

    Returns:
        Translated bounding box as [x_min, y_min, z_min, x_max, y_max, z_max]
    """
    dx, dy, dz = translation
    return [
        bbox[0] + dx,  # x_min
        bbox[1] + dy,  # y_min
        bbox[2] + dz,  # z_min
        bbox[3] + dx,  # x_max
        bbox[4] + dy,  # y_max
        bbox[5] + dz,  # z_max
    ]


def rotate_bounding_box_2d(
    bbox: list[float], yaw: float, center: tuple[float, float]
) -> list[float]:
    """Rotate a bounding box around a center point in 2D (yaw rotation only).

    This function rotates the bounding box corners and computes the new axis-aligned
    bounding box that contains all rotated corners.

    Args:
        bbox: Bounding box as [x_min, y_min, z_min, x_max, y_max, z_max]
        yaw: Rotation angle in radians (around z-axis)
        center: Center of rotation as (cx, cy)

    Returns:
        Rotated bounding box as [x_min, y_min, z_min, x_max, y_max, z_max]
    """
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    cx, cy = center

    # Get the four corners of the original bounding box (in 2D)
    corners = [
        (bbox[0], bbox[1]),  # bottom-left
        (bbox[3], bbox[1]),  # bottom-right
        (bbox[3], bbox[4]),  # top-right
        (bbox[0], bbox[4]),  # top-left
    ]

    # Rotate each corner around the center
    rotated_corners = []
    for x, y in corners:
        # Translate to origin
        x_rel = x - cx
        y_rel = y - cy

        # Rotate
        x_rot = x_rel * cos_yaw - y_rel * sin_yaw
        y_rot = x_rel * sin_yaw + y_rel * cos_yaw

        # Translate back
        rotated_corners.append((x_rot + cx, y_rot + cy))

    # Find the new axis-aligned bounding box
    x_coords = [corner[0] for corner in rotated_corners]
    y_coords = [corner[1] for corner in rotated_corners]

    return [
        min(x_coords),  # x_min
        min(y_coords),  # y_min
        bbox[2],  # z_min (unchanged)
        max(x_coords),  # x_max
        max(y_coords),  # y_max
        bbox[5],  # z_max (unchanged)
    ]
