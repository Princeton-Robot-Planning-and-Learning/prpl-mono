"""Utility functions for TidyBot environments."""

import math

import numpy as np
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
    # Implementation copied from transforms3d.euler.euler2mat

    # Axis constants for Euler angle conversions
    _NEXT_AXIS = [1, 2, 0, 1]

    ai, aj, ak = angle_z, angle_x, angle_y
    firstaxis, parity, repetition, frame = (1, 1, 0, 1)

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, ci = math.sin(ai), math.cos(ai)
    sj, cj = math.sin(aj), math.cos(aj)
    sk, ck = math.sin(ak), math.cos(ak)

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    mat = np.eye(3)
    if repetition:
        mat[i, i] = cj
        mat[i, j] = sj * si
        mat[i, k] = sj * ci
        mat[j, i] = sj * sk
        mat[j, j] = -cj * (si * sk) + (ci * ck)
        mat[j, k] = -cj * (ci * sk) - (si * ck)
        mat[k, i] = -sj * ck
        mat[k, j] = cj * (si * ck) + (ci * sk)
        mat[k, k] = cj * (ci * ck) - (si * sk)
    else:
        mat[i, i] = cj * ck
        mat[i, j] = sj * (si * ck) - (ci * sk)
        mat[i, k] = sj * (ci * ck) + (si * sk)
        mat[j, i] = cj * sk
        mat[j, j] = sj * (si * sk) + (ci * ck)
        mat[j, k] = sj * (ci * sk) - (si * ck)
        mat[k, i] = -sj
        mat[k, j] = cj * si
        mat[k, k] = cj * ci

    return mat.astype(np.float64)


def mat2euler_rxyz(
    rotation_matrix: NDArray[np.float64],
) -> tuple[float, float, float]:
    """Convert rotation matrix to Euler angles using RXYZ convention.

    Args:
        rotation_matrix: 3x3 rotation matrix

    Returns:
        Tuple of (roll, pitch, yaw) angles in radians
    """
    # Implementation copied from transforms3d.euler.mat2euler

    _EPS4 = 8.881784197001252e-16

    # Axis constants for Euler angle conversions
    _NEXT_AXIS = [1, 2, 0, 1]

    firstaxis, parity, repetition, frame = (2, 1, 0, 1)

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    mat = np.array(rotation_matrix, dtype=np.float64, copy=False)[:3, :3]

    if repetition:
        sy = math.sqrt(mat[i, j] * mat[i, j] + mat[i, k] * mat[i, k])
        if sy > _EPS4:
            ax = math.atan2(mat[i, j], mat[i, k])
            ay = math.atan2(sy, mat[i, i])
            az = math.atan2(mat[j, i], -mat[k, i])
        else:
            ax = math.atan2(-mat[j, k], mat[j, j])
            ay = math.atan2(sy, mat[i, i])
            az = 0.0
    else:
        cy = math.sqrt(mat[i, i] * mat[i, i] + mat[j, i] * mat[j, i])
        if cy > _EPS4:
            ax = math.atan2(mat[k, j], mat[k, k])
            ay = math.atan2(-mat[k, i], cy)
            az = math.atan2(mat[j, i], mat[i, i])
        else:
            ax = math.atan2(-mat[j, k], mat[j, j])
            ay = math.atan2(-mat[k, i], cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax

    return (float(ax), float(ay), float(az))


def convert_yaw_to_quaternion(yaw: float) -> list[float]:
    """Convert yaw angle (in radians) to quaternion representation.

    Args:
        yaw: Yaw angle in radians

    Returns:
        Quaternion as a list [w, x, y, z]
    """
    half_yaw = yaw / 2
    return [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)]  # w, x, y, z


def quat_to_yaw(quat: NDArray[np.float32] | list[float]) -> float:
    """Convert quaternion to yaw angle (rotation around z-axis).

    Args:
        quat: Quaternion as [w, x, y, z]

    Returns:
        Yaw angle in radians
    """
    w, _x, _y, z = quat
    # Extract yaw from quaternion using atan2
    # For a pure rotation around z-axis: yaw = 2 * atan2(z, w)
    yaw = 2.0 * np.arctan2(z, w)
    return float(yaw)


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
    for _x, _y in corners:
        # Translate to origin
        x_rel = _x - cx
        y_rel = _y - cy

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
