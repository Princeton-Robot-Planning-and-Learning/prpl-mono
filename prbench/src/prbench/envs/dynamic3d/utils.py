"""Utility functions for TidyBot environments."""

import numpy as np
from numpy.typing import NDArray


def convert_yaw_to_quaternion(yaw: float) -> list[float]:
    """Convert yaw angle (in radians) to quaternion representation.

    Args:
        yaw: Yaw angle in radians

    Returns:
        Quaternion as a list [w, x, y, z]
    """
    half_yaw = yaw / 2
    return [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)]  # w, x, y, z


def bboxes_overlap(bbox1: list[float], bbox2: list[float], margin: float = 0.2) -> bool:
    """Check if two bounding boxes overlap with a safety margin.

    Args:
        bbox1: First bounding box as [x_min, y_min, x_max, y_max]
        bbox2: Second bounding box as [x_min, y_min, x_max, y_max]
        margin: Safety margin in meters to add between bounding boxes

    Returns:
        True if bounding boxes overlap (including margin), False otherwise
    """
    return not (
        bbox1[2] + margin <= bbox2[0]  # bbox1 right + margin <= bbox2 left
        or bbox2[2] + margin <= bbox1[0]  # bbox2 right + margin <= bbox1 left
        or bbox1[3] + margin <= bbox2[1]  # bbox1 top + margin <= bbox2 bottom
        or bbox2[3] + margin <= bbox1[1]
    )  # bbox2 top + margin <= bbox1 bottom


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
