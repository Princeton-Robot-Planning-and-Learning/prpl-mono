"""Utility functions for TidyBot environments."""

from typing import Any

import numpy as np
from numpy.typing import NDArray


def get_table_bbox(
    pos: NDArray[np.float32], table_config: dict[str, Any]
) -> list[float]:
    """Get bounding box for a table given its position and config.

    Args:
        pos: Position of the table as [x, y, z] array
        table_config: Dictionary containing table configuration with keys:
            - "shape": Shape of the table - "rectangle" or "circle"
            - "length": Total table length in meters (for rectangle)
            - "width": Total table width in meters (for rectangle)
            - "diameter": Diameter of circular table in meters (for circle)

    Returns:
        Bounding box as [x_min, y_min, x_max, y_max]

    Raises:
        ValueError: If table shape is not supported
    """
    if table_config["shape"] == "rectangle":
        half_length = table_config["length"] / 2
        half_width = table_config["width"] / 2
        return [
            pos[0] - half_length,  # x_min
            pos[1] - half_width,  # y_min
            pos[0] + half_length,  # x_max
            pos[1] + half_width,  # y_max
        ]
    if table_config["shape"] == "circle":
        radius = table_config["diameter"] / 2
        return [
            pos[0] - radius,  # x_min
            pos[1] - radius,  # y_min
            pos[0] + radius,  # x_max
            pos[1] + radius,  # y_max
        ]
    raise ValueError(f"Unknown table shape: {table_config['shape']}")


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


def sample_collision_free_position(
    table_config: dict[str, Any],
    placed_bboxes: list[list[float]],
    np_random: np.random.Generator,
    max_attempts: int = 100,
    x_range: tuple[float, float] = (-2.0, 2.0),
    y_range: tuple[float, float] = (0.5, 2.5),
) -> NDArray[np.float32]:
    """Sample a collision-free position for a table.

    Args:
        table_config: Dictionary containing table configuration
        placed_bboxes: List of bounding boxes for already placed tables
        np_random: Random number generator
        max_attempts: Maximum number of sampling attempts
        x_range: Range for x coordinate sampling as (min, max)
        y_range: Range for y coordinate sampling as (min, max)

    Returns:
        Position as [x, y, z] array where z is always 0.0

    Raises:
        None: Returns fallback position with warning if no collision-free position found
    """
    for _ in range(max_attempts):
        # Sample a candidate position
        candidate_pos = np.array(
            [
                np_random.uniform(x_range[0], x_range[1]),  # x coordinate
                np_random.uniform(y_range[0], y_range[1]),  # y coordinate
                0.0,  # z coordinate (fixed at 0)
            ]
        )

        # Get bounding box for this candidate position
        candidate_bbox = get_table_bbox(candidate_pos, table_config)

        # Check if it collides with any existing table
        collision = False
        for existing_bbox in placed_bboxes:
            if bboxes_overlap(candidate_bbox, existing_bbox):
                collision = True
                break

        # If no collision, return this position
        if not collision:
            return candidate_pos

    # If we couldn't find a collision-free position after max_attempts,
    # return a fallback position (this shouldn't happen often with reasonable
    # table sizes)
    print(
        f"Warning: Could not find collision-free position after {max_attempts} attempts"
    )
    return np.array(
        [
            np_random.uniform(x_range[0], x_range[1]),
            np_random.uniform(y_range[0], y_range[1]),
            0.0,
        ]
    )


def sample_pose_in_region(
    regions: list[list[float]],
    np_random: np.random.Generator,
    z_coordinate: float = 0.02,
) -> tuple[float, float, float]:
    """Sample a pose (x, y, z) uniformly randomly from one of the provided regions.

    Args:
        regions: List of bounding boxes, where each bounding box is a list of 4 floats:
                [x_start, y_start, x_end, y_end]
        np_random: Random number generator
        z_coordinate: Z coordinate for the sampled pose (height above ground)

    Returns:
        Tuple of (x, y, z) coordinates sampled from one of the regions

    Raises:
        ValueError: If regions list is empty or if any region has invalid bounds
    """
    if not regions:
        raise ValueError("Regions list cannot be empty")

    # Randomly select one of the regions
    selected_region = np_random.choice(regions)

    # Validate the selected region
    if len(selected_region) != 4:
        raise ValueError(
            f"Each region must have exactly 4 values [x_start, y_start, x_end, y_end], "
            f"got {len(selected_region)}"
        )

    x_start, y_start, x_end, y_end = selected_region

    # Validate bounds
    if x_start >= x_end:
        raise ValueError(f"x_start ({x_start}) must be less than x_end ({x_end})")
    if y_start >= y_end:
        raise ValueError(f"y_start ({y_start}) must be less than y_end ({y_end})")

    # Sample uniformly within the selected region
    x = np_random.uniform(x_start, x_end)
    y = np_random.uniform(y_start, y_end)

    return (x, y, z_coordinate)
