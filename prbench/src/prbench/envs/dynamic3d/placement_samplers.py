"""Placement sampling utilities for dynamic3d environments."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from prbench.envs.dynamic3d import utils
from prbench.envs.dynamic3d.objects import get_fixture_class

# Default yaw range in degrees (full rotation)
DEFAULT_YAW_RANGE = (0.0, 360.0)


def sample_collision_free_positions(
    fixtures: dict[str, dict[str, dict[str, Any]]],
    np_random: np.random.Generator,
    fixture_ranges: dict[str, tuple[float, float, float, float]] | None = None,
    fixture_yaw_ranges: dict[str, tuple[float, float]] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Sample collision-free positions and yaws for multiple fixtures.

    Args:
        fixtures: Dictionary mapping fixture types to dictionaries of fixture
                 configurations (fixture_name -> fixture_config)
        np_random: Random number generator
        fixture_ranges: Dictionary mapping fixture names to sampling ranges as
                       (x_min, y_min, x_max, y_max). If None, uses default range
                       (-2.0, 0.5, 2.0, 2.5) for all fixtures.
        fixture_yaw_ranges: Dictionary mapping fixture names to yaw rotation ranges as
                           (yaw_min, yaw_max) in degrees. If None, uses default range
                           (0.0, 360.0) for all fixtures.

    Returns:
        Dictionary mapping fixture types to dictionaries of fixture poses
        (fixture_name -> {"position": position, "yaw": yaw})
    """
    fixture_poses: dict[str, dict[str, dict[str, Any]]] = {}
    placed_bboxes: list[list[float]] = []

    # Default range if none provided
    default_range = (-2.0, 0.5, 2.0, 2.5)
    default_yaw_range = DEFAULT_YAW_RANGE

    for fixture_type, fixture_configs in fixtures.items():
        fixture_poses[fixture_type] = {}
        for fixture_name, fixture_config in fixture_configs.items():
            # Get the range for this fixture
            if fixture_ranges and fixture_name in fixture_ranges:
                x_min, y_min, x_max, y_max = fixture_ranges[fixture_name]
                x_range = (x_min, x_max)
                y_range = (y_min, y_max)
            else:
                x_min, y_min, x_max, y_max = default_range
                x_range = (x_min, x_max)
                y_range = (y_min, y_max)

            # Get the yaw range for this fixture
            if fixture_yaw_ranges and fixture_name in fixture_yaw_ranges:
                yaw_range = fixture_yaw_ranges[fixture_name]
            else:
                yaw_range = default_yaw_range

            init_bbox = get_fixture_class(fixture_type).get_bounding_box_from_config(
                np.array([0.0, 0.0, 0.0], dtype=np.float32), fixture_config
            )
            # Sample a collision-free position and yaw for each fixture
            position, yaw = sample_collision_free_position(
                list(init_bbox),
                placed_bboxes=placed_bboxes,
                np_random=np_random,
                x_range=x_range,
                y_range=y_range,
                yaw_range=yaw_range,
            )
            bbox = get_fixture_class(fixture_type).get_bounding_box_from_config(
                position, fixture_config
            )
            placed_bboxes.append(list(bbox))
            fixture_poses[fixture_type][fixture_name] = {
                "position": position,
                "yaw": yaw,
            }
    return fixture_poses


def sample_collision_free_position(
    bounding_box_at_origin: list[float],
    placed_bboxes: list[list[float]],
    np_random: np.random.Generator,
    max_attempts: int = 100,
    x_range: tuple[float, float] = (-2.0, 2.0),
    y_range: tuple[float, float] = (0.5, 2.5),
    yaw_range: tuple[float, float] = DEFAULT_YAW_RANGE,
) -> tuple[NDArray[np.float32], float]:
    """Sample a collision-free position and yaw for a fixture.

    Args:
        bounding_box_at_origin: Initial bounding box as
                               [x_min, y_min, z_min, x_max, y_max, z_max]
        placed_bboxes: List of bounding boxes for already placed fixtures
        np_random: Random number generator
        max_attempts: Maximum number of sampling attempts
        x_range: Range for x coordinate sampling as (min, max)
        y_range: Range for y coordinate sampling as (min, max)
        yaw_range: Range for yaw rotation sampling as (min, max) in degrees

    Returns:
        Tuple of (position, yaw) where position is [x, y, z] array (z is always 0.0)
        and yaw is the rotation angle in radians

    Raises:
        None: Returns fallback position with warning if no collision-free position found
    """
    # Get the center of the original bounding box for rotation
    bbox_center_x = (bounding_box_at_origin[0] + bounding_box_at_origin[3]) / 2
    bbox_center_y = (bounding_box_at_origin[1] + bounding_box_at_origin[4]) / 2

    for _ in range(max_attempts):
        # Sample a candidate position
        candidate_pos = np.array(
            [
                np_random.uniform(x_range[0], x_range[1]),  # x coordinate
                np_random.uniform(y_range[0], y_range[1]),  # y coordinate
                0.0,  # z coordinate (fixed at 0)
            ]
        )

        # Sample a random yaw angle from the specified range (in degrees)
        candidate_yaw_deg = np_random.uniform(yaw_range[0], yaw_range[1])
        # Convert to radians for internal use
        candidate_yaw = np.radians(candidate_yaw_deg)

        # Translate the bounding box to the candidate position
        translation = candidate_pos - np.array(
            [bbox_center_x, bbox_center_y, bounding_box_at_origin[2]]
        )
        translated_bbox = utils.translate_bounding_box(
            bounding_box_at_origin, translation
        )

        # Rotate the bounding box around its new center
        new_center = (candidate_pos[0], candidate_pos[1])
        candidate_bbox = utils.rotate_bounding_box_2d(
            translated_bbox, candidate_yaw, new_center
        )

        # Check if it collides with any existing fixture (using 2D overlap for now)
        candidate_bbox_2d = candidate_bbox[:4]  # [x_min, y_min, x_max, y_max]
        collision = False
        for existing_bbox in placed_bboxes:
            existing_bbox_2d = existing_bbox[:4]  # [x_min, y_min, x_max, y_max]
            if utils.bboxes_overlap(candidate_bbox_2d, existing_bbox_2d):
                collision = True
                break

        # If no collision, return this position and yaw
        if not collision:
            return candidate_pos, candidate_yaw

    # If we couldn't find a collision-free position after max_attempts,
    # return a fallback position (this shouldn't happen often with reasonable
    # fixture sizes)
    print(
        f"Warning: Could not find collision-free position after {max_attempts} "
        f"attempts"
    )
    fallback_pos = np.array(
        [
            np_random.uniform(x_range[0], x_range[1]),
            np_random.uniform(y_range[0], y_range[1]),
            0.0,
        ]
    )
    fallback_yaw_deg = np_random.uniform(DEFAULT_YAW_RANGE[0], DEFAULT_YAW_RANGE[1])
    fallback_yaw = np.radians(fallback_yaw_deg)
    return fallback_pos, fallback_yaw


def sample_pose_in_region(
    regions: list[list[float]],
    np_random: np.random.Generator,
    z_coordinate: float = 0.02,
    yaw_ranges: list[tuple[float, float]] | None = None,
) -> tuple[float, float, float, float]:
    """Sample a pose (x, y, z, yaw) uniformly randomly from one of the provided regions.

    Args:
        regions: List of bounding boxes, where each bounding box is a list of 4
                floats: [x_start, y_start, x_end, y_end]
        np_random: Random number generator
        z_coordinate: Z coordinate for the sampled pose (height above ground)
        yaw_ranges: Optional list of yaw rotation ranges as [(yaw_min, yaw_max), ...]
                   in degrees, one for each region. If None, uses full range
                   (0.0, 360.0) for all regions.

    Returns:
        Tuple of (x, y, z, yaw) coordinates sampled from one of the regions,
        where yaw is in radians

    Raises:
        ValueError: If regions list is empty or if any region has invalid bounds
    """
    if not regions:
        raise ValueError("Regions list cannot be empty")

    # Randomly select one of the regions
    selected_region_index = np_random.choice(len(regions))
    selected_region = regions[selected_region_index]

    # Validate the selected region
    if len(selected_region) != 4:
        raise ValueError(
            f"Each region must have exactly 4 values "
            f"[x_start, y_start, x_end, y_end], got {len(selected_region)}"
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

    # Sample yaw from the specified range (convert from degrees to radians)
    yaw_range = DEFAULT_YAW_RANGE
    if yaw_ranges is not None and len(yaw_ranges) > 0:
        # Use the yaw_range corresponding to the selected region index
        yaw_range = yaw_ranges[selected_region_index]
    yaw_deg = np_random.uniform(yaw_range[0], yaw_range[1])
    yaw = np.radians(yaw_deg)

    return (x, y, z_coordinate, yaw)
