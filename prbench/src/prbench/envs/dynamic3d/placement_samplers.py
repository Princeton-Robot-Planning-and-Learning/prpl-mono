"""Placement sampling utilities for dynamic3d environments."""

from typing import Any, Union

import numpy as np
from numpy.typing import NDArray

from prbench.envs.dynamic3d import utils
from prbench.envs.dynamic3d.objects import (
    MujocoFixture,
    MujocoObject,
    get_fixture_class,
    get_object_class,
)

# Default yaw range in degrees (full rotation)
DEFAULT_YAW_RANGE = (0.0, 360.0)


def sample_collision_free_positions(
    configs: dict[str, dict[str, dict[str, Any]]],
    np_random: np.random.Generator,
    entity_region_names: dict[str, str] | None = None,
    entity_pos_yaw_samplers: dict[str, Any] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Sample collision-free positions and yaws for multiple entities (fixtures or
    objects).

    Args:
        configs: Dictionary mapping entity types to dictionaries of entity configurations
                (entity_name -> entity_config). Can be fixture or object configurations.
        np_random: Random number generator
        # entity_ranges: Dictionary mapping entity names to sampling ranges as
        #               (x_min, y_min, x_max, y_max). If None, uses default range
        #               (-2.0, 0.5, 2.0, 2.5) for all entities.
        # entity_yaw_ranges: Dictionary mapping entity names to yaw rotation ranges as
        #                   (yaw_min, yaw_max) in degrees. If None, uses default range
        #                   (0.0, 360.0) for all entities.

    Returns:
        Dictionary mapping entity types to dictionaries of entity poses
        (entity_name -> {"position": position, "yaw": yaw})
    """
    if entity_region_names is None:
        entity_region_names = {}
    if entity_pos_yaw_samplers is None:
        entity_pos_yaw_samplers = {}

    entity_poses: dict[str, dict[str, dict[str, Any]]] = {}
    placed_bboxes: list[list[float]] = []

    for entity_type, entity_configs in configs.items():
        entity_poses[entity_type] = {}
        for entity_name, entity_config in entity_configs.items():

            if entity_name not in entity_pos_yaw_samplers:
                continue
            assert entity_name in entity_region_names, (
                f"Entity '{entity_name}' must have a region name specified in "
                f"entity_region_names if a pos_yaw_sampler is provided."
            )

            # Try to get the entity class (fixture or object)
            entity_class: Union[type[MujocoFixture], type[MujocoObject]]
            try:
                entity_class = get_fixture_class(entity_type)
            except ValueError:
                # If not a fixture, try as an object
                entity_class = get_object_class(entity_type)

            init_bbox = entity_class.get_bounding_box_from_config(
                np.array([0.0, 0.0, 0.0], dtype=np.float32), entity_config
            )
            # Sample a collision-free position and yaw for each entity
            position, yaw = sample_collision_free_position(
                list(init_bbox),
                placed_bboxes=placed_bboxes,
                np_random=np_random,
                region_name=entity_region_names[entity_name],
                pos_yaw_sampler=entity_pos_yaw_samplers[entity_name],
            )
            bbox = entity_class.get_bounding_box_from_config(position, entity_config)
            placed_bboxes.append(list(bbox))
            entity_poses[entity_type][entity_name] = {
                "position": position,
                "yaw": yaw,
            }
    return entity_poses


def sample_collision_free_position(
    bounding_box_at_origin: list[float],
    placed_bboxes: list[list[float]],
    np_random: np.random.Generator,
    region_name: str,
    pos_yaw_sampler: Any,
    max_attempts: int = 100,
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
    bbox_center_z = (bounding_box_at_origin[2] + bounding_box_at_origin[5]) / 2

    for _ in range(max_attempts):
        # Sample a candidate pose
        candidate_x, candidate_y, candidate_z, candidate_yaw = pos_yaw_sampler(
            region_name, np_random
        )

        candidate_pos = np.array(
            [candidate_x, candidate_y, candidate_z], dtype=np.float32
        )
        # print(f"Sampled candidate position trial {i_attempt}: {candidate_pos}")

        # Translate the bounding box to the candidate position
        translation = candidate_pos - np.array(
            [bbox_center_x, bbox_center_y, bbox_center_z]
        )
        translated_bbox = utils.translate_bounding_box(
            bounding_box_at_origin, translation
        )

        # Rotate the bounding box around its new center
        new_center = (candidate_pos[0], candidate_pos[1])
        candidate_bbox = utils.rotate_bounding_box_2d(
            translated_bbox, candidate_yaw, new_center
        )

        # Check if it collides with any existing fixture (using 3D overlap)
        collision = False
        for existing_bbox in placed_bboxes:
            if utils.bboxes_overlap(candidate_bbox, existing_bbox, margin=0.0):
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
    # pylint: disable=fixme
    fallback_pos = np.array(
        [
            0.0,
            0.0,
            0.0,  # TODO: consider ground thickness
        ]
    )
    # pylint: enable=fixme
    fallback_yaw_deg = np_random.uniform(DEFAULT_YAW_RANGE[0], DEFAULT_YAW_RANGE[1])
    fallback_yaw = np.radians(fallback_yaw_deg)
    return fallback_pos, fallback_yaw
