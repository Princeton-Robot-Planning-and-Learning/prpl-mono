"""Tests for placement sampling utilities for dynamic3d environments."""

import numpy as np
import pytest

from prbench.envs.dynamic3d.placement_samplers import (
    sample_collision_free_position,
    sample_pose_in_region,
)
from prbench.envs.dynamic3d.objects import Table
from prbench.envs.dynamic3d.utils import bboxes_overlap


# Tests for sample_collision_free_position function


def test_no_existing_tables():
    """Test sampling when no tables are placed yet."""
    np_random = np.random.default_rng(42)
    table_config = {"shape": "rectangle", "length": 0.5, "width": 0.5, "height": 0.8}
    placed_bboxes = []

    # Get initial bounding box at origin
    initial_bbox = Table.get_bounding_box_from_config(
        np.array([0.0, 0.0, 0.0], dtype=np.float32), table_config
    )

    pos, yaw = sample_collision_free_position(initial_bbox, placed_bboxes, np_random)

    # Should return a valid position and yaw
    assert isinstance(pos, np.ndarray)
    assert pos.shape == (3,)
    assert pos[2] == 0.0  # z should always be 0
    assert isinstance(yaw, float)

    # Position should be within default ranges
    assert -2.0 <= pos[0] <= 2.0
    assert 0.5 <= pos[1] <= 2.5


def test_with_existing_tables():
    """Test sampling with existing tables."""
    np_random = np.random.default_rng(42)
    table_config = {"shape": "rectangle", "length": 0.5, "width": 0.5, "height": 0.8}
    # Place a table in the middle of the sampling area
    placed_bboxes = [[0.0, 1.0, 0.0, 1.0, 2.0, 0.8]]

    # Get initial bounding box at origin
    initial_bbox = Table.get_bounding_box_from_config(
        np.array([0.0, 0.0, 0.0], dtype=np.float32), table_config
    )

    pos, _ = sample_collision_free_position(initial_bbox, placed_bboxes, np_random)

    # Check that the sampled position doesn't create an overlapping bbox
    new_bbox = Table.get_bounding_box_from_config(pos, table_config)
    for existing_bbox in placed_bboxes:
        assert not bboxes_overlap(new_bbox[:4], existing_bbox[:4])


def test_custom_ranges():
    """Test sampling with custom x and y ranges."""
    np_random = np.random.default_rng(42)
    table_config = {"shape": "rectangle", "length": 0.5, "width": 0.5, "height": 0.8}
    placed_bboxes = []
    x_range = (5.0, 6.0)
    y_range = (10.0, 11.0)

    # Get initial bounding box at origin
    initial_bbox = Table.get_bounding_box_from_config(
        np.array([0.0, 0.0, 0.0], dtype=np.float32), table_config
    )

    pos, _ = sample_collision_free_position(
        initial_bbox, placed_bboxes, np_random, x_range=x_range, y_range=y_range
    )

    assert x_range[0] <= pos[0] <= x_range[1]
    assert y_range[0] <= pos[1] <= y_range[1]
    assert pos[2] == 0.0


def test_deterministic_with_seed():
    """Test that sampling is deterministic with same seed."""
    table_config = {"shape": "rectangle", "length": 0.5, "width": 0.5, "height": 0.8}
    placed_bboxes = []

    # Get initial bounding box at origin
    initial_bbox = Table.get_bounding_box_from_config(
        np.array([0.0, 0.0, 0.0], dtype=np.float32), table_config
    )

    # Sample with first generator
    rng1 = np.random.default_rng(123)
    pos1, yaw1 = sample_collision_free_position(initial_bbox, placed_bboxes, rng1)

    # Sample with second generator with same seed
    rng2 = np.random.default_rng(123)
    pos2, yaw2 = sample_collision_free_position(initial_bbox, placed_bboxes, rng2)

    np.testing.assert_array_equal(pos1, pos2)
    assert yaw1 == yaw2


def test_crowded_scenario_fallback():
    """Test fallback behavior when space is very crowded."""
    np_random = np.random.default_rng(42)
    # Create a scenario where it's very hard to find collision-free space
    large_table_config = {
        "shape": "rectangle", "length": 3.0, "width": 3.0, "height": 0.8
    }
    placed_bboxes = [
        [-2.0, 0.5, 0.0, 0.0, 2.5, 0.8],  # Left side
        [0.0, 0.5, 0.0, 2.0, 2.5, 0.8],   # Right side
    ]

    # Get initial bounding box at origin
    initial_bbox = Table.get_bounding_box_from_config(
        np.array([0.0, 0.0, 0.0], dtype=np.float32), large_table_config
    )

    # Should still return a position (though it might overlap)
    pos, yaw = sample_collision_free_position(
        initial_bbox, placed_bboxes, np_random, max_attempts=5
    )

    assert isinstance(pos, np.ndarray)
    assert pos.shape == (3,)
    assert pos[2] == 0.0
    assert isinstance(yaw, float)


def test_circular_table():
    """Test sampling with circular table configuration."""
    np_random = np.random.default_rng(42)
    circle_config = {"shape": "circle", "diameter": 0.8, "height": 0.8}
    placed_bboxes = []

    # Get initial bounding box at origin
    initial_bbox = Table.get_bounding_box_from_config(
        np.array([0.0, 0.0, 0.0], dtype=np.float32), circle_config
    )

    pos, yaw = sample_collision_free_position(initial_bbox, placed_bboxes, np_random)

    assert isinstance(pos, np.ndarray)
    assert pos.shape == (3,)
    assert pos[2] == 0.0
    assert isinstance(yaw, float)


# Tests for sample_pose_in_region function


def test_single_region():
    """Test sampling from a single region."""
    np_random = np.random.default_rng(42)
    regions = [[1.0, 2.0, 3.0, 4.0]]  # [x_start, y_start, x_end, y_end]

    x, y, z = sample_pose_in_region(regions, np_random)

    assert 1.0 <= x <= 3.0
    assert 2.0 <= y <= 4.0
    assert z == 0.02  # Default z coordinate


def test_multiple_regions():
    """Test sampling from multiple regions."""
    np_random = np.random.default_rng(42)
    regions = [
        [0.0, 0.0, 1.0, 1.0],  # Region 1
        [5.0, 5.0, 6.0, 6.0],  # Region 2
        [10.0, 10.0, 11.0, 11.0],  # Region 3
    ]

    # Sample many times to check all regions can be selected
    sampled_regions = set()
    for _ in range(100):
        x, y, z = sample_pose_in_region(regions, np_random)

        # Determine which region this sample came from
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            sampled_regions.add(0)
        elif 5.0 <= x <= 6.0 and 5.0 <= y <= 6.0:
            sampled_regions.add(1)
        elif 10.0 <= x <= 11.0 and 10.0 <= y <= 11.0:
            sampled_regions.add(2)

        assert z == 0.02

    # Should have sampled from all regions at least once
    assert len(sampled_regions) == 3


def test_custom_z_coordinate():
    """Test sampling with custom z coordinate."""
    np_random = np.random.default_rng(42)
    regions = [[0.0, 0.0, 1.0, 1.0]]
    custom_z = 0.5

    x, y, z = sample_pose_in_region(regions, np_random, z_coordinate=custom_z)

    assert 0.0 <= x <= 1.0
    assert 0.0 <= y <= 1.0
    assert z == custom_z


def test_deterministic_with_seed_pose():
    """Test that sampling is deterministic with same seed."""
    regions = [[0.0, 0.0, 1.0, 1.0]]

    # Sample with first generator
    rng1 = np.random.default_rng(123)
    pose1 = sample_pose_in_region(regions, rng1)

    # Sample with second generator with same seed
    rng2 = np.random.default_rng(123)
    pose2 = sample_pose_in_region(regions, rng2)

    assert pose1 == pose2


def test_empty_regions_list():
    """Test that empty regions list raises ValueError."""
    np_random = np.random.default_rng(42)
    regions = []

    with pytest.raises(ValueError, match="Regions list cannot be empty"):
        sample_pose_in_region(regions, np_random)


def test_invalid_region_format():
    """Test that invalid region format raises ValueError."""
    np_random = np.random.default_rng(42)
    # Region with wrong number of elements
    regions = [[0.0, 0.0, 1.0]]  # Missing y_end

    with pytest.raises(ValueError, match="Each region must have exactly 4 values"):
        sample_pose_in_region(regions, np_random)


def test_invalid_x_bounds():
    """Test that invalid x bounds raise ValueError."""
    np_random = np.random.default_rng(42)
    regions = [[1.0, 0.0, 0.0, 1.0]]  # x_start > x_end

    with pytest.raises(ValueError, match="x_start .* must be less than x_end"):
        sample_pose_in_region(regions, np_random)


def test_invalid_y_bounds():
    """Test that invalid y bounds raise ValueError."""
    np_random = np.random.default_rng(42)
    regions = [[0.0, 1.0, 1.0, 0.0]]  # y_start > y_end

    with pytest.raises(ValueError, match="y_start .* must be less than y_end"):
        sample_pose_in_region(regions, np_random)


def test_equal_bounds():
    """Test that equal bounds raise ValueError."""
    np_random = np.random.default_rng(42)

    # x_start == x_end
    regions = [[1.0, 0.0, 1.0, 1.0]]
    with pytest.raises(ValueError, match="x_start .* must be less than x_end"):
        sample_pose_in_region(regions, np_random)

    # y_start == y_end
    regions = [[0.0, 1.0, 1.0, 1.0]]
    with pytest.raises(ValueError, match="y_start .* must be less than y_end"):
        sample_pose_in_region(regions, np_random)


def test_point_region():
    """Test sampling from a very small region."""
    np_random = np.random.default_rng(42)
    regions = [[0.0, 0.0, 0.001, 0.001]]  # Very small region

    x, y, z = sample_pose_in_region(regions, np_random)

    assert 0.0 <= x <= 0.001
    assert 0.0 <= y <= 0.001
    assert z == 0.02


def test_negative_coordinates():
    """Test sampling from regions with negative coordinates."""
    np_random = np.random.default_rng(42)
    regions = [[-2.0, -3.0, -1.0, -1.0]]

    x, y, z = sample_pose_in_region(regions, np_random)

    assert -2.0 <= x <= -1.0
    assert -3.0 <= y <= -1.0
    assert z == 0.02
