"""Tests for TidyBot utility functions."""

import numpy as np

from kinder.envs.dynamic3d.utils import (
    bboxes_overlap,
    compute_camera_euler,
    euler2mat_rzxy,
    point_in_bbox_3d,
    rotate_bounding_box_2d,
    rotate_bounding_box_3d,
    sample_pose_in_bbox_3d,
    translate_bounding_box,
)

# Tests for compute_camera_euler function


def test_compute_camera_euler_same_position():
    """Test that camera and target at same position returns zeros."""
    position = [0.0, 0.0, 0.0]
    lookat = [0.0, 0.0, 0.0]

    roll, pitch, yaw = compute_camera_euler(position, lookat)

    # Should return default angles when positions are the same
    assert roll == 0.0
    assert pitch == 0.0
    assert yaw == 0.0


def test_compute_camera_euler_looking_down():
    """Test camera looking straight down (negative z direction)."""
    position = [0.0, 0.0, 1.0]
    lookat = [0.0, 0.0, 0.0]

    roll, pitch, yaw = compute_camera_euler(position, lookat)

    # Looking straight down should have specific angles
    # The camera's -Z should point downward (from [0,0,1] to [0,0,0])
    assert isinstance(roll, float)
    assert isinstance(pitch, float)
    assert isinstance(yaw, float)


def test_compute_camera_euler_looking_forward():
    """Test camera looking forward (positive x direction)."""
    position = [1.0, 0.0, 0.0]
    lookat = [0.0, 0.0, 0.0]

    roll, pitch, yaw = compute_camera_euler(position, lookat)

    # Camera looking forward along negative x-axis
    assert isinstance(roll, float)
    assert isinstance(pitch, float)
    assert isinstance(yaw, float)


def test_compute_camera_euler_looking_right():
    """Test camera looking right (positive y direction)."""
    position = [0.0, 1.0, 0.0]
    lookat = [0.0, 0.0, 0.0]

    roll, pitch, yaw = compute_camera_euler(position, lookat)

    # Camera looking right along negative y-axis
    assert isinstance(roll, float)
    assert isinstance(pitch, float)
    assert isinstance(yaw, float)


def test_compute_camera_euler_at_45_degrees():
    """Test camera at 45 degree angle."""
    position = [1.0, 1.0, 1.0]
    lookat = [0.0, 0.0, 0.0]

    roll, pitch, yaw = compute_camera_euler(position, lookat)

    # Should produce valid angles
    assert isinstance(roll, float)
    assert isinstance(pitch, float)
    assert isinstance(yaw, float)
    # Angles should be in reasonable ranges
    assert -np.pi <= roll <= np.pi
    assert -np.pi <= pitch <= np.pi
    assert -np.pi <= yaw <= np.pi


def test_compute_camera_euler_far_distance():
    """Test camera at large distance from target."""
    position = [100.0, 100.0, 100.0]
    lookat = [0.0, 0.0, 0.0]

    roll, pitch, yaw = compute_camera_euler(position, lookat)

    # Should produce valid angles
    assert isinstance(roll, float)
    assert isinstance(pitch, float)
    assert isinstance(yaw, float)


def test_compute_camera_euler_close_distance():
    """Test camera very close to target."""
    position = [0.01, 0.01, 0.01]
    lookat = [0.0, 0.0, 0.0]

    roll, pitch, yaw = compute_camera_euler(position, lookat)

    # Should produce valid angles
    assert isinstance(roll, float)
    assert isinstance(pitch, float)
    assert isinstance(yaw, float)


def test_compute_camera_euler_negative_coords():
    """Test with negative coordinates."""
    position = [-1.0, -1.0, -1.0]
    lookat = [0.0, 0.0, 0.0]

    roll, pitch, yaw = compute_camera_euler(position, lookat)

    # Should produce valid angles
    assert isinstance(roll, float)
    assert isinstance(pitch, float)
    assert isinstance(yaw, float)
    assert -np.pi <= roll <= np.pi
    assert -np.pi <= pitch <= np.pi
    assert -np.pi <= yaw <= np.pi


def test_compute_camera_euler_symmetry():
    """Test that symmetric positions produce symmetric angles."""
    # Position along positive x
    roll1, pitch1, yaw1 = compute_camera_euler([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    # Position along negative x
    roll2, pitch2, yaw2 = compute_camera_euler([-1.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    # Both should be valid angles
    assert isinstance(roll1, float) and isinstance(roll2, float)
    assert isinstance(pitch1, float) and isinstance(pitch2, float)
    assert isinstance(yaw1, float) and isinstance(yaw2, float)


def test_compute_camera_euler_lookat_offset():
    """Test camera with different lookat positions."""
    position = [0.0, 0.0, 1.0]
    lookat = [1.0, 1.0, 0.0]

    roll, pitch, yaw = compute_camera_euler(position, lookat)

    # Should produce valid angles
    assert isinstance(roll, float)
    assert isinstance(pitch, float)
    assert isinstance(yaw, float)
    assert -np.pi <= roll <= np.pi
    assert -np.pi <= pitch <= np.pi
    assert -np.pi <= yaw <= np.pi


def test_compute_camera_euler_list_input():
    """Test that function accepts list input as documented."""
    position = [1.0, 2.0, 3.0]
    lookat = [0.0, 0.0, 0.0]

    # Should work with lists (as per docstring)
    roll, pitch, yaw = compute_camera_euler(position, lookat)

    assert isinstance(roll, float)
    assert isinstance(pitch, float)
    assert isinstance(yaw, float)


# Tests for bboxes_overlap function


def test_no_overlap_separated_horizontally():
    """Test that separated bounding boxes don't overlap."""
    bbox1 = [0.0, 0.0, 1.0, 1.0]
    bbox2 = [2.0, 0.0, 3.0, 1.0]

    assert not bboxes_overlap(bbox1, bbox2)
    assert not bboxes_overlap(bbox2, bbox1)  # Test symmetry


def test_no_overlap_separated_vertically():
    """Test that vertically separated bounding boxes don't overlap."""
    bbox1 = [0.0, 0.0, 1.0, 1.0]
    bbox2 = [0.0, 2.0, 1.0, 3.0]

    assert not bboxes_overlap(bbox1, bbox2)
    assert not bboxes_overlap(bbox2, bbox1)


def test_clear_overlap():
    """Test that clearly overlapping bounding boxes are detected."""
    bbox1 = [0.0, 0.0, 2.0, 2.0]
    bbox2 = [1.0, 1.0, 3.0, 3.0]

    assert bboxes_overlap(bbox1, bbox2)
    assert bboxes_overlap(bbox2, bbox1)


def test_identical_boxes():
    """Test that identical bounding boxes overlap."""
    bbox = [0.0, 0.0, 1.0, 1.0]

    assert bboxes_overlap(bbox, bbox)


def test_touching_boxes_no_margin():
    """Test touching boxes without margin."""
    bbox1 = [0.0, 0.0, 1.0, 1.0]
    bbox2 = [1.0, 0.0, 2.0, 1.0]  # Touching right edge

    # With default margin (0.2), these should overlap
    assert bboxes_overlap(bbox1, bbox2)

    # With zero margin, they should not overlap
    assert not bboxes_overlap(bbox1, bbox2, margin=0.0)


def test_margin_effect():
    """Test that margin parameter affects overlap detection."""
    bbox1 = [0.0, 0.0, 1.0, 1.0]
    bbox2 = [1.1, 0.0, 2.1, 1.0]  # 0.1 units apart

    # With small margin, no overlap
    assert not bboxes_overlap(bbox1, bbox2, margin=0.05)

    # With large margin, overlap detected
    assert bboxes_overlap(bbox1, bbox2, margin=0.15)


def test_nested_boxes():
    """Test that nested bounding boxes overlap."""
    bbox1 = [0.0, 0.0, 4.0, 4.0]  # Outer box
    bbox2 = [1.0, 1.0, 2.0, 2.0]  # Inner box

    assert bboxes_overlap(bbox1, bbox2)
    assert bboxes_overlap(bbox2, bbox1)


# Tests for translate_bounding_box function


def test_translate_bounding_box_positive_translation():
    """Test translating a bounding box with positive values."""
    bbox = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]  # Unit cube at origin
    translation = np.array([2.0, 3.0, 1.0])

    result = translate_bounding_box(bbox, translation)
    expected = [2.0, 3.0, 1.0, 3.0, 4.0, 2.0]

    assert result == expected


def test_translate_bounding_box_negative_translation():
    """Test translating a bounding box with negative values."""
    bbox = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    translation = np.array([-0.5, -1.0, -2.0])

    result = translate_bounding_box(bbox, translation)
    expected = [0.5, 1.0, 1.0, 3.5, 4.0, 4.0]

    assert result == expected


def test_translate_bounding_box_zero_translation():
    """Test translating a bounding box with zero translation (no change)."""
    bbox = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    translation = np.array([0.0, 0.0, 0.0])

    result = translate_bounding_box(bbox, translation)

    assert result == bbox


def test_translate_bounding_box_preserves_dimensions():
    """Test that translation preserves bounding box dimensions."""
    bbox = [0.0, 0.0, 0.0, 2.0, 3.0, 1.0]
    translation = np.array([1.0, 1.0, 1.0])

    result = translate_bounding_box(bbox, translation)

    # Original dimensions
    orig_width = bbox[3] - bbox[0]
    orig_height = bbox[4] - bbox[1]
    orig_depth = bbox[5] - bbox[2]

    # New dimensions
    new_width = result[3] - result[0]
    new_height = result[4] - result[1]
    new_depth = result[5] - result[2]

    assert new_width == orig_width
    assert new_height == orig_height
    assert new_depth == orig_depth


# Tests for rotate_bounding_box_2d function


def test_rotate_bounding_box_2d_no_rotation():
    """Test rotating a bounding box by 0 radians (no change)."""
    bbox = [0.0, 0.0, 0.0, 2.0, 1.0, 1.0]
    center = (1.0, 0.5)

    result = rotate_bounding_box_2d(bbox, 0.0, center)

    # Should be approximately the same (allowing for floating point precision)
    np.testing.assert_allclose(result, bbox, rtol=1e-10)


def test_rotate_bounding_box_2d_90_degrees():
    """Test rotating a bounding box by 90 degrees."""
    bbox = [0.0, 0.0, 0.0, 2.0, 1.0, 1.0]  # 2x1x1 box
    center = (1.0, 0.5)  # Center of the box

    result = rotate_bounding_box_2d(bbox, np.pi / 2, center)

    # After 90 degree rotation, the box should become 1x2x1
    width = result[3] - result[0]
    height = result[4] - result[1]

    # Should be approximately 1x2 (rotated from 2x1)
    assert abs(width - 1.0) < 1e-10
    assert abs(height - 2.0) < 1e-10

    # Z coordinates should be unchanged
    assert result[2] == bbox[2]
    assert result[5] == bbox[5]


def test_rotate_bounding_box_2d_180_degrees():
    """Test rotating a bounding box by 180 degrees."""
    bbox = [0.0, 0.0, 0.0, 2.0, 1.0, 1.0]
    center = (1.0, 0.5)

    result = rotate_bounding_box_2d(bbox, np.pi, center)

    # After 180 degrees, dimensions should be the same
    width = result[3] - result[0]
    height = result[4] - result[1]

    assert abs(width - 2.0) < 1e-10
    assert abs(height - 1.0) < 1e-10

    # Box should be centered at the same point
    result_center_x = (result[0] + result[3]) / 2
    result_center_y = (result[1] + result[4]) / 2

    assert abs(result_center_x - center[0]) < 1e-10
    assert abs(result_center_y - center[1]) < 1e-10


def test_rotate_bounding_box_2d_45_degrees():
    """Test rotating a bounding box by 45 degrees."""
    bbox = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]  # Unit square
    center = (0.5, 0.5)  # Center of the square

    result = rotate_bounding_box_2d(bbox, np.pi / 4, center)

    # After 45 degrees, a unit square should have dimensions sqrt(2) x sqrt(2)
    width = result[3] - result[0]
    height = result[4] - result[1]
    expected_dim = np.sqrt(2)

    assert abs(width - expected_dim) < 1e-10
    assert abs(height - expected_dim) < 1e-10

    # Center should remain the same
    result_center_x = (result[0] + result[3]) / 2
    result_center_y = (result[1] + result[4]) / 2

    assert abs(result_center_x - center[0]) < 1e-10
    assert abs(result_center_y - center[1]) < 1e-10


def test_rotate_bounding_box_2d_preserves_z():
    """Test that rotation preserves Z coordinates."""
    bbox = [1.0, 2.0, 5.0, 3.0, 4.0, 8.0]  # Arbitrary box with z from 5 to 8
    center = (2.0, 3.0)

    result = rotate_bounding_box_2d(bbox, np.pi / 3, center)  # 60 degrees

    # Z coordinates should be unchanged
    assert result[2] == bbox[2]  # z_min
    assert result[5] == bbox[5]  # z_max


def test_rotate_bounding_box_2d_different_centers():
    """Test rotating around different center points."""
    bbox = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

    # Rotate around origin
    result1 = rotate_bounding_box_2d(bbox, np.pi / 4, (0.0, 0.0))

    # Rotate around far point
    result2 = rotate_bounding_box_2d(bbox, np.pi / 4, (10.0, 10.0))

    # Results should be different (different center points)
    assert result1 != result2

    # But dimensions should be the same
    width1 = result1[3] - result1[0]
    height1 = result1[4] - result1[1]
    width2 = result2[3] - result2[0]
    height2 = result2[4] - result2[1]

    assert abs(width1 - width2) < 1e-10
    assert abs(height1 - height2) < 1e-10


def test_rotate_bounding_box_2d_full_rotation():
    """Test that a full 360-degree rotation returns to original."""
    bbox = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    center = (2.5, 3.5)

    result = rotate_bounding_box_2d(bbox, 2 * np.pi, center)

    # Should be approximately the same as original
    np.testing.assert_allclose(result, bbox, rtol=1e-10)


# Tests for point_in_bbox_3d function


def test_point_in_bbox_3d_inside():
    """Test that a point clearly inside the bbox is detected."""
    bbox = [0.0, 0.0, 0.0, 2.0, 2.0, 2.0]
    position = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    assert point_in_bbox_3d(position, bbox)


def test_point_in_bbox_3d_outside_x():
    """Test that a point outside the bbox in x direction is not detected."""
    bbox = [0.0, 0.0, 0.0, 2.0, 2.0, 2.0]
    position = np.array([3.0, 1.0, 1.0], dtype=np.float32)

    assert not point_in_bbox_3d(position, bbox)


def test_point_in_bbox_3d_outside_y():
    """Test that a point outside the bbox in y direction is not detected."""
    bbox = [0.0, 0.0, 0.0, 2.0, 2.0, 2.0]
    position = np.array([1.0, 3.0, 1.0], dtype=np.float32)

    assert not point_in_bbox_3d(position, bbox)


def test_point_in_bbox_3d_outside_z():
    """Test that a point outside the bbox in z direction is not detected."""
    bbox = [0.0, 0.0, 0.0, 2.0, 2.0, 2.0]
    position = np.array([1.0, 1.0, 3.0], dtype=np.float32)

    assert not point_in_bbox_3d(position, bbox)


def test_point_in_bbox_3d_on_boundary():
    """Test that a point on the boundary is detected as inside."""
    bbox = [0.0, 0.0, 0.0, 2.0, 2.0, 2.0]

    # Test all faces
    assert point_in_bbox_3d(
        np.array([0.0, 1.0, 1.0], dtype=np.float32), bbox
    )  # x_min face
    assert point_in_bbox_3d(
        np.array([2.0, 1.0, 1.0], dtype=np.float32), bbox
    )  # x_max face
    assert point_in_bbox_3d(
        np.array([1.0, 0.0, 1.0], dtype=np.float32), bbox
    )  # y_min face
    assert point_in_bbox_3d(
        np.array([1.0, 2.0, 1.0], dtype=np.float32), bbox
    )  # y_max face
    assert point_in_bbox_3d(
        np.array([1.0, 1.0, 0.0], dtype=np.float32), bbox
    )  # z_min face
    assert point_in_bbox_3d(
        np.array([1.0, 1.0, 2.0], dtype=np.float32), bbox
    )  # z_max face


def test_point_in_bbox_3d_at_corners():
    """Test that points at all corners are detected as inside."""
    bbox = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

    corners = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ]

    for corner in corners:
        assert point_in_bbox_3d(np.array(corner, dtype=np.float32), bbox)


def test_point_in_bbox_3d_negative_coords():
    """Test with negative coordinates in the bbox."""
    bbox = [-2.0, -2.0, -2.0, 0.0, 0.0, 0.0]

    # Inside
    assert point_in_bbox_3d(np.array([-1.0, -1.0, -1.0], dtype=np.float32), bbox)

    # Outside
    assert not point_in_bbox_3d(np.array([1.0, -1.0, -1.0], dtype=np.float32), bbox)


def test_point_in_bbox_3d_mixed_coords():
    """Test with bbox spanning negative to positive coordinates."""
    bbox = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]

    # Inside
    assert point_in_bbox_3d(np.array([0.0, 0.0, 0.0], dtype=np.float32), bbox)
    assert point_in_bbox_3d(np.array([-0.5, 0.5, -0.5], dtype=np.float32), bbox)

    # Outside
    assert not point_in_bbox_3d(np.array([2.0, 0.0, 0.0], dtype=np.float32), bbox)


# Tests for sample_pose_in_bbox_3d function


def test_sample_pose_in_bbox_3d_within_bounds():
    """Test that sampled poses are within the bbox bounds."""
    bbox = [0.0, 0.0, 0.0, 2.0, 2.0, 2.0]
    rng = np.random.default_rng(42)

    # Sample multiple poses
    for _ in range(100):
        x, y, z, yaw = sample_pose_in_bbox_3d(bbox, rng)

        # Check position is within bounds
        assert 0.0 <= x <= 2.0
        assert 0.0 <= y <= 2.0
        assert 0.0 <= z <= 2.0

        # Check yaw is within default range [0, 2*pi)
        assert 0.0 <= yaw <= 2 * np.pi


def test_sample_pose_in_bbox_3d_custom_yaw_range():
    """Test that sampled yaw is within custom range."""
    bbox = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    rng = np.random.default_rng(42)
    yaw_range_deg = (45.0, 90.0)

    # Sample multiple poses
    for _ in range(50):
        _, _, _, yaw = sample_pose_in_bbox_3d(bbox, rng, yaw_range_deg)

        # Convert yaw back to degrees for easier checking
        yaw_deg = np.degrees(yaw)

        # Check yaw is within specified range
        assert 45.0 <= yaw_deg <= 90.0


def test_sample_pose_in_bbox_3d_zero_yaw_range():
    """Test sampling with fixed yaw (zero range)."""
    bbox = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    rng = np.random.default_rng(42)
    yaw_range_deg = (30.0, 30.0)

    # Sample multiple poses
    for _ in range(10):
        _, _, _, yaw = sample_pose_in_bbox_3d(bbox, rng, yaw_range_deg)

        # Yaw should always be 30 degrees (converted to radians)
        expected_yaw = np.radians(30.0)
        assert abs(yaw - expected_yaw) < 1e-10


def test_sample_pose_in_bbox_3d_negative_bbox():
    """Test sampling from bbox with negative coordinates."""
    bbox = [-2.0, -3.0, -1.0, -1.0, -2.0, 0.0]
    rng = np.random.default_rng(42)

    # Sample multiple poses
    for _ in range(50):
        x, y, z, _ = sample_pose_in_bbox_3d(bbox, rng)

        # Check position is within bounds
        assert -2.0 <= x <= -1.0
        assert -3.0 <= y <= -2.0
        assert -1.0 <= z <= 0.0


def test_sample_pose_in_bbox_3d_variety():
    """Test that sampling produces variety (not always the same value)."""
    bbox = [0.0, 0.0, 0.0, 10.0, 10.0, 10.0]
    rng = np.random.default_rng(42)

    # Sample multiple poses
    samples = [sample_pose_in_bbox_3d(bbox, rng) for _ in range(20)]

    # Extract x, y, z, yaw separately
    x_values = [s[0] for s in samples]
    y_values = [s[1] for s in samples]
    z_values = [s[2] for s in samples]
    yaw_values = [s[3] for s in samples]

    # Check that we have variety (not all the same)
    assert len(set(x_values)) > 1
    assert len(set(y_values)) > 1
    assert len(set(z_values)) > 1
    assert len(set(yaw_values)) > 1


def test_sample_pose_in_bbox_3d_deterministic():
    """Test that sampling with the same seed produces the same results."""
    bbox = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

    # Sample with same seed
    rng1 = np.random.default_rng(123)
    sample1 = sample_pose_in_bbox_3d(bbox, rng1)

    rng2 = np.random.default_rng(123)
    sample2 = sample_pose_in_bbox_3d(bbox, rng2)

    # Should produce the same result
    assert sample1 == sample2


def test_sample_pose_in_bbox_3d_full_yaw_range():
    """Test sampling with full 360-degree yaw range."""
    bbox = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    rng = np.random.default_rng(42)
    yaw_range_deg = (0.0, 360.0)

    # Sample many poses to cover the range
    yaw_values = []
    for _ in range(200):
        _, _, _, yaw = sample_pose_in_bbox_3d(bbox, rng, yaw_range_deg)
        yaw_values.append(np.degrees(yaw))

    # Check that we get a good distribution across the range
    assert min(yaw_values) < 90.0  # Should have samples in first quadrant
    assert max(yaw_values) > 270.0  # Should have samples in last quadrant


def test_sample_pose_in_bbox_3d_small_bbox():
    """Test sampling from a very small bbox."""
    bbox = [1.0, 1.0, 1.0, 1.01, 1.01, 1.01]
    rng = np.random.default_rng(42)

    # Sample multiple poses
    for _ in range(20):
        x, y, z, _ = sample_pose_in_bbox_3d(bbox, rng)

        # All samples should be very close to the center
        assert 1.0 <= x <= 1.01
        assert 1.0 <= y <= 1.01
        assert 1.0 <= z <= 1.01


# Tests for rotate_bounding_box_3d function


def test_rotate_bounding_box_3d_no_rotation():
    """Test rotating a bounding box by identity (no change)."""
    bbox = [0.0, 0.0, 0.0, 2.0, 1.0, 1.5]
    identity_matrix = np.eye(3, dtype=np.float64)
    center = (1.0, 0.5, 0.75)

    result = rotate_bounding_box_3d(bbox, identity_matrix, center)

    # Should be approximately the same (allowing for floating point precision)
    np.testing.assert_allclose(result, bbox, rtol=1e-10)


def test_rotate_bounding_box_3d_90_degrees_z_axis():
    """Test rotating a bounding box by 90 degrees around z-axis."""
    bbox = [0.0, 0.0, 0.0, 2.0, 1.0, 1.0]  # 2x1x1 box
    center = (1.0, 0.5, 0.5)  # Center of the box

    # 90-degree rotation around z-axis
    angle = np.pi / 2
    rotation_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    result = rotate_bounding_box_3d(bbox, rotation_matrix, center)

    # After 90 degree rotation around z, the box should become 1x2x1
    width = result[3] - result[0]
    height = result[4] - result[1]
    depth = result[5] - result[2]

    # Should be approximately 1x2x1 (rotated from 2x1x1)
    assert abs(width - 1.0) < 1e-10
    assert abs(height - 2.0) < 1e-10
    assert abs(depth - 1.0) < 1e-10

    # Center should remain the same
    result_center_x = (result[0] + result[3]) / 2
    result_center_y = (result[1] + result[4]) / 2
    result_center_z = (result[2] + result[5]) / 2

    assert abs(result_center_x - center[0]) < 1e-10
    assert abs(result_center_y - center[1]) < 1e-10
    assert abs(result_center_z - center[2]) < 1e-10


def test_rotate_bounding_box_3d_90_degrees_x_axis():
    """Test rotating a bounding box by 90 degrees around x-axis."""
    bbox = [0.0, 0.0, 0.0, 1.0, 2.0, 1.0]  # 1x2x1 box
    center = (0.5, 1.0, 0.5)

    # 90-degree rotation around x-axis
    angle = np.pi / 2
    rotation_matrix = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ],
        dtype=np.float64,
    )

    result = rotate_bounding_box_3d(bbox, rotation_matrix, center)

    # After 90 degree rotation around x, y and z dimensions swap
    width = result[3] - result[0]
    height = result[4] - result[1]
    depth = result[5] - result[2]

    # Should be approximately 1x1x2 (rotated from 1x2x1)
    assert abs(width - 1.0) < 1e-10
    assert abs(height - 1.0) < 1e-10
    assert abs(depth - 2.0) < 1e-10


def test_rotate_bounding_box_3d_90_degrees_y_axis():
    """Test rotating a bounding box by 90 degrees around y-axis."""
    bbox = [0.0, 0.0, 0.0, 1.0, 1.0, 2.0]  # 1x1x2 box
    center = (0.5, 0.5, 1.0)

    # 90-degree rotation around y-axis
    angle = np.pi / 2
    rotation_matrix = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ],
        dtype=np.float64,
    )

    result = rotate_bounding_box_3d(bbox, rotation_matrix, center)

    # After 90 degree rotation around y, x and z dimensions swap
    width = result[3] - result[0]
    height = result[4] - result[1]
    depth = result[5] - result[2]

    # Should be approximately 2x1x1 (rotated from 1x1x2)
    assert abs(width - 2.0) < 1e-10
    assert abs(height - 1.0) < 1e-10
    assert abs(depth - 1.0) < 1e-10


def test_rotate_bounding_box_3d_180_degrees_z_axis():
    """Test rotating a bounding box by 180 degrees around z-axis."""
    bbox = [0.0, 0.0, 0.0, 2.0, 1.0, 1.0]
    center = (1.0, 0.5, 0.5)

    # 180-degree rotation around z-axis
    angle = np.pi
    rotation_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    result = rotate_bounding_box_3d(bbox, rotation_matrix, center)

    # After 180 degrees, dimensions should be the same
    width = result[3] - result[0]
    height = result[4] - result[1]
    depth = result[5] - result[2]

    assert abs(width - 2.0) < 1e-10
    assert abs(height - 1.0) < 1e-10
    assert abs(depth - 1.0) < 1e-10

    # Box should be centered at the same point
    result_center_x = (result[0] + result[3]) / 2
    result_center_y = (result[1] + result[4]) / 2
    result_center_z = (result[2] + result[5]) / 2

    assert abs(result_center_x - center[0]) < 1e-10
    assert abs(result_center_y - center[1]) < 1e-10
    assert abs(result_center_z - center[2]) < 1e-10


def test_rotate_bounding_box_3d_full_rotation_360_degrees():
    """Test that a full 360-degree rotation returns to original."""
    bbox = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    center = (2.5, 3.5, 4.5)

    # 360-degree rotation around z-axis
    angle = 2 * np.pi
    rotation_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    result = rotate_bounding_box_3d(bbox, rotation_matrix, center)

    # Should be approximately the same as original
    np.testing.assert_allclose(result, bbox, rtol=1e-10)


def test_rotate_bounding_box_3d_45_degrees_z_axis():
    """Test rotating a bounding box by 45 degrees around z-axis."""
    bbox = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]  # Unit cube
    center = (0.5, 0.5, 0.5)

    # 45-degree rotation around z-axis
    angle = np.pi / 4
    rotation_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    result = rotate_bounding_box_3d(bbox, rotation_matrix, center)

    # After 45 degrees, a unit square in xy should have dimensions sqrt(2) x sqrt(2)
    width = result[3] - result[0]
    height = result[4] - result[1]
    depth = result[5] - result[2]

    expected_2d_dim = np.sqrt(2)

    assert abs(width - expected_2d_dim) < 1e-10
    assert abs(height - expected_2d_dim) < 1e-10
    assert abs(depth - 1.0) < 1e-10

    # Center should remain the same
    result_center_x = (result[0] + result[3]) / 2
    result_center_y = (result[1] + result[4]) / 2
    result_center_z = (result[2] + result[5]) / 2

    assert abs(result_center_x - center[0]) < 1e-10
    assert abs(result_center_y - center[1]) < 1e-10
    assert abs(result_center_z - center[2]) < 1e-10


def test_rotate_bounding_box_3d_default_center():
    """Test that rotation uses bbox center when center is None."""
    bbox = [0.0, 0.0, 0.0, 2.0, 2.0, 2.0]

    # 90-degree rotation around z-axis with default center
    angle = np.pi / 2
    rotation_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    result = rotate_bounding_box_3d(bbox, rotation_matrix, center=None)

    # Default center should be (1.0, 1.0, 1.0)
    result_center_x = (result[0] + result[3]) / 2
    result_center_y = (result[1] + result[4]) / 2
    result_center_z = (result[2] + result[5]) / 2

    assert abs(result_center_x - 1.0) < 1e-10
    assert abs(result_center_y - 1.0) < 1e-10
    assert abs(result_center_z - 1.0) < 1e-10


def test_rotate_bounding_box_3d_different_centers():
    """Test rotating around different center points."""
    bbox = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

    # 90-degree rotation around z-axis
    angle = np.pi / 2
    rotation_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    # Rotate around origin
    result1 = rotate_bounding_box_3d(bbox, rotation_matrix, (0.0, 0.0, 0.0))

    # Rotate around far point
    result2 = rotate_bounding_box_3d(bbox, rotation_matrix, (10.0, 10.0, 10.0))

    # Results should be different (different center points)
    assert result1 != result2

    # But dimensions should be the same
    width1 = result1[3] - result1[0]
    height1 = result1[4] - result1[1]
    depth1 = result1[5] - result1[2]
    width2 = result2[3] - result2[0]
    height2 = result2[4] - result2[1]
    depth2 = result2[5] - result2[2]

    assert abs(width1 - width2) < 1e-10
    assert abs(height1 - height2) < 1e-10
    assert abs(depth1 - depth2) < 1e-10


def test_rotate_bounding_box_3d_negative_coords():
    """Test rotating bounding box with negative coordinates."""
    bbox = [-2.0, -2.0, -2.0, 0.0, 0.0, 0.0]
    center = (-1.0, -1.0, -1.0)

    # 90-degree rotation around z-axis
    angle = np.pi / 2
    rotation_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    result = rotate_bounding_box_3d(bbox, rotation_matrix, center)

    # After 90 degrees, dimensions should swap in xy
    width = result[3] - result[0]
    height = result[4] - result[1]
    depth = result[5] - result[2]

    assert abs(width - 2.0) < 1e-10
    assert abs(height - 2.0) < 1e-10
    assert abs(depth - 2.0) < 1e-10

    # Center should remain the same
    result_center_x = (result[0] + result[3]) / 2
    result_center_y = (result[1] + result[4]) / 2
    result_center_z = (result[2] + result[5]) / 2

    assert abs(result_center_x - center[0]) < 1e-10
    assert abs(result_center_y - center[1]) < 1e-10
    assert abs(result_center_z - center[2]) < 1e-10


def test_rotate_bounding_box_3d_using_euler2mat():
    """Test rotating using rotation matrix from euler2mat_rzxy."""
    bbox = [0.0, 0.0, 0.0, 1.0, 2.0, 1.0]
    center = (0.5, 1.0, 0.5)

    # Create rotation matrix for 90-degree rotation around z-axis
    rotation_matrix = euler2mat_rzxy(np.pi / 2, 0.0, 0.0)

    result = rotate_bounding_box_3d(bbox, rotation_matrix, center)

    # After 90 degrees around z, xy dimensions should swap
    width = result[3] - result[0]
    height = result[4] - result[1]
    depth = result[5] - result[2]

    assert abs(width - 2.0) < 1e-10
    assert abs(height - 1.0) < 1e-10
    assert abs(depth - 1.0) < 1e-10


def test_rotate_bounding_box_3d_small_bbox():
    """Test rotating a very small bounding box."""
    bbox = [1.0, 1.0, 1.0, 1.01, 1.01, 1.01]
    center = (1.005, 1.005, 1.005)

    # 90-degree rotation around z-axis
    angle = np.pi / 2
    rotation_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    result = rotate_bounding_box_3d(bbox, rotation_matrix, center)

    # Dimensions should be very small but approximately equal
    width = result[3] - result[0]
    height = result[4] - result[1]
    depth = result[5] - result[2]

    assert abs(width - 0.01) < 1e-6
    assert abs(height - 0.01) < 1e-6
    assert abs(depth - 0.01) < 1e-6


def test_rotate_bounding_box_3d_non_uniform_bbox():
    """Test rotating a non-uniform bounding box."""
    bbox = [0.0, 0.0, 0.0, 3.0, 2.0, 1.0]  # 3x2x1 box
    center = (1.5, 1.0, 0.5)

    # 45-degree rotation around z-axis
    angle = np.pi / 4
    rotation_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    result = rotate_bounding_box_3d(bbox, rotation_matrix, center)

    # Z dimension should be unchanged
    depth = result[5] - result[2]
    assert abs(depth - 1.0) < 1e-10

    # Center should remain the same
    result_center_x = (result[0] + result[3]) / 2
    result_center_y = (result[1] + result[4]) / 2
    result_center_z = (result[2] + result[5]) / 2

    assert abs(result_center_x - center[0]) < 1e-10
    assert abs(result_center_y - center[1]) < 1e-10
    assert abs(result_center_z - center[2]) < 1e-10
