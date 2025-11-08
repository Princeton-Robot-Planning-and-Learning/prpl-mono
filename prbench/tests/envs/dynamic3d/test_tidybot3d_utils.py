"""Tests for TidyBot utility functions."""

import numpy as np

from prbench.envs.dynamic3d.utils import (
    bboxes_overlap,
    rotate_bounding_box_2d,
    translate_bounding_box,
)

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
