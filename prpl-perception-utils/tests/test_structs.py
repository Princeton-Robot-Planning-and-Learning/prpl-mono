"""Tests for structs.py."""

import numpy as np
import pytest

from prpl_perception_utils.structs import (
    BoundingBox,
    DetectedObject2D,
    LanguageObjectDetectionID,
)


def test_language_object_detection_id_creation():
    """Test creating a LanguageObjectDetectionID."""
    obj_id = LanguageObjectDetectionID("person")
    assert obj_id.language_id == "person"


def test_language_object_detection_id_string():
    """Test string representation of LanguageObjectDetectionID."""
    obj_id = LanguageObjectDetectionID("car")
    assert str(obj_id) == "car"
    assert repr(obj_id) == "LanguageID(car)"


def test_language_object_detection_id_equality():
    """Test equality between LanguageObjectDetectionID instances."""
    obj_id1 = LanguageObjectDetectionID("dog")
    obj_id2 = LanguageObjectDetectionID("dog")
    obj_id3 = LanguageObjectDetectionID("cat")

    assert obj_id1 == obj_id2
    assert obj_id1 != obj_id3


def test_language_object_detection_id_hash():
    """Test hashing of LanguageObjectDetectionID."""
    obj_id = LanguageObjectDetectionID("bird")
    hash_value = hash(obj_id)
    assert isinstance(hash_value, int)


def test_bounding_box_creation():
    """Test creating a BoundingBox."""
    bbox = BoundingBox(10, 20, 30, 40)
    assert bbox.x1 == 10
    assert bbox.y1 == 20
    assert bbox.x2 == 30
    assert bbox.y2 == 40


def test_bounding_box_properties():
    """Test BoundingBox width and height properties."""
    bbox = BoundingBox(5, 10, 25, 30)
    assert bbox.width == 20
    assert bbox.height == 20


def test_bounding_box_validation():
    """Test BoundingBox validation on creation."""
    # Valid bbox
    BoundingBox(0, 0, 10, 10)

    # Invalid bbox - should raise assertion error
    with pytest.raises(AssertionError):
        BoundingBox(20, 20, 10, 10)  # x1 > x2

    with pytest.raises(AssertionError):
        BoundingBox(10, 30, 20, 20)  # y1 > y2


def test_detected_object_2d_creation():
    """Test creating a DetectedObject2D."""
    obj_id = LanguageObjectDetectionID("person")
    bbox = BoundingBox(0, 0, 10, 10)
    mask = np.zeros((10, 10), dtype=np.uint8)

    obj = DetectedObject2D(obj_id, bbox, mask, 0.8)
    assert obj.object_id == obj_id
    assert obj.bounding_box == bbox
    assert obj.confidence_score == 0.8


def test_detected_object_2d_mask_validation():
    """Test DetectedObject2D mask shape validation."""
    obj_id = LanguageObjectDetectionID("car")
    bbox = BoundingBox(0, 0, 5, 3)

    # Correct mask shape
    correct_mask = np.zeros((3, 5), dtype=np.uint8)
    DetectedObject2D(obj_id, bbox, correct_mask)

    # Wrong mask shape - should raise assertion error
    wrong_mask = np.zeros((5, 3), dtype=np.uint8)
    with pytest.raises(AssertionError):
        DetectedObject2D(obj_id, bbox, wrong_mask)


def test_detected_object_2d_confidence_validation():
    """Test DetectedObject2D confidence score validation."""
    obj_id = LanguageObjectDetectionID("dog")
    bbox = BoundingBox(0, 0, 10, 10)
    mask = np.zeros((10, 10), dtype=np.uint8)

    # Valid confidence scores
    DetectedObject2D(obj_id, bbox, mask, 0.0)
    DetectedObject2D(obj_id, bbox, mask, 0.5)
    DetectedObject2D(obj_id, bbox, mask, 1.0)

    # Invalid confidence scores - should raise assertion error
    with pytest.raises(AssertionError):
        DetectedObject2D(obj_id, bbox, mask, -0.1)

    with pytest.raises(AssertionError):
        DetectedObject2D(obj_id, bbox, mask, 1.1)
