"""Tests for camera.py."""

from pybullet_helpers.camera import capture_image


def test_capture_image(physics_client_id):
    """Tests for capture_image()."""
    img = capture_image(physics_client_id, image_width=32, image_height=32)
    assert img.shape == (32, 32, 3)
