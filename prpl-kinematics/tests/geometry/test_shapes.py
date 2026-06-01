"""Unit tests for geometry shapes."""

from prpl_kinematics.geometry.shapes import BoxShape


def test_shape_color_defaults_to_none():
    """Shapes are uncolored by default, so backends apply their own defaults."""
    assert BoxShape(size=(1.0, 1.0, 1.0)).color is None
