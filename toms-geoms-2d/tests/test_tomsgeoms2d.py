"""Tests for tomsgeoms2d."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from tomsgeoms2d.structs import Circle, LineSegment, Rectangle, Triangle
from tomsgeoms2d.utils import geom2ds_intersect


def test_line_segment():
    """Tests for LineSegment()."""
    _, ax = plt.subplots(1, 1)
    ax.set_xlim((-5, 5))
    ax.set_ylim((-8, 8))

    seg1 = LineSegment(x1=0, y1=1, x2=3, y2=7)
    assert seg1.x1 == 0
    assert seg1.y1 == 1
    assert seg1.x2 == 3
    assert seg1.y2 == 7
    seg1.plot(ax, color="red", linewidth=2)
    assert seg1.contains_point(2, 5)
    assert not seg1.contains_point(2.1, 5)
    assert not seg1.contains_point(2, 4.9)

    seg2 = LineSegment(x1=2, y1=-5, x2=1, y2=6)
    seg2.plot(ax, color="blue", linewidth=2)

    seg3 = LineSegment(x1=-2, y1=-3, x2=-4, y2=2)
    seg3.plot(ax, color="green", linewidth=2)

    assert geom2ds_intersect(seg1, seg2)
    assert not geom2ds_intersect(seg1, seg3)
    assert not geom2ds_intersect(seg2, seg3)

    rng = np.random.default_rng(0)
    for _ in range(10):
        p1 = seg1.sample_random_point(rng)
        assert seg1.contains_point(p1[0], p1[1])
        plt.plot(p1[0], p1[1], "bo")

    # Uncomment for debugging.
    # plt.savefig("/tmp/line_segment_unit_test.png")

    # Legacy tests.
    seg1 = LineSegment(2, 5, 7, 6)
    seg2 = LineSegment(2.5, 7.1, 7.4, 5.3)
    assert geom2ds_intersect(seg1, seg2)

    seg1 = LineSegment(1, 3, 5, 3)
    seg2 = LineSegment(3, 7, 3, 2)
    assert geom2ds_intersect(seg1, seg2)

    seg1 = LineSegment(2, 5, 7, 6)
    seg2 = LineSegment(2, 6, 7, 7)
    assert not geom2ds_intersect(seg1, seg2)

    seg1 = LineSegment(1, 1, 3, 3)
    seg2 = LineSegment(2, 2, 4, 4)
    assert not geom2ds_intersect(seg1, seg2)

    seg1 = LineSegment(1, 1, 3, 3)
    seg2 = LineSegment(1, 1, 6.7, 7.4)
    assert not geom2ds_intersect(seg1, seg2)


def test_circle():
    """Tests for Circle()."""
    _, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_xlim((-11, 5))
    ax.set_ylim((-6, 10))

    circ1 = Circle(x=0, y=1, radius=3)
    assert circ1.x == 0
    assert circ1.y == 1
    assert circ1.radius == 3
    circ1.plot(ax, color="red", alpha=0.5)

    assert circ1.contains_point(0, 1)
    assert circ1.contains_point(0.5, 1)
    assert circ1.contains_point(0, 0.5)
    assert circ1.contains_point(0.25, 1.25)
    assert not circ1.contains_point(0, 4.1)
    assert not circ1.contains_point(3.1, 0)
    assert not circ1.contains_point(0, -2.1)
    assert not circ1.contains_point(-3.1, 0)

    circ2 = Circle(x=-3, y=2, radius=6)
    circ2.plot(ax, color="blue", alpha=0.5)

    circ3 = Circle(x=-6, y=1, radius=1)
    circ3.plot(ax, color="green", alpha=0.5)

    assert geom2ds_intersect(circ1, circ2)
    assert not geom2ds_intersect(circ1, circ3)
    assert geom2ds_intersect(circ2, circ3)

    rng = np.random.default_rng(0)
    for _ in range(10):
        p3 = circ3.sample_random_point(rng)
        assert circ3.contains_point(p3[0], p3[1])
        plt.plot(p3[0], p3[1], "bo")

    # Uncomment for debugging.
    # plt.savefig("/tmp/circle_unit_test.png")


def test_triangle():
    """Tests for Triangle()."""
    _, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_xlim((-10.0, 10.0))
    ax.set_ylim((-10.0, 10.0))

    tri1 = Triangle(5.0, 5.0, 7.5, 7.5, 5.0, 7.5)
    assert tri1.contains_point(5.5, 6)
    assert tri1.contains_point(5.9999, 6)
    assert tri1.contains_point(5.8333, 6.6667)
    assert tri1.contains_point(7.3, 7.4)
    assert not tri1.contains_point(6, 6)
    assert not tri1.contains_point(5.1, 5.1)
    assert not tri1.contains_point(5.2, 5.1)
    assert not tri1.contains_point(5.1, 7.6)
    assert not tri1.contains_point(4.9, 7.3)
    assert not tri1.contains_point(5.0, 7.5)
    assert not tri1.contains_point(7.6, 7.6)
    tri1.plot(ax, color="red", alpha=0.5)

    tri2 = Triangle(-3.0, -4.0, -6.2, -5.6, -9.0, -1.7)
    tri2.plot(ax, color="blue", alpha=0.5)

    # Almost degenerate triangle.
    tri3 = Triangle(0.0, 0.0, 1.0, 1.0, -1.0, -1.001)
    assert tri3.contains_point(0.0, -0.001 / 3.0)
    tri3.plot(ax, color="green", alpha=0.5)

    # Degenerate triangle (a line).
    with pytest.raises(ValueError) as e:
        Triangle(0.0, 0.0, 1.0, 1.0, -1.0, -1.0)
    assert "Degenerate triangle" in str(e)

    rng = np.random.default_rng(0)
    for _ in range(10):
        p1 = tri1.sample_random_point(rng)
        assert tri1.contains_point(p1[0], p1[1])
        plt.plot(p1[0], p1[1], "bo")

    # Uncomment for debugging.
    # plt.savefig("/tmp/triangle_unit_test.png")


def test_rectangle():
    """Tests for Rectangle()."""
    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_xlim((-5, 5))
    ax.set_ylim((-5, 5))

    rect1 = Rectangle(x=-2, y=-1, width=4, height=3, theta=0)
    assert rect1.x == -2
    assert rect1.y == -1
    assert rect1.width == 4
    assert rect1.height == 3
    assert rect1.theta == 0
    rect1.plot(ax, color="red", alpha=0.5)

    assert np.allclose(rect1.center, (0, 0.5))

    circ1 = rect1.circumscribed_circle
    assert np.allclose((circ1.x, circ1.y), (0, 0.5))
    assert np.allclose(circ1.radius, 2.5)
    circ1.plot(ax, facecolor="none", edgecolor="black", linewidth=1, linestyle="dashed")

    expected_vertices = np.array([(-2, -1), (-2, 2), (2, -1), (2, 2)])
    assert np.allclose(sorted(rect1.vertices), expected_vertices)
    for x, y in rect1.vertices:
        v = Circle(x, y, radius=0.1)
        v.plot(ax, facecolor="none", edgecolor="black", linewidth=1, linestyle="dashed")

    for seg in rect1.line_segments:
        seg.plot(ax, color="black", linewidth=1, linestyle="dashed")

    assert not rect1.contains_point(-2.1, 0)
    assert rect1.contains_point(-1.9, 0)
    assert not rect1.contains_point(0, 2.1)
    assert rect1.contains_point(0, 1.9)
    assert not rect1.contains_point(2.1, 0)
    assert rect1.contains_point(1.9, 0)
    assert not rect1.contains_point(0, -1.1)
    assert rect1.contains_point(0, -0.9)
    assert rect1.contains_point(0, 0.5)
    assert not rect1.contains_point(100, 100)

    rect2 = Rectangle(x=1, y=-2, width=2, height=2, theta=0.5)
    rect2.plot(ax, color="blue", alpha=0.5)

    rect3 = Rectangle(x=-1.5, y=1, width=1, height=1, theta=-0.5)
    rect3.plot(ax, color="green", alpha=0.5)

    assert geom2ds_intersect(rect1, rect2)
    assert geom2ds_intersect(rect1, rect3)
    assert geom2ds_intersect(rect3, rect1)
    assert not geom2ds_intersect(rect2, rect3)

    rect4 = Rectangle(x=0.8, y=1e-5, height=0.1, width=0.07, theta=0)
    assert not rect4.contains_point(0.2, 0.05)

    rect5 = Rectangle(x=-4, y=-2, height=0.25, width=2, theta=-np.pi / 4)
    rect5.plot(ax, facecolor="yellow", edgecolor="gray")
    origin = Circle(x=-3.5, y=-2.3, radius=0.05)
    origin.plot(ax, color="black")
    rect6 = rect5.rotate_about_point(origin.x, origin.y, rot=np.pi / 4)
    rect6.plot(ax, facecolor="none", edgecolor="black", linestyle="dashed")

    rect7 = Rectangle.from_center(
        center_x=1, center_y=2, width=2, height=4, rotation_about_center=0
    )
    rect7.plot(ax, facecolor="grey")
    assert rect7.center == (1, 2)

    rng = np.random.default_rng(0)
    for _ in range(100):
        p5 = rect5.sample_random_point(rng)
        assert rect5.contains_point(p5[0], p5[1])
        plt.plot(p5[0], p5[1], "bo")

    # Uncomment for debugging.
    # plt.savefig("/tmp/rectangle_unit_test.png")

    # Tests scale_about_center().
    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_xlim((-5, 5))
    ax.set_ylim((-5, 5))

    rect8 = Rectangle(x=-2, y=-1, width=4, height=3, theta=0)
    rect8.plot(ax, color="red", alpha=0.5)
    rect9 = rect8.scale_about_center(width_scale=0.75, height_scale=0.5)
    assert set(rect9.vertices) == {
        (-1.5, -0.25),
        (-1.5, 1.25),
        (1.5, 1.25),
        (1.5, -0.25),
    }
    rect9.plot(ax, color="blue", alpha=0.5)

    # Uncomment for debugging.
    # plt.savefig("/tmp/rectangle_unit_test2.png")


def test_line_segment_circle_intersection():
    """Tests for line_segment_intersects_circle()."""
    seg1 = LineSegment(-3, 0, 0, 0)
    circ1 = Circle(0, 0, 1)
    assert geom2ds_intersect(seg1, circ1)
    assert geom2ds_intersect(circ1, seg1)

    seg2 = LineSegment(-3, 3, 4, 3)
    assert not geom2ds_intersect(seg2, circ1)
    assert not geom2ds_intersect(circ1, seg2)

    seg3 = LineSegment(0, -2, 1, -2.5)
    assert not geom2ds_intersect(seg3, circ1)
    assert not geom2ds_intersect(circ1, seg3)

    seg4 = LineSegment(0, -3, 0, -4)
    assert not geom2ds_intersect(seg4, circ1)
    assert not geom2ds_intersect(circ1, seg4)
    assert not geom2ds_intersect(seg2, circ1)

    # Uncomment for debugging.
    # plt.savefig("/tmp/line_segment_circle_unit_test.png")


def test_line_segment_rectangle_intersection():
    """Tests for line_segment_intersects_rectangle()."""
    seg1 = LineSegment(-3, 0, 0, 0)
    rect1 = Rectangle(-1, -1, 2, 2, 0)
    assert geom2ds_intersect(seg1, rect1)
    assert geom2ds_intersect(rect1, seg1)

    seg2 = LineSegment(-3, 3, 4, 3)
    assert not geom2ds_intersect(seg2, rect1)
    assert not geom2ds_intersect(rect1, seg2)

    seg3 = LineSegment(0, -2, 1, -2.5)
    assert not geom2ds_intersect(seg3, rect1)
    assert not geom2ds_intersect(rect1, seg3)

    seg4 = LineSegment(0, -3, 0, -4)
    assert not geom2ds_intersect(seg4, rect1)
    assert not geom2ds_intersect(rect1, seg4)


def test_rectangle_circle_intersection():
    """Tests for rectangle_intersects_circle()."""
    rect1 = Rectangle(x=0, y=0, width=4, height=3, theta=0)
    circ1 = Circle(x=0, y=0, radius=1)
    assert geom2ds_intersect(rect1, circ1)
    assert geom2ds_intersect(circ1, rect1)

    circ2 = Circle(x=1, y=1, radius=0.5)
    assert geom2ds_intersect(rect1, circ2)
    assert geom2ds_intersect(circ2, rect1)

    rect2 = Rectangle(x=1, y=1, width=1, height=1, theta=0)
    assert not geom2ds_intersect(rect2, circ1)
    assert not geom2ds_intersect(circ1, rect2)

    circ3 = Circle(x=0, y=0, radius=100)
    assert geom2ds_intersect(rect1, circ3)
    assert geom2ds_intersect(circ3, rect1)
    assert geom2ds_intersect(rect2, circ3)
    assert geom2ds_intersect(circ3, rect2)


def test_geom2ds_intersect():
    """Tests for geom2ds_intersect()."""
    with pytest.raises(NotImplementedError):
        geom2ds_intersect(None, None)
