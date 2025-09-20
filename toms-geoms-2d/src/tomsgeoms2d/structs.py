"""Data structures."""

from __future__ import annotations

import abc
import functools
from dataclasses import dataclass
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from numpy.typing import NDArray


class Geom2D(abc.ABC):
    """A 2D shape that contains some points."""

    @abc.abstractmethod
    def plot(self, ax: plt.Axes, **kwargs: Any) -> None:
        """Plot the shape on a given pyplot axis."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def contains_point(self, x: float, y: float) -> bool:
        """Checks if a point is contained in the shape."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def sample_random_point(self, rng: np.random.Generator) -> Tuple[float, float]:
        """Samples a random point inside the 2D shape."""
        raise NotImplementedError("Override me!")

    def intersects(self, other: Geom2D) -> bool:
        """Checks if this shape intersects with another one."""
        return geom2ds_intersect(self, other)


@dataclass(frozen=True)
class LineSegment(Geom2D):
    """A helper class for visualizing and collision checking line segments."""

    x1: float
    y1: float
    x2: float
    y2: float

    def plot(self, ax: plt.Axes, **kwargs: Any) -> None:
        ax.plot([self.x1, self.x2], [self.y1, self.y2], **kwargs)

    def contains_point(self, x: float, y: float) -> bool:
        # https://stackoverflow.com/questions/328107
        a = (self.x1, self.y1)
        b = (self.x2, self.y2)
        c = (x, y)
        # Need to use an epsilon for numerical stability. But we are checking
        # if the distance from a to b is (approximately) equal to the distance
        # from a to c and the distance from c to b.
        eps = 1e-6

        def _dist(p: Tuple[float, float], q: Tuple[float, float]) -> float:
            return np.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)

        return -eps < _dist(a, c) + _dist(c, b) - _dist(a, b) < eps

    def sample_random_point(self, rng: np.random.Generator) -> Tuple[float, float]:
        line_slope = (self.y2 - self.y1) / (self.x2 - self.x1)
        y_intercept = self.y2 - (line_slope * self.x2)
        random_x_point = rng.uniform(self.x1, self.x2)
        random_y_point_on_line = line_slope * random_x_point + y_intercept
        assert self.contains_point(random_x_point, random_y_point_on_line)
        return (random_x_point, random_y_point_on_line)


@dataclass(frozen=True)
class Circle(Geom2D):
    """A helper class for visualizing and collision checking circles."""

    x: float
    y: float
    radius: float

    def plot(self, ax: plt.Axes, **kwargs: Any) -> None:
        patch = patches.Circle((self.x, self.y), self.radius, **kwargs)
        ax.add_patch(patch)

    def contains_point(self, x: float, y: float) -> bool:
        return (x - self.x) ** 2 + (y - self.y) ** 2 <= self.radius**2

    def contains_circle(self, other_circle: Circle) -> bool:
        """Check whether this circle wholly contains another one."""
        dist_between_centers = np.sqrt(
            (other_circle.x - self.x) ** 2 + (other_circle.y - self.y) ** 2
        )
        return (dist_between_centers + other_circle.radius) <= self.radius

    def sample_random_point(self, rng: np.random.Generator) -> Tuple[float, float]:
        rand_mag = rng.uniform(0, self.radius)
        rand_theta = rng.uniform(0, 2 * np.pi)
        x_point = self.x + rand_mag * np.cos(rand_theta)
        y_point = self.y + rand_mag * np.sin(rand_theta)
        assert self.contains_point(x_point, y_point)
        return (x_point, y_point)


@dataclass(frozen=True)
class Triangle(Geom2D):
    """A helper class for visualizing and collision checking triangles."""

    x1: float
    y1: float
    x2: float
    y2: float
    x3: float
    y3: float

    def plot(self, ax: plt.Axes, **kwargs: Any) -> None:
        patch = patches.Polygon(
            [[self.x1, self.y1], [self.x2, self.y2], [self.x3, self.y3]], **kwargs
        )
        ax.add_patch(patch)

    def __post_init__(self) -> None:
        dist1 = np.sqrt((self.x1 - self.x2) ** 2 + (self.y1 - self.y2) ** 2)
        dist2 = np.sqrt((self.x2 - self.x3) ** 2 + (self.y2 - self.y3) ** 2)
        dist3 = np.sqrt((self.x3 - self.x1) ** 2 + (self.y3 - self.y1) ** 2)
        dists = sorted([dist1, dist2, dist3])
        assert dists[0] + dists[1] >= dists[2]
        if dists[0] + dists[1] == dists[2]:
            raise ValueError("Degenerate triangle!")

    def contains_point(self, x: float, y: float) -> bool:
        # Adapted from https://stackoverflow.com/questions/2049582/.
        sign1 = (
            (x - self.x2) * (self.y1 - self.y2) - (self.x1 - self.x2) * (y - self.y2)
        ) > 0
        sign2 = (
            (x - self.x3) * (self.y2 - self.y3) - (self.x2 - self.x3) * (y - self.y3)
        ) > 0
        sign3 = (
            (x - self.x1) * (self.y3 - self.y1) - (self.x3 - self.x1) * (y - self.y1)
        ) > 0
        has_neg = (not sign1) or (not sign2) or (not sign3)
        has_pos = sign1 or sign2 or sign3
        return not has_neg or not has_pos

    def sample_random_point(self, rng: np.random.Generator) -> Tuple[float, float]:
        a = np.array([self.x2 - self.x1, self.y2 - self.y1])
        b = np.array([self.x3 - self.x1, self.y3 - self.y1])
        u1 = rng.uniform(0, 1)
        u2 = rng.uniform(0, 1)
        if u1 + u2 > 1.0:
            u1 = 1 - u1
            u2 = 1 - u2
        point_in_triangle = (u1 * a + u2 * b) + np.array([self.x1, self.y1])
        assert self.contains_point(point_in_triangle[0], point_in_triangle[1])
        return (point_in_triangle[0], point_in_triangle[1])


@dataclass(frozen=True)
class Rectangle(Geom2D):
    """A helper class for visualizing and collision checking rectangles.

    Following the convention in plt.Rectangle, the origin is at the
    bottom left corner, and rotation is anti-clockwise about that point.

    Unlike plt.Rectangle, the angle is in radians.
    """

    x: float
    y: float
    width: float
    height: float
    theta: float  # in radians, between -np.pi and np.pi

    def __post_init__(self) -> None:
        assert -np.pi <= self.theta <= np.pi, "Expecting angle in [-pi, pi]."

    @staticmethod
    def from_center(
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        rotation_about_center: float,
    ) -> Rectangle:
        """Create a rectangle given an (x, y) for the center, with theta
        rotating about that center point."""
        x = center_x - width / 2
        y = center_y - height / 2
        norm_rect = Rectangle(x, y, width, height, 0.0)
        assert np.isclose(norm_rect.center[0], center_x)
        assert np.isclose(norm_rect.center[1], center_y)
        return norm_rect.rotate_about_point(center_x, center_y, rotation_about_center)

    @functools.cached_property
    def rotation_matrix(self) -> NDArray[np.float64]:
        """Get the rotation matrix."""
        return np.array(
            [
                [np.cos(self.theta), -np.sin(self.theta)],
                [np.sin(self.theta), np.cos(self.theta)],
            ]
        )

    @functools.cached_property
    def inverse_rotation_matrix(self) -> NDArray[np.float64]:
        """Get the inverse rotation matrix."""
        return np.array(
            [
                [np.cos(self.theta), np.sin(self.theta)],
                [-np.sin(self.theta), np.cos(self.theta)],
            ]
        )

    @functools.cached_property
    def vertices(self) -> List[Tuple[float, float]]:
        """Get the four vertices for the rectangle."""
        scale_matrix = np.array(
            [
                [self.width, 0],
                [0, self.height],
            ]
        )
        translate_vector = np.array([self.x, self.y])
        vertices = np.array(
            [
                (0, 0),
                (0, 1),
                (1, 1),
                (1, 0),
            ]
        )
        vertices = vertices @ scale_matrix.T
        vertices = vertices @ self.rotation_matrix.T
        vertices = translate_vector + vertices
        # Convert to a list of tuples. Slightly complicated to appease both
        # type checking and linting.
        return list(map(lambda p: (p[0], p[1]), vertices))

    @functools.cached_property
    def line_segments(self) -> List[LineSegment]:
        """Get the four line segments for the rectangle."""
        vs = list(zip(self.vertices, self.vertices[1:] + [self.vertices[0]]))
        line_segments = []
        for (x1, y1), (x2, y2) in vs:
            line_segments.append(LineSegment(x1, y1, x2, y2))
        return line_segments

    @functools.cached_property
    def center(self) -> Tuple[float, float]:
        """Get the point at the center of the rectangle."""
        x, y = np.mean(self.vertices, axis=0)
        return (x, y)

    @functools.cached_property
    def circumscribed_circle(self) -> Circle:
        """Returns x, y, radius."""
        x, y = self.center
        radius = np.sqrt((self.width / 2) ** 2 + (self.height / 2) ** 2)
        return Circle(x, y, radius)

    def contains_point(self, x: float, y: float) -> bool:
        # First invert translation, then invert rotation.
        rx, ry = np.array([x - self.x, y - self.y]) @ self.inverse_rotation_matrix.T
        return 0 <= rx <= self.width and 0 <= ry <= self.height

    def sample_random_point(self, rng: np.random.Generator) -> Tuple[float, float]:
        rand_width = rng.uniform(0, self.width)
        rand_height = rng.uniform(0, self.height)
        # First rotate, then translate.
        rx, ry = np.array([rand_width, rand_height]) @ self.rotation_matrix.T
        x = rx + self.x
        y = ry + self.y
        assert self.contains_point(x, y)
        return (x, y)

    def rotate_about_point(self, x: float, y: float, rot: float) -> Rectangle:
        """Create a new rectangle that is this rectangle, but rotated CCW by
        the given rotation (in radians), relative to the (x, y) origin.

        Rotates the vertices first, then uses them to recompute the new
        theta.
        """
        vertices = np.array(self.vertices)
        origin = np.array([x, y])
        # Translate the vertices so that they become the "origin".
        vertices = vertices - origin
        # Rotate.
        rotate_matrix = np.array(
            [[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]]
        )
        vertices = vertices @ rotate_matrix.T
        # Translate the vertices back.
        vertices = vertices + origin
        # Recompute theta.
        (lx, ly), _, _, (rx, ry) = vertices
        theta = np.arctan2(ry - ly, rx - lx)
        rect = Rectangle(lx, ly, self.width, self.height, theta)
        assert np.allclose(rect.vertices, vertices)
        return rect

    def scale_about_center(self, width_scale: float, height_scale: float) -> Rectangle:
        """Create a new rectangle that is this rectangle, shrunken or stretched
        about the center of this rectangle."""
        new_height = self.height * height_scale
        new_width = self.width * width_scale
        return Rectangle.from_center(
            self.center[0], self.center[1], new_width, new_height, self.theta
        )

    def plot(self, ax: plt.Axes, **kwargs: Any) -> None:
        angle = self.theta * 180 / np.pi
        patch = patches.Rectangle(
            (self.x, self.y), self.width, self.height, angle=angle, **kwargs
        )
        ax.add_patch(patch)


def line_segments_intersect(seg1: LineSegment, seg2: LineSegment) -> bool:
    """Checks if two line segments intersect.

    This method, which works by checking relative orientation, allows
    for collinearity, and only checks if each segment straddles the line
    containing the other.
    """

    def _subtract(
        a: Tuple[float, float], b: Tuple[float, float]
    ) -> Tuple[float, float]:
        x1, y1 = a
        x2, y2 = b
        return (x1 - x2), (y1 - y2)

    def _cross_product(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        x1, y1 = b
        x2, y2 = a
        return x1 * y2 - x2 * y1

    def _direction(
        a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]
    ) -> float:
        return _cross_product(_subtract(a, c), _subtract(a, b))

    p1 = (seg1.x1, seg1.y1)
    p2 = (seg1.x2, seg1.y2)
    p3 = (seg2.x1, seg2.y1)
    p4 = (seg2.x2, seg2.y2)
    d1 = _direction(p3, p4, p1)
    d2 = _direction(p3, p4, p2)
    d3 = _direction(p1, p2, p3)
    d4 = _direction(p1, p2, p4)

    return ((d2 < 0 < d1) or (d1 < 0 < d2)) and ((d4 < 0 < d3) or (d3 < 0 < d4))


def circles_intersect(circ1: Circle, circ2: Circle) -> bool:
    """Checks if two circles intersect."""
    x1, y1, r1 = circ1.x, circ1.y, circ1.radius
    x2, y2, r2 = circ2.x, circ2.y, circ2.radius
    return (x1 - x2) ** 2 + (y1 - y2) ** 2 < (r1 + r2) ** 2


def rectangles_intersect(rect1: Rectangle, rect2: Rectangle) -> bool:
    """Checks if two rectangles intersect."""
    # Optimization: if the circumscribed circles don't intersect, then
    # the rectangles also don't intersect.
    if not circles_intersect(rect1.circumscribed_circle, rect2.circumscribed_circle):
        return False
    # Case 1: line segments intersect.
    if any(
        line_segments_intersect(seg1, seg2)
        for seg1 in rect1.line_segments
        for seg2 in rect2.line_segments
    ):
        return True
    # Case 2: rect1 inside rect2.
    if rect1.contains_point(rect2.center[0], rect2.center[1]):
        return True
    # Case 3: rect2 inside rect1.
    if rect2.contains_point(rect1.center[0], rect1.center[1]):
        return True
    # Not intersecting.
    return False


def line_segment_intersects_circle(
    seg: LineSegment,
    circ: Circle,
) -> bool:
    """Checks if a line segment intersects a circle."""
    # First check if the end points of the segment are in the circle.
    if circ.contains_point(seg.x1, seg.y1):
        return True
    if circ.contains_point(seg.x2, seg.y2):
        return True
    # Project the circle radius onto the extended line.
    c = (circ.x, circ.y)
    # Project (a, c) onto (a, b).
    a = (seg.x1, seg.y1)
    b = (seg.x2, seg.y2)
    ba = np.subtract(b, a)
    ca = np.subtract(c, a)
    da = ba * np.dot(ca, ba) / np.dot(ba, ba)
    # The point on the extended line that is the closest to the center.
    dx, dy = (a[0] + da[0], a[1] + da[1])
    # Check if the point is on the line. If it's not, there is no intersection,
    # because we already checked that the circle does not contain the end
    # points of the line segment.
    if not seg.contains_point(dx, dy):
        return False
    # So d is on the segment. Check if it's in the circle.
    return circ.contains_point(dx, dy)


def line_segment_intersects_rectangle(seg: LineSegment, rect: Rectangle) -> bool:
    """Checks if a line segment intersects a rectangle."""
    # Case 1: one of the end points of the segment is in the rectangle.
    if rect.contains_point(seg.x1, seg.y1) or rect.contains_point(seg.x2, seg.y2):
        return True
    # Case 2: the segment intersects with one of the rectangle sides.
    return any(line_segments_intersect(s, seg) for s in rect.line_segments)


def rectangle_intersects_circle(rect: Rectangle, circ: Circle) -> bool:
    """Checks if a rectangle intersects a circle."""
    # Optimization: if the circumscribed circle of the rectangle doesn't
    # intersect with the circle, then there can't be an intersection.
    if not circles_intersect(rect.circumscribed_circle, circ):
        return False
    # Case 1: the circle's center is in the rectangle.
    if rect.contains_point(circ.x, circ.y):
        return True
    # Case 2: one of the sides of the rectangle intersects the circle.
    for seg in rect.line_segments:
        if line_segment_intersects_circle(seg, circ):
            return True
    return False


def geom2ds_intersect(geom1: Geom2D, geom2: Geom2D) -> bool:
    """Check if two 2D bodies intersect."""
    if isinstance(geom1, LineSegment) and isinstance(geom2, LineSegment):
        return line_segments_intersect(geom1, geom2)
    if isinstance(geom1, LineSegment) and isinstance(geom2, Circle):
        return line_segment_intersects_circle(geom1, geom2)
    if isinstance(geom1, LineSegment) and isinstance(geom2, Rectangle):
        return line_segment_intersects_rectangle(geom1, geom2)
    if isinstance(geom1, Rectangle) and isinstance(geom2, LineSegment):
        return line_segment_intersects_rectangle(geom2, geom1)
    if isinstance(geom1, Circle) and isinstance(geom2, LineSegment):
        return line_segment_intersects_circle(geom2, geom1)
    if isinstance(geom1, Rectangle) and isinstance(geom2, Rectangle):
        return rectangles_intersect(geom1, geom2)
    if isinstance(geom1, Rectangle) and isinstance(geom2, Circle):
        return rectangle_intersects_circle(geom1, geom2)
    if isinstance(geom1, Circle) and isinstance(geom2, Rectangle):
        return rectangle_intersects_circle(geom2, geom1)
    if isinstance(geom1, Circle) and isinstance(geom2, Circle):
        return circles_intersect(geom1, geom2)
    raise NotImplementedError(
        "Intersection not implemented for geoms " f"{geom1} and {geom2}"
    )
