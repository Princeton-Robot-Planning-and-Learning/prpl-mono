"""Utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from tomsgeoms2d.structs import Circle, Geom2D, LineSegment, Lobject, Rectangle


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


def lobject_intersects_rectangle(lobj: Lobject, rect: Rectangle) -> bool:
    """Checks if a Lobject intersects a rectangle."""
    # Case 1: any vertex of the rectangle is inside the Lobject.
    if any(lobj.contains_point(vx, vy) for vx, vy in rect.vertices):
        return True
    # Case 2: any vertex of the Lobject is inside the rectangle.
    if any(rect.contains_point(vx, vy) for vx, vy in lobj.vertices):
        return True
    # Case 3: any edge of the Lobject intersects the rectangle.
    for seg1 in lobj.line_segments:
        for seg2 in rect.line_segments:
            if line_segments_intersect(seg1, seg2):
                return True
    return False


def lobject_intersects_circle(lobj1: Lobject, lobj2: Circle) -> bool:
    """Checks if a Lobject intersects a circle."""
    # Case 1: the circle's center is inside the Lobject.
    if lobj1.contains_point(lobj2.x, lobj2.y):
        return True
    # Case 2: any vertex of the Lobject is inside the circle.
    if any(lobj2.contains_point(vx, vy) for vx, vy in lobj1.vertices):
        return True
    # Case 3: any edge of the Lobject intersects the circle.
    for seg1 in lobj1.line_segments:
        if line_segment_intersects_circle(seg1, lobj2):
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
    if isinstance(geom1, Lobject) and isinstance(geom2, Rectangle):
        return lobject_intersects_rectangle(geom1, geom2)
    if isinstance(geom1, Rectangle) and isinstance(geom2, Lobject):
        return lobject_intersects_rectangle(geom2, geom1)
    if isinstance(geom1, Lobject) and isinstance(geom2, Circle):
        return lobject_intersects_circle(geom1, geom2)
    if isinstance(geom1, Circle) and isinstance(geom2, Lobject):
        return lobject_intersects_circle(geom2, geom1)
    raise NotImplementedError(
        "Intersection not implemented for geoms " f"{geom1} and {geom2}"
    )


def find_closest_point_line(
    line: LineSegment, point: Tuple[float, float]
) -> Tuple[Tuple[float, float], float]:
    """Find the closest point on a line segment to a given point."""
    # Get the line segment endpoints.
    x1, y1 = line.x1, line.y1
    x2, y2 = line.x2, line.y2
    # Vector from point to line start.
    px, py = point
    dx, dy = x2 - x1, y2 - y1
    # Project point onto the line.
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    # Clamp t to the line segment.
    t = max(0, min(1, t))
    # Find the closest point on the line segment.
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    min_dist = np.linalg.norm(np.array((closest_x, closest_y)) - np.array(point)).item()

    return (closest_x, closest_y), min_dist


def find_closest_point_circle(
    circ: Circle, point: Tuple[float, float]
) -> Tuple[Tuple[float, float], float]:
    """Find the closest point on a circle to a given point."""
    # Get the circle center and radius.
    cx, cy, radius = circ.x, circ.y, circ.radius
    # Vector from center to point.
    dx, dy = point[0] - cx, point[1] - cy
    # If the point is inside the circle, the closest point is the point itself.
    if np.linalg.norm((dx, dy)) < radius:
        return point, 0.0
    # Otherwise, project the point onto the circle.
    angle = np.arctan2(dy, dx)
    closest_x = cx + radius * np.cos(angle)
    closest_y = cy + radius * np.sin(angle)

    min_dist = np.linalg.norm(np.array((closest_x, closest_y)) - np.array(point)).item()

    return (closest_x, closest_y), min_dist


def find_closest_points_line_line(
    line1: LineSegment, line2: LineSegment
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Find the closest points between two line segments."""
    # Get the line segment endpoints.
    p1 = np.array([line1.x1, line1.y1])
    p2 = np.array([line1.x2, line1.y2])
    p3 = np.array([line2.x1, line2.y1])
    p4 = np.array([line2.x2, line2.y2])

    # Define segment directions
    d1 = p2 - p1  # Direction vector of segment S1
    d2 = p4 - p3  # Direction vector of segment S2
    r = p1 - p3

    a = np.dot(d1, d1)  # Squared length of segment S1
    e = np.dot(d2, d2)  # Squared length of segment S2
    f = np.dot(d2, r)

    EPS = 1e-12
    if a <= EPS and e <= EPS:
        # Both segments are just points
        return p1, p3

    if a <= EPS:
        # First segment is a point
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    elif e <= EPS:
        # Second segment is a point
        t = 0.0
        s = np.clip(-np.dot(d1, r) / a, 0.0, 1.0)
    else:
        b = np.dot(d1, d2)
        c = np.dot(d1, r)

        denom = a * e - b * b
        if denom != 0.0:
            s = np.clip((b * f - c * e) / denom, 0.0, 1.0)
        else:
            # Parallel segments
            s = 0.0

        tnom = b * s + f
        if tnom < 0.0:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        elif tnom > e:
            t = 1.0
            s = np.clip((b - c) / a, 0.0, 1.0)
        else:
            t = tnom / e

    closest_point1 = p1 + s * d1
    closest_point2 = p3 + t * d2

    return closest_point1, closest_point2


def find_closest_points_circle_circle(
    circ1: Circle, circ2: Circle
) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """Find the closest points between two circles."""
    # Get the centers and radii of the circles.
    cx1, cy1, r1 = circ1.x, circ1.y, circ1.radius
    cx2, cy2, r2 = circ2.x, circ2.y, circ2.radius

    # Vector from center of circle 1 to center of circle 2.
    dx, dy = cx2 - cx1, cy2 - cy1
    dist = np.linalg.norm((dx, dy)).item()

    # If the circles overlap, the closest points are the same.
    if dist < r1 + r2:
        return (cx1, cy1), (cx1, cy1), dist

    # Otherwise, find the closest points on the edges of the circles.
    angle = np.arctan2(dy, dx)
    closest1 = (cx1 + r1 * np.cos(angle), cy1 + r1 * np.sin(angle))
    closest2 = (cx2 - r2 * np.cos(angle), cy2 - r2 * np.sin(angle))

    min_dist = np.linalg.norm(np.array(closest1) - np.array(closest2)).item()

    return closest1, closest2, min_dist


def find_closest_points_object_circle(
    obj: Lobject | Rectangle, circ: Circle
) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """Find the closest points between an L-object and a circle."""
    # Get the center of the circle.
    circ_center = (circ.x, circ.y)
    # Find the minimum distance of the circle and each line segment
    min_dist = float("inf")
    closest_lobj_point = (obj.x, obj.y)
    for i in range(len(obj.line_segments)):
        seg = obj.line_segments[i]
        closest_point, dist = find_closest_point_line(seg, circ_center)
        if dist < min_dist:
            min_dist = dist
            closest_lobj_point = closest_point

    return closest_lobj_point, circ_center, min_dist


def find_closest_points_object_object(
    obj1: Lobject | Rectangle, obj2: Lobject | Rectangle
) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """Find the closest points between two L-objects or rectangles."""

    min_dist = float("inf")
    closest_obj1_point = (obj1.x, obj1.y)
    closest_obj2_point = (obj2.x, obj2.y)
    for i in range(len(obj1.line_segments)):
        seg1 = obj1.line_segments[i]
        for j in range(len(obj2.line_segments)):
            seg2 = obj2.line_segments[j]
            closest_points = find_closest_points_line_line(seg1, seg2)
            if line_segments_intersect(seg1, seg2):
                return closest_points[0], closest_points[1], 0.0
            dist = np.linalg.norm(
                np.array(closest_points[0]) - np.array(closest_points[1])
            ).item()
            if dist < min_dist:
                min_dist = dist
                closest_obj1_point = closest_points[0]
                closest_obj2_point = closest_points[1]

    return closest_obj1_point, closest_obj2_point, min_dist


def find_closest_points(
    geom1: Geom2D, geom2: Geom2D
) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """Find the closest points between two objects."""
    if isinstance(geom1, Circle) and isinstance(geom2, Circle):
        return find_closest_points_circle_circle(geom1, geom2)
    if isinstance(geom1, Rectangle) and isinstance(geom2, Circle):
        return find_closest_points_object_circle(geom1, geom2)
    if isinstance(geom1, Circle) and isinstance(geom2, Rectangle):
        return find_closest_points_object_circle(geom2, geom1)
    if isinstance(geom1, Lobject) and isinstance(geom2, Circle):
        return find_closest_points_object_circle(geom1, geom2)
    if isinstance(geom1, Circle) and isinstance(geom2, Lobject):
        return find_closest_points_object_circle(geom2, geom1)
    if isinstance(geom1, Lobject) and isinstance(geom2, Lobject):
        return find_closest_points_object_object(geom1, geom2)
    if isinstance(geom1, Lobject) and isinstance(geom2, Rectangle):
        return find_closest_points_object_object(geom1, geom2)
    if isinstance(geom1, Rectangle) and isinstance(geom2, Lobject):
        return find_closest_points_object_object(geom2, geom1)
    if isinstance(geom1, Rectangle) and isinstance(geom2, Rectangle):
        return find_closest_points_object_object(geom1, geom2)

    raise TypeError(
        f"Incompatible objects: {geom1} and {geom2}. "
        f"Only Circle, Rectangle, and Lobject are supported."
    )
