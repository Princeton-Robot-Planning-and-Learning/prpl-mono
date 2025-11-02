"""Coordinate frame converter."""

import math


class CoordFrameConverter:
    """Coordinate frame converter."""

    def __init__(
        self,
        pose_in_map: tuple[float, float, float],
        pose_in_odom: tuple[float, float, float],
    ) -> None:
        """Initialize the coordinate frame converter."""
        self.origin = (0.0, 0.0)
        self.basis = 0.0
        self.update(pose_in_map, pose_in_odom)

    def update(
        self,
        pose_in_map: tuple[float, float, float],
        pose_in_odom: tuple[float, float, float],
    ) -> None:
        """Update the coordinate frame converter."""
        self.basis = pose_in_map[2] - pose_in_odom[2]
        dx = pose_in_odom[0] * math.cos(self.basis) - pose_in_odom[1] * math.sin(
            self.basis
        )
        dy = pose_in_odom[0] * math.sin(self.basis) + pose_in_odom[1] * math.cos(
            self.basis
        )
        self.origin = (pose_in_map[0] - dx, pose_in_map[1] - dy)

    def convert_position(self, position: tuple[float, float]) -> tuple[float, float]:
        """Convert a position from the one frame to another frame."""
        x, y = position
        x = x - self.origin[0]
        y = y - self.origin[1]
        xp = x * math.cos(-self.basis) - y * math.sin(
            -self.basis
        )  # pylint: disable=invalid-unary-operand-type
        yp = x * math.sin(-self.basis) + y * math.cos(
            -self.basis
        )  # pylint: disable=invalid-unary-operand-type
        # Invert output coordinates for this mobile base
        return (xp, yp)

    def convert_heading(self, th: float) -> float:
        """Convert a heading from the one frame to another frame."""
        # Invert heading for this mobile base
        return th - self.basis

    def convert_pose(
        self, pose: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """Convert a pose from the one frame to another frame."""
        x, y, th = pose
        return (*self.convert_position((x, y)), self.convert_heading(th))
