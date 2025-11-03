"""Coordinate frame converter."""

import math

from spatialmath import SE2


class CoordFrameConverter:
    """Coordinate frame converter."""

    def __init__(
        self,
        pose_in_frame_a: SE2,
        pose_in_frame_b: SE2,
    ) -> None:
        """Initialize the coordinate frame converter."""
        self.origin = (0.0, 0.0)
        self.basis = 0.0
        self.update(pose_in_frame_a, pose_in_frame_b)

    def update(
        self,
        pose_in_frame_a: SE2,
        pose_in_frame_b: SE2,
    ) -> None:
        """Update the coordinate frame converter."""
        self.basis = pose_in_frame_a.theta() - pose_in_frame_b.theta()
        dx = pose_in_frame_b.x * math.cos(self.basis) - pose_in_frame_b.y * math.sin(
            self.basis
        )
        dy = pose_in_frame_b.x * math.sin(self.basis) + pose_in_frame_b.y * math.cos(
            self.basis
        )
        self.origin = (pose_in_frame_a.x - dx, pose_in_frame_a.y - dy)

    def convert_pose(self, pose: SE2) -> SE2:
        """Convert a pose from frame a to frame b."""
        x, y, th = pose.x, pose.y, pose.theta()

        # Convert position
        x = x - self.origin[0]
        y = y - self.origin[1]
        xp = x * math.cos(-self.basis) - y * math.sin(-self.basis)
        yp = x * math.sin(-self.basis) + y * math.cos(-self.basis)

        # Convert heading
        converted_th = th - self.basis

        return SE2(xp, yp, converted_th)
