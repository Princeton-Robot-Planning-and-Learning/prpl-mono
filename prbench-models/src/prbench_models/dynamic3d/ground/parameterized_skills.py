"""Parameterized skills for the TidyBot3D ground environment."""

import numpy as np
from spatialmath import SE2


# Utility functions.
def get_target_robot_pose_from_parameters(
    target_object_pose: SE2, target_distance: float, target_rot: float
) -> SE2:
    """Determine the pose for the robot given the state and parameters.

    The robot will be facing the target_object_pose position while being target_distance
    away, and rotated w.r.t. the target_object_pose rotation by target_rot.
    """
    # Absolute angle of the line from the robot to the target.
    ang = target_object_pose.theta() + target_rot

    # Place the robot `target_distance` away from the target along -ang
    tx, ty = target_object_pose.t  # target translation (x, y).
    rx = tx - target_distance * np.cos(ang)
    ry = ty - target_distance * np.sin(ang)

    # Robot faces the target: heading points along +ang (toward the target).
    return SE2(rx, ry, ang)
