"""Utils for kinematic3d environments."""

import numpy as np
from pybullet_helpers.geometry import SE2Pose


# Utility functions.
def get_target_robot_pose_from_parameters(
    target_object_pose: SE2Pose, target_distance: float, target_rot: float
) -> SE2Pose:
    """Determine the pose for the robot given the state and parameters.

    The robot will be facing the target_object_pose position while being target_distance
    away, and rotated w.r.t. the target_object_pose rotation by target_rot.
    """
    # Absolute angle of the line from the robot to the target.
    ang = target_object_pose.rot + target_rot

    # Place the robot `target_distance` away from the target along -ang
    tx, ty = target_object_pose.x, target_object_pose.y  # target translation (x, y).
    rx = tx - target_distance * np.cos(ang)
    ry = ty - target_distance * np.sin(ang)

    # Robot faces the target: heading points along +ang (toward the target).
    return SE2Pose(rx, ry, ang)
