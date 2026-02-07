"""Constants for kinematic3d environments."""

import numpy as np
from pybullet_helpers.geometry import Pose

GRASP_TRANSFORM_TO_OBJECT = Pose((0.0, 0, 0.02), (0.707, 0.707, 0, 0))
GRIPPER_OPEN_THRESHOLD = 0.01
HOME_JOINT_POSITIONS = np.deg2rad([0, -20, 180, -146, 0, -50, 90, 0, 0, 0, 0, 0, 0])
GRIPPER_CLOSE_THRESHOLD = 0.05
