"""Data structures."""

from dataclasses import dataclass

import spatialmath
from prpl_utils.structs import Image

from prpl_tidybot.constants import CAMERA_DIMS


@dataclass(frozen=True)
class TidyBotObservation:
    """Raw observations from the TidyBot environment."""

    arm_conf: list[float]  # 7-DOF joints
    base_pose: spatialmath.SE2  # base pose for the robot
    gripper: float  # 1 = closed, 0 = open
    wrist_camera: Image  # see CAMERA_DIMS
    base_camera: Image  # see CAMERA_DIMS

    def __post_init__(self) -> None:
        # Make sure the camera dimensions are as expected.
        assert self.wrist_camera.shape == CAMERA_DIMS
        assert self.base_camera.shape == CAMERA_DIMS


@dataclass(frozen=True)
class TidyBotAction:
    """Low-level joint and base commands for the real TidyBot environment."""

    arm_goal: list[float]  # absolute arm position
    base_local_goal: spatialmath.SE2  # relative to current base pose
    gripper_goal: float  # absolute
