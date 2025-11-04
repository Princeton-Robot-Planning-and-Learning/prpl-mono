"""Base movement functions."""

import time

import numpy as np
from spatialmath import SE2

from prpl_tidybot.constants import POLICY_CONTROL_PERIOD
from prpl_tidybot.coord_converter import CoordFrameConverter
from prpl_tidybot.interfaces.interface import RealInterface
from prpl_tidybot.structs import TidyBotAction


def reach_target_pose(
    interface: RealInterface,
    target_map_pose: SE2,
    map_to_odom_converter: CoordFrameConverter,
    odom_to_map_converter: CoordFrameConverter,
    max_iter: int = 100,
    tolerance: float = 0.01,
) -> None:
    """Reaches the target pose in the map frame."""
    target_odom_pose = map_to_odom_converter.convert_pose(target_map_pose)
    for _ in range(max_iter):
        tidybot_action = TidyBotAction(
            arm_goal=[0.0] * 7,
            base_local_goal=target_odom_pose,
            gripper_goal=1.0,
        )
        interface.execute_base_action(tidybot_action)
        time.sleep(POLICY_CONTROL_PERIOD)
        observation = interface.get_observation()
        map_to_odom_converter.update(observation.map_base_pose, observation.base_pose)
        odom_to_map_converter.update(observation.base_pose, observation.map_base_pose)
        if (
            np.linalg.norm(
                np.array([target_map_pose.x, target_map_pose.y])
                - np.array([observation.map_base_pose.x, observation.map_base_pose.y])
            )
            < tolerance
            and abs(target_map_pose.theta() - observation.map_base_pose.theta())
            < tolerance
        ):
            break

        target_odom_pose = map_to_odom_converter.convert_pose(target_map_pose)
