"""Tests for real interface base map action."""

import math
import time

import numpy as np
from spatialmath import SE2

from prpl_tidybot.constants import POLICY_CONTROL_PERIOD
from prpl_tidybot.coord_converter import CoordFrameConverter
from prpl_tidybot.interfaces.interface import RealInterface
from prpl_tidybot.structs import TidyBotAction

if __name__ == "__main__":
    interface = RealInterface()

    # initialization
    pose_map = SE2(0, 0, 0)
    pose_odom = SE2(0, 0, 0)
    map_to_odom_converter = CoordFrameConverter(pose_map, pose_odom)
    odom_to_map_converter = CoordFrameConverter(pose_odom, pose_map)

    # get initial pose
    observation = interface.get_observation()
    map_to_odom_converter.update(observation.map_base_pose, observation.base_pose)
    odom_to_map_converter.update(observation.base_pose, observation.map_base_pose)

    try:
        target_map_pose = SE2(0.5, 0.5, -math.pi / 2)
        target_odom_pose = map_to_odom_converter.convert_pose(target_map_pose)
        print(
            "target_odom_pose:",
            target_odom_pose.x,
            target_odom_pose.y,
            target_odom_pose.theta(),
        )
        for i in range(100):
            tidybot_action = TidyBotAction(
                arm_goal=[0.0] * 7,
                base_local_goal=target_odom_pose,
                gripper_goal=1.0,
            )
            interface.execute_base_action(tidybot_action)
            time.sleep(POLICY_CONTROL_PERIOD)
            observation = interface.get_observation()
            print(
                "base pose (quat):",
                observation.base_pose.x,
                observation.base_pose.y,
                observation.base_pose.theta(),
            )
            print(
                "map base pose (quat):",
                observation.map_base_pose.x,
                observation.map_base_pose.y,
                observation.map_base_pose.theta(),
            )
            map_to_odom_converter.update(
                observation.map_base_pose, observation.base_pose
            )
            odom_to_map_converter.update(
                observation.base_pose, observation.map_base_pose
            )
            if (
                np.linalg.norm(
                    np.array([target_map_pose.x, target_map_pose.y])
                    - np.array(
                        [observation.map_base_pose.x, observation.map_base_pose.y]
                    )
                )
                < 0.01
                and abs(target_map_pose.theta() - observation.map_base_pose.theta())
                < 0.01
            ):
                print(
                    "Reached target pose:",
                    target_odom_pose.x,
                    target_odom_pose.y,
                    target_odom_pose.theta(),
                )
                break

            target_odom_pose = map_to_odom_converter.convert_pose(target_map_pose)

    finally:
        interface.close()
