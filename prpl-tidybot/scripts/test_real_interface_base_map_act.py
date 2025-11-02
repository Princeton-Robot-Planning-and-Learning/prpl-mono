"""Tests for real_env.py."""

import time

import numpy as np

from prpl_tidybot.constants import POLICY_CONTROL_PERIOD
from prpl_tidybot.coord_converter import CoordFrameConverter
from prpl_tidybot.interfaces.interface import RealInterface

if __name__ == "__main__":
    interface = RealInterface()

    # initialization
    pose_map = (0, 0, 0)
    pose_odom = (0, 0, 0)
    map_to_odom_converter = CoordFrameConverter(pose_map, pose_odom)
    odom_to_map_converter = CoordFrameConverter(pose_odom, pose_map)

    try:
        for i in range(20):
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
                (
                    observation.map_base_pose.x,
                    observation.map_base_pose.y,
                    observation.map_base_pose.theta(),
                ),
                (
                    observation.base_pose.x,
                    observation.base_pose.y,
                    observation.base_pose.theta(),
                ),
            )
            odom_to_map_converter.update(
                (
                    observation.base_pose.x,
                    observation.base_pose.y,
                    observation.base_pose.theta(),
                ),
                (
                    observation.map_base_pose.x,
                    observation.map_base_pose.y,
                    observation.map_base_pose.theta(),
                ),
            )

            interface.execute_base_action(
                {"base_pose": np.array([(i / 20) * 0.5, 0.0, 0.0])}
            )
            time.sleep(POLICY_CONTROL_PERIOD)

        target_map_pose = (0.0, 0.0, 0.0)
        target_odom_pose = map_to_odom_converter.convert_pose(target_map_pose)
        for i in range(100):
            interface.execute_base_action(
                {
                    "base_pose": np.array(
                        [target_odom_pose[0], target_odom_pose[1], target_odom_pose[2]]
                    )
                }
            )
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

    finally:
        interface.close()
