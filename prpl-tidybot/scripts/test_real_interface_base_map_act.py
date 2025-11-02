"""Tests for real interface base map action."""

import math
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

    # get initial pose
    observation = interface.get_observation()
    pose_map = (
        observation.map_base_pose.x,
        observation.map_base_pose.y,
        observation.map_base_pose.theta(),
    )
    pose_odom = (
        observation.base_pose.x,
        observation.base_pose.y,
        observation.base_pose.theta(),
    )
    map_to_odom_converter.update(pose_map, pose_odom)
    odom_to_map_converter.update(pose_odom, pose_map)

    try:
        target_map_pose = (-0.5, -0.5, math.pi)
        target_odom_pose = map_to_odom_converter.convert_pose(target_map_pose)
        print(f"target_odom_pose: {target_odom_pose}")
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
            pose_map = (
                observation.map_base_pose.x,
                observation.map_base_pose.y,
                observation.map_base_pose.theta(),
            )
            pose_odom = (
                observation.base_pose.x,
                observation.base_pose.y,
                observation.base_pose.theta(),
            )
            map_to_odom_converter.update(pose_map, pose_odom)
            odom_to_map_converter.update(pose_odom, pose_map)
            if (
                np.linalg.norm(
                    np.array([target_map_pose[0], target_map_pose[1]])
                    - np.array([pose_map[0], pose_map[1]])
                )
                < 0.01
                and abs(target_map_pose[2] - pose_map[2]) < 0.01
            ):
                print(f"Reached target pose: {target_odom_pose}")
                break

            target_odom_pose = map_to_odom_converter.convert_pose(target_map_pose)

    finally:
        interface.close()
