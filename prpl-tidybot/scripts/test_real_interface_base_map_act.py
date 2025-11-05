"""Tests for real interface base map action."""

import math
import time

from spatialmath import SE2

from prpl_tidybot.base_movement import reach_target_pose
from prpl_tidybot.coord_converter import CoordFrameConverter
from prpl_tidybot.interfaces.interface import RealInterface

if __name__ == "__main__":
    interface = RealInterface()

    try:
        # initialization
        pose_map = SE2(0, 0, 0)
        pose_odom = SE2(0, 0, 0)
        map_to_odom_converter = CoordFrameConverter(pose_map, pose_odom)
        odom_to_map_converter = CoordFrameConverter(pose_odom, pose_map)

        # get initial pose
        observation = interface.get_observation()
        map_to_odom_converter.update(observation.map_base_pose, observation.base_pose)
        odom_to_map_converter.update(observation.base_pose, observation.map_base_pose)

        target_map_pose_list = [
            SE2(0.5, 0.5, math.pi / 2),
            SE2(-0.5, 0.5, 0),
            SE2(0.5, -0.5, math.pi),
        ]
        for target_map_pose in target_map_pose_list:
            reach_target_pose(
                interface, target_map_pose, map_to_odom_converter, odom_to_map_converter
            )
            time.sleep(1)

    finally:
        interface.close()
