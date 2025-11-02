"""Tests for real interface base action in local coordinate frame."""

import time

import numpy as np

from prpl_tidybot.constants import POLICY_CONTROL_PERIOD
from prpl_tidybot.interfaces.interface import RealInterface

if __name__ == "__main__":
    interface = RealInterface()
    try:
        for i in range(50):
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
            interface.execute_base_action(
                {"base_pose": np.array([(i / 50) * 0.5, 0.0, 0.0])}
            )
            time.sleep(POLICY_CONTROL_PERIOD)
    finally:
        interface.close()
