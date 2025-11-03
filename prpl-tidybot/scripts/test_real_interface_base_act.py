"""Tests for real interface base action in local coordinate frame."""

import time

from spatialmath import SE2

from prpl_tidybot.constants import POLICY_CONTROL_PERIOD
from prpl_tidybot.interfaces.interface import RealInterface
from prpl_tidybot.structs import TidyBotAction

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
            tidybot_action = TidyBotAction(
                arm_goal=[0.0] * 7,
                base_local_goal=SE2(x=(i / 50) * 0.5, y=0.0, theta=0.0),
                gripper_goal=1.0,
            )
            interface.execute_base_action(tidybot_action)
            time.sleep(POLICY_CONTROL_PERIOD)
    finally:
        interface.close()
