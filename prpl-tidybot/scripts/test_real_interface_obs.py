"""Tests for real interface observation."""

import time

import cv2 as cv

from prpl_tidybot.constants import POLICY_CONTROL_PERIOD
from prpl_tidybot.interfaces.interface import RealInterface

if __name__ == "__main__":
    interface = RealInterface()
    try:
        for i in range(2):
            observation = interface.get_observation()
            print("arm conf:", observation.arm_conf)
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
            print("gripper:", observation.gripper)
            base_image = observation.base_camera
            wrist_image = observation.wrist_camera
            cv.imwrite(
                f"test_images/base_image_{i}.jpg",
                cv.cvtColor(base_image, cv.COLOR_RGB2BGR),
            )
            cv.imwrite(
                f"test_images/wrist_image_{i}.jpg",
                cv.cvtColor(wrist_image, cv.COLOR_RGB2BGR),
            )
            time.sleep(POLICY_CONTROL_PERIOD)
    finally:
        interface.close()
