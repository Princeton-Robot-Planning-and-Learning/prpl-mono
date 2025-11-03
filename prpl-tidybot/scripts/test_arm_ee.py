"""Test for arm with end-effector control."""

import time

import numpy as np

from prpl_tidybot.arm_server import ArmManager
from prpl_tidybot.constants import (
    ARM_RPC_HOST,
    ARM_RPC_PORT,
    POLICY_CONTROL_PERIOD,
    RPC_AUTHKEY,
)

if __name__ == "__main__":
    manager = ArmManager(address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
    manager.connect()
    arm = manager.Arm()  # type: ignore # pylint: disable=no-member
    try:
        arm.reset()
        for i in range(50):
            arm.execute_action(
                {
                    "arm_pos": np.array([0.135, 0.002, 0.211]),
                    "arm_quat": np.array([0.706, 0.707, 0.029, 0.029]),
                    "gripper_pos": np.zeros(1),
                }
            )
            print(arm.get_state())
            time.sleep(POLICY_CONTROL_PERIOD)  # Note: Not precise
    finally:
        arm.close()
