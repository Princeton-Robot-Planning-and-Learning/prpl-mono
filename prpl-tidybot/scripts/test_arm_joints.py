"""Test for arm with end-effector control."""

import time

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
            print(arm.get_joint_angles())
            print(arm.get_gripper_position())
            time.sleep(POLICY_CONTROL_PERIOD)  # Note: Not precise
    finally:
        arm.close()
