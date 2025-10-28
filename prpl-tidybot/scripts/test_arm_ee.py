# Author: Jimmy Wu
# Date: October 2024
#
# This RPC server allows other processes to communicate with the Kinova arm
# low-level controller, which runs in its own, dedicated real-time process.
#
# Note: Operations that are not time-sensitive should be run in a separate,
# non-real-time process to avoid interfering with the low-level control and
# causing latency spikes.

import time
import numpy as np
from prpl_tidybot.arm_server import ArmManager
from prpl_tidybot.constants import ARM_RPC_HOST, ARM_RPC_PORT, RPC_AUTHKEY
from prpl_tidybot.constants import POLICY_CONTROL_PERIOD

if __name__ == '__main__':
    manager = ArmManager(address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
    manager.connect()
    arm = manager.Arm()
    try:
        arm.reset()
        for i in range(50):
            arm.execute_action({
                'arm_pos': np.array([0.135, 0.002, 0.211]),
                'arm_quat': np.array([0.706, 0.707, 0.029, 0.029]),
                'gripper_pos': np.zeros(1),
            })
            print(arm.get_state())
            time.sleep(POLICY_CONTROL_PERIOD)  # Note: Not precise
    finally:
        arm.close()
