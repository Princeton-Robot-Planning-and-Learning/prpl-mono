"""Tests for real interface base action in local coordinate frame."""

import time

import numpy as np

from prpl_tidybot.constants import POLICY_CONTROL_PERIOD
from prpl_tidybot.ik_solver import IKSolver
from prpl_tidybot.interfaces.interface import RealInterface
from prpl_tidybot.structs import TidyBotAction

if __name__ == "__main__":
    retract_qpos = np.deg2rad([0, -20, 180, -146, 0, -50, 90])
    ik_solver = IKSolver()  # type: ignore
    home_pos, home_quat = np.array([0.456, 0.0, 0.434]), np.array([0.5, 0.5, 0.5, 0.5])
    home_joint_angles = ik_solver.solve(home_pos, home_quat, retract_qpos)  # type: ignore # pylint: disable=line-too-long
    interface = RealInterface()
    try:
        for i in range(20):
            observation = interface.get_observation()
            print(
                "current arm joint angles:",
                observation.arm_conf,
            )
            tidybot_action = TidyBotAction(
                base_local_goal=interface.get_base_state(),
                arm_goal=retract_qpos.tolist(),
                gripper_goal=interface.get_gripper_state(),
            )
            interface.execute_arm_action(tidybot_action)
            time.sleep(POLICY_CONTROL_PERIOD)
        # print('closing gripper')
        # for i in range(10):
        #     observation = interface.get_observation()
        #     print(
        #         "current arm joint angles:",
        #         observation.arm_conf,
        #     )
        #     tidybot_action = TidyBotAction(
        #         base_local_goal=interface.get_base_state(),
        #         arm_goal=interface.get_arm_state(),
        #         gripper_goal=1.0,
        #     )
        #     interface.execute_gripper_action(tidybot_action)
        #     time.sleep(POLICY_CONTROL_PERIOD)

    finally:
        interface.close()
