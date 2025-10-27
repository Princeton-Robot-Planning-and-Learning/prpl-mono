# Author: Jimmy Wu
# Date: October 2024
#
# This RPC server allows other processes to communicate with the Kinova arm
# low-level controller, which runs in its own, dedicated real-time process.
#
# Note: Operations that are not time-sensitive should be run in a separate,
# non-real-time process to avoid interfering with the low-level control and
# causing latency spikes.

import queue
import time
from multiprocessing.managers import BaseManager as MPBaseManager
import numpy as np
from arm_controller import JointCompliantController
from constants import ARM_RPC_HOST, ARM_RPC_PORT, RPC_AUTHKEY
from ik_solver import IKSolver
from kinova import TorqueControlledArm

class Arm:
    def __init__(self):
        self.arm = TorqueControlledArm()
        self.arm.set_joint_limits(speed_limits=(7 * (30,)), acceleration_limits=(7 * (80,)))
        self.command_queue = queue.Queue(1)
        self.controller = None
        self.ik_solver = IKSolver(ee_offset=0.12)

    def reset(self):
        # Stop low-level control
        if self.arm.cyclic_running:
            time.sleep(0.75)  # Wait for arm to stop moving
            self.arm.stop_cyclic()

        # Clear faults
        self.arm.clear_faults()

        # Reset arm configuration
        self.arm.open_gripper()
        self.arm.retract()

        # Create new instance of controller
        self.controller = JointCompliantController(self.command_queue)

        # Start low-level control
        self.arm.init_cyclic(self.controller.control_callback)
        while not self.arm.cyclic_running:
            time.sleep(0.01)

    def execute_action(self, action):
        qpos = self.ik_solver.solve(action['arm_pos'], action['arm_quat'], self.arm.q)
        self.command_queue.put((qpos, action['gripper_pos'].item()))

    def get_state(self):
        arm_pos, arm_quat = self.arm.get_tool_pose()
        if arm_quat[3] < 0.0:  # Enforce quaternion uniqueness
            np.negative(arm_quat, out=arm_quat)
        state = {
            'arm_pos': arm_pos,
            'arm_quat': arm_quat,
            'gripper_pos': np.array([self.arm.gripper_pos]),
        }
        return state

    def close(self):
        if self.arm.cyclic_running:
            time.sleep(0.75)  # Wait for arm to stop moving
            self.arm.stop_cyclic()
        self.arm.disconnect()

class ArmManager(MPBaseManager):
    pass

ArmManager.register('Arm', Arm)

if __name__ == '__main__':
    manager = ArmManager(address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
    server = manager.get_server()
    print(f'Arm manager server started at {ARM_RPC_HOST}:{ARM_RPC_PORT}')
    server.serve_forever()
    # import numpy as np
    # from constants import POLICY_CONTROL_PERIOD
    # manager = ArmManager(address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
    # manager.connect()
    # arm = manager.Arm()
    # try:
    #     arm.reset()
    #     for i in range(50):
    #         arm.execute_action({
    #             'arm_pos': np.array([0.135, 0.002, 0.211]),
    #             'arm_quat': np.array([0.706, 0.707, 0.029, 0.029]),
    #             'gripper_pos': np.zeros(1),
    #         })
    #         print(arm.get_state())
    #         time.sleep(POLICY_CONTROL_PERIOD)  # Note: Not precise
    # finally:
    #     arm.close()
