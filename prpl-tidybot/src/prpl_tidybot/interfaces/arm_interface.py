"""Arm interface."""

import abc

from prpl_tidybot.arm_server import ArmManager
from prpl_tidybot.constants import (
    ARM_RPC_HOST,
    ARM_RPC_PORT,
    RETRACT_ARM_CONF,
    RPC_AUTHKEY,
)


class ArmInterface(abc.ABC):
    """Arm interface."""

    @abc.abstractmethod
    def get_arm_state(self) -> list[float]:
        """Get the current arm state."""

    @abc.abstractmethod
    def get_gripper_state(self) -> float:
        """Get the current gripper state."""


class FakeArmInterface(ArmInterface):
    """Fake arm interface."""

    def __init__(self):
        self.arm_state = RETRACT_ARM_CONF
        self.gripper_state = 0.0

    def get_arm_state(self) -> list[float]:
        return self.arm_state

    def get_gripper_state(self) -> float:
        return self.gripper_state


class RealArmInterface(ArmInterface):
    """Real arm interface."""

    def __init__(self) -> None:
        self.arm_state = RETRACT_ARM_CONF
        self.gripper_state = 0.0

        self.manager = ArmManager(
            address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY
        )
        self.manager.connect()
        self.arm = self.manager.Arm()  # type: ignore # pylint: disable=no-member
        self.arm.reset()

    def get_arm_state(self) -> list[float]:
        return self.arm.get_joint_angles()

    def get_gripper_state(self) -> float:
        return self.arm.get_gripper_position()

    def execute_action(self, action) -> None:
        """Execute an joint space action on the arm."""
        raise NotImplementedError("Real arm execute_action not implemented yet.")

    def close(self) -> None:
        """Close the arm interface."""
        self.arm.close()
