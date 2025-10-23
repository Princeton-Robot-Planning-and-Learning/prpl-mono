"""Arm interface."""

import abc

from prpl_tidybot.constants import RETRACT_ARM_CONF


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
    """Real arm interface.

    Coming soon!
    """

    def __init__(self):
        self.arm_state = RETRACT_ARM_CONF
        self.gripper_state = 0.0

    def get_arm_state(self) -> list[float]:
        raise NotImplementedError("Real arm interface not implemented yet.")

    def get_gripper_state(self) -> float:
        raise NotImplementedError("Real arm interface not implemented yet.")
