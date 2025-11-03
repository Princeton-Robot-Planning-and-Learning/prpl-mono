"""The lowest-level and only direct interface into the real world.

Keep all real-world code in this file so that everything else should be testable without
the real robot.
"""

import abc

import spatialmath
from prpl_utils.structs import Image

from prpl_tidybot.interfaces.arm_interface import FakeArmInterface, RealArmInterface
from prpl_tidybot.interfaces.base_interface import FakeBaseInterface, RealBaseInterface
from prpl_tidybot.interfaces.camera_interface import (
    FakeCameraInterface,
    RealCameraInterface,
)
from prpl_tidybot.structs import TidyBotAction, TidyBotObservation


class Interface(abc.ABC):
    """A generic interface, which may be subclassed by real or fake."""

    @abc.abstractmethod
    def get_base_state(self) -> spatialmath.SE2:
        """Get the base pose."""

    @abc.abstractmethod
    def execute_base_action(self, action: TidyBotAction) -> None:
        """Execute a base action in the local frame."""

    @abc.abstractmethod
    def get_map_base_state(self) -> spatialmath.SE2:
        """Get the base pose in the map frame."""

    @abc.abstractmethod
    def get_arm_state(self) -> list[float]:
        """Get the 7-DOF joint positions."""

    @abc.abstractmethod
    def get_gripper_state(self) -> float:
        """Get gripper state (1 is open, 0 is closed)."""

    @abc.abstractmethod
    def get_wrist_image(self) -> Image:
        """Get the current wrist image."""

    @abc.abstractmethod
    def get_base_image(self) -> Image:
        """Get the current base image."""

    def get_observation(self) -> TidyBotObservation:
        """Construct a TidyBotObservation()."""
        return TidyBotObservation(
            arm_conf=self.get_arm_state(),
            base_pose=self.get_base_state(),
            map_base_pose=self.get_map_base_state(),
            gripper=self.get_gripper_state(),
            wrist_camera=self.get_wrist_image(),
            base_camera=self.get_base_image(),
        )


class RealInterface(Interface):
    """The real and sole interface to the real robot."""

    def __init__(self) -> None:
        self.camera_interface = RealCameraInterface()
        self.base_interface = RealBaseInterface()
        self.arm_interface = RealArmInterface()

    def get_base_state(self) -> spatialmath.SE2:
        return self.base_interface.get_base_state()

    def execute_base_action(self, action: TidyBotAction) -> None:
        return self.base_interface.execute_action(action.base_local_goal)

    def get_map_base_state(self) -> spatialmath.SE2:
        return self.base_interface.get_map_base_state()

    def get_arm_state(self) -> list[float]:
        return self.arm_interface.get_arm_state()

    def get_gripper_state(self) -> float:
        return self.arm_interface.get_gripper_state()

    def get_wrist_image(self) -> Image:
        return self.camera_interface.get_wrist_image()

    def get_base_image(self) -> Image:
        return self.camera_interface.get_base_image()

    def close(self) -> None:
        """Close the real interface."""
        self.base_interface.close()
        self.arm_interface.close()
        self.camera_interface.close()


class FakeInterface(Interface):
    """A fake interface that can be used for testing without a real robot."""

    def __init__(self):
        self.arm_interface = FakeArmInterface()
        self.base_interface = FakeBaseInterface()
        self.camera_interface = FakeCameraInterface()

    def get_base_state(self) -> spatialmath.SE2:
        return self.base_interface.get_base_state()

    def execute_base_action(self, action: TidyBotAction) -> None:
        return self.base_interface.execute_action(action.base_local_goal)

    def get_map_base_state(self) -> spatialmath.SE2:
        return self.base_interface.get_map_base_state()

    def get_arm_state(self) -> list[float]:
        return self.arm_interface.get_arm_state()

    def get_gripper_state(self) -> float:
        return self.arm_interface.get_gripper_state()

    def get_wrist_image(self) -> Image:
        return self.camera_interface.get_wrist_image()

    def get_base_image(self) -> Image:
        return self.camera_interface.get_base_image()
