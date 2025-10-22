"""The lowest-level and only direct interface into the real world.

Keep all real-world code in this file so that everything else should be testable without
the real robot.
"""

import abc

import numpy as np
import spatialmath
from prpl_utils.structs import Image

from prpl_tidybot.constants import CAMERA_DIMS, RETRACT_ARM_CONF
from prpl_tidybot.structs import TidyBotObservation


class Interface(abc.ABC):
    """A generic interface, which may be subclassed by real or fake."""

    @abc.abstractmethod
    def get_base_state(self) -> spatialmath.SE2:
        """Get the base pose."""

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
            gripper=self.get_gripper_state(),
            wrist_camera=self.get_wrist_image(),
            base_camera=self.get_base_image(),
        )


# TO BE IMPLEMENTED SOON!
# class RealInterface(Interface):
#     """The real and sole interface to the real robot."""

#     def get_base_state(self) -> spatialmath.SE2:
#         import ipdb; ipdb.set_trace()

#     def get_arm_state(self) -> list[float]:
#         import ipdb; ipdb.set_trace()

#     def get_gripper_state(self) -> float:
#         import ipdb; ipdb.set_trace()

#     def get_wrist_image(self) -> Image:
#         import ipdb; ipdb.set_trace()

#     def get_base_image(self) -> Image:
#         import ipdb; ipdb.set_trace()


class FakeInterface(Interface):
    """A fake interface that can be used for testing without a real robot."""

    def __init__(self):
        self.base_state = spatialmath.SE2(x=0, y=0, theta=0)
        self.arm_state = RETRACT_ARM_CONF
        self.gripper_state = 0.0
        self.wrist_image = np.zeros(CAMERA_DIMS, dtype=np.uint8)
        self.base_image = np.zeros(CAMERA_DIMS, dtype=np.uint8)

    def get_base_state(self) -> spatialmath.SE2:
        return self.base_state.copy()

    def get_arm_state(self) -> list[float]:
        return self.arm_state.copy()

    def get_gripper_state(self) -> float:
        return self.gripper_state

    def get_wrist_image(self) -> Image:
        return self.wrist_image.copy()

    def get_base_image(self) -> Image:
        return self.base_image.copy()
