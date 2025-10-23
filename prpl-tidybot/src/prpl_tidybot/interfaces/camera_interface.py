"""Camera interface."""

import abc

import numpy as np
from prpl_utils.structs import Image

from prpl_tidybot.constants import CAMERA_DIMS


class CameraInterface(abc.ABC):
    """Camera interface."""

    @abc.abstractmethod
    def get_wrist_image(self) -> Image:
        """Get the current wrist image."""

    @abc.abstractmethod
    def get_base_image(self) -> Image:
        """Get the current base image."""


class FakeCameraInterface(CameraInterface):
    """Fake camera interface."""

    def __init__(self):
        self.wrist_image = np.zeros(CAMERA_DIMS, dtype=np.uint8)
        self.base_image = np.zeros(CAMERA_DIMS, dtype=np.uint8)

    def get_wrist_image(self) -> Image:
        return self.wrist_image

    def get_base_image(self) -> Image:
        return self.base_image


class RealCameraInterface(CameraInterface):
    """Real camera interface.

    Coming soon!
    """

    def __init__(self):
        self.wrist_image = None
        self.base_image = None

    def get_wrist_image(self) -> Image:
        raise NotImplementedError("Real camera interface not implemented yet.")

    def get_base_image(self) -> Image:
        raise NotImplementedError("Real camera interface not implemented yet.")
