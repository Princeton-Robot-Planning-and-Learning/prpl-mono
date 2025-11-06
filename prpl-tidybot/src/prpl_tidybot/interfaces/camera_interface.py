"""Camera interface."""

import abc

import numpy as np
from prpl_utils.structs import Image

from prpl_tidybot.cameras import KinovaCamera, LogitechCamera
from prpl_tidybot.constants import (
    BASE_CAMERA_DIMS,
    BASE_CAMERA_SERIAL,
    WRIST_CAMERA_DIMS,
)


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
        self.wrist_image = np.zeros(WRIST_CAMERA_DIMS, dtype=np.uint8)
        self.base_image = np.zeros(BASE_CAMERA_DIMS, dtype=np.uint8)

    def get_wrist_image(self) -> Image:
        return self.wrist_image

    def get_base_image(self) -> Image:
        return self.base_image


class RealCameraInterface(CameraInterface):
    """Real camera interface.

    wrist camera: kinova camera
    base camera: logitech camera
    """

    def __init__(self) -> None:
        self.base_camera = LogitechCamera(BASE_CAMERA_SERIAL)  # type: ignore
        self.wrist_camera = KinovaCamera()  # type: ignore

    def get_wrist_image(self) -> Image:
        return self.wrist_camera.get_image()  # type: ignore

    def get_base_image(self) -> Image:
        return self.base_camera.get_image()  # type: ignore

    def close(self) -> None:
        """Close the camera interface."""
        self.base_camera.close()  # type: ignore
        self.wrist_camera.close()  # type: ignore
