"""Test for real cameras."""

import cv2 as cv

from prpl_tidybot.cameras import KinovaCamera, LogitechCamera
from prpl_tidybot.constants import BASE_CAMERA_SERIAL


def test_real_cameras() -> None:
    """Test for real cameras.

    Args:
        base_camera: LogitechCamera object
        wrist_camera: KinovaCamera object
    """
    base_camera = LogitechCamera(BASE_CAMERA_SERIAL)  # type: ignore
    wrist_camera = KinovaCamera()  # type: ignore
    base_image = base_camera.get_image()  # type: ignore
    wrist_image = wrist_camera.get_image()  # type: ignore
    cv.imwrite(
        "test_images/base_image.jpg",
        cv.cvtColor(base_image, cv.COLOR_RGB2BGR),
    )
    cv.imwrite(
        "test_images/wrist_image.jpg",
        cv.cvtColor(wrist_image, cv.COLOR_RGB2BGR),
    )
    base_camera.close()  # type: ignore
    wrist_camera.close()  # type: ignore


if __name__ == "__main__":
    test_real_cameras()
