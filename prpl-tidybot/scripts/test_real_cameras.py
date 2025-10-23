"""Test for real cameras."""

import cv2 as cv

from prpl_tidybot.cameras import KinovaCamera, LogitechCamera
from prpl_tidybot.constants import BASE_CAMERA_SERIAL


def test_real_cameras():
    """Test for real cameras.

    Args:
        base_camera: LogitechCamera object
        wrist_camera: KinovaCamera object
    """
    base_camera = LogitechCamera(BASE_CAMERA_SERIAL)
    wrist_camera = KinovaCamera()
    base_image = base_camera.get_image()
    wrist_image = wrist_camera.get_image()
    cv.imwrite("test_images/base_image.jpg", cv.cvtColor(base_image, cv.COLOR_RGB2BGR))
    cv.imwrite(
        "test_images/wrist_image.jpg", cv.cvtColor(wrist_image, cv.COLOR_RGB2BGR)
    )
    base_camera.close()
    wrist_camera.close()


if __name__ == "__main__":
    test_real_cameras()
