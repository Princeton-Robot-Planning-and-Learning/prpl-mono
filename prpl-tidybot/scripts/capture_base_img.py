import cv2 as cv
from prpl_tidybot.cameras import LogitechCamera
from prpl_tidybot.constants import BASE_CAMERA_SERIAL

base_camera = LogitechCamera(BASE_CAMERA_SERIAL)
base_image = None
while base_image is None:
    base_image = base_camera.get_image()
cv.imwrite('test_images/base-image.jpg', cv.cvtColor(base_image, cv.COLOR_RGB2BGR))
base_camera.close()
