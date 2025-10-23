import cv2 as cv
from prpl_tidybot.cameras import KinovaCamera

wrist_camera = KinovaCamera()
wrist_image = None
while wrist_image is None:
    wrist_image = wrist_camera.get_image()
cv.imwrite('test_images/wrist-image.jpg', cv.cvtColor(wrist_image, cv.COLOR_RGB2BGR))
wrist_camera.close()
