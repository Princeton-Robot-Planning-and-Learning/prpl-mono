# Author: Jimmy Wu
# Date: October 2024

# mypy: ignore-errors
# pylint: disable=all

import threading
import time

import cv2 as cv
import numpy as np
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.client_stubs.VisionConfigClientRpc import VisionConfigClient
from kortex_api.autogen.messages import DeviceConfig_pb2, VisionConfig_pb2

from prpl_tidybot.constants import BASE_CAMERA_SERIAL
from prpl_tidybot.kinova import DeviceConnection


class Camera:
    def __init__(self):
        self.image = None
        self.last_read_time = time.time()
        threading.Thread(target=self.camera_worker, daemon=True).start()

    def camera_worker(self):
        # Note: We read frames at 30 fps but not every frame is necessarily
        # saved during teleop or used during policy inference
        while True:
            # Reading new frames too quickly causes latency spikes
            while time.time() - self.last_read_time < 0.0333:  # 30 fps
                time.sleep(0.0001)
            _, bgr_image = self.cap.read()
            self.last_read_time = time.time()
            if bgr_image is not None:
                self.image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)

    def get_image(self):
        return self.image

    def close(self):
        self.cap.release()


class LogitechCamera(Camera):
    def __init__(self, serial, frame_width=640, frame_height=360, focus=0):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.focus = focus  # Note: Set this to 100 when using fisheye lens attachment
        self.cap = self.get_cap(serial)
        super().__init__()

    def get_cap(self, serial):
        cap = cv.VideoCapture(
            f"/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_{serial}-video-index0"
        )
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc("M", "J", "P", "G"))
        cap.set(cv.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)  # Important - results in much better latency

        # Disable autofocus
        cap.set(cv.CAP_PROP_AUTOFOCUS, 0)

        # Read several frames to let settings (especially gain/exposure) stabilize
        for _ in range(30):
            cap.read()
            cap.set(cv.CAP_PROP_FOCUS, self.focus)  # Fixed focus

        # Check all settings match expected
        assert cap.get(cv.CAP_PROP_FRAME_WIDTH) == self.frame_width
        assert cap.get(cv.CAP_PROP_FRAME_HEIGHT) == self.frame_height
        assert cap.get(cv.CAP_PROP_BUFFERSIZE) == 1
        assert cap.get(cv.CAP_PROP_AUTOFOCUS) == 0
        assert cap.get(cv.CAP_PROP_FOCUS) == self.focus

        return cap


def find_fisheye_center(image):
    # Find contours
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    # Fit a minimum enclosing circle around all contours
    return cv.minEnclosingCircle(np.vstack(contours))


def check_fisheye_centered(image):
    height, width, _ = image.shape
    center, _ = find_fisheye_center(image)
    if center is None:
        return True
    return (
        abs(width / 2 - center[0]) < 0.05 * width
        and abs(height / 2 - center[1]) < 0.05 * height
    )


class KinovaCamera(Camera):
    def __init__(self):
        # GStreamer video capture (see https://github.com/Kinovarobotics/kortex/issues/88)
        # Note: max-buffers=1 and drop=true are added to reduce latency spikes
        self.cap = cv.VideoCapture(
            "rtspsrc location=rtsp://192.168.1.10/color latency=0 ! decodebin ! videoconvert ! appsink sync=false max-buffers=1 drop=true",
            cv.CAP_GSTREAMER,
        )
        # self.cap = cv.VideoCapture('rtsp://192.168.1.10/color', cv.CAP_FFMPEG)  # This stream is high latency but works with pip-installed OpenCV
        assert (
            self.cap.isOpened()
        ), "Unable to open stream. Please make sure OpenCV was built from source with GStreamer support."

        # Apply camera settings
        threading.Thread(target=self.apply_camera_settings, daemon=True).start()
        super().__init__()

        # Wait for camera to warm up
        image = None
        while image is None:
            image = self.get_image()

        # Make sure fisheye lens did not accidentally get bumped
        if not check_fisheye_centered(image):
            raise Exception(
                "The fisheye lens on the Kinova wrist camera appears to be off-center"
            )

    def apply_camera_settings(self):
        # Note: This function adds significant camera latency when it is called
        # directly in __init__, so we call it in a separate thread instead

        # Use Kortex API to set camera settings
        with DeviceConnection.createTcpConnection() as router:
            device_manager = DeviceManagerClient(router)
            vision_config = VisionConfigClient(router)

            # Get vision device ID
            device_handles = device_manager.ReadAllDevices()
            vision_device_ids = [
                handle.device_identifier
                for handle in device_handles.device_handle
                if handle.device_type == DeviceConfig_pb2.VISION
            ]
            assert len(vision_device_ids) == 1
            vision_device_id = vision_device_ids[0]

            # Check that resolution, frame rate, and bit rate are correct
            sensor_id = VisionConfig_pb2.SensorIdentifier()
            sensor_id.sensor = VisionConfig_pb2.SENSOR_COLOR
            sensor_settings = vision_config.GetSensorSettings(
                sensor_id, vision_device_id
            )
            try:
                assert (
                    sensor_settings.resolution == VisionConfig_pb2.RESOLUTION_640x480
                )  # FOV 65 ± 3° (diagonal)
                assert sensor_settings.frame_rate == VisionConfig_pb2.FRAMERATE_30_FPS
                assert sensor_settings.bit_rate == VisionConfig_pb2.BITRATE_10_MBPS
            except:
                sensor_settings.sensor = VisionConfig_pb2.SENSOR_COLOR
                sensor_settings.resolution = VisionConfig_pb2.RESOLUTION_640x480
                sensor_settings.frame_rate = VisionConfig_pb2.FRAMERATE_30_FPS
                sensor_settings.bit_rate = VisionConfig_pb2.BITRATE_10_MBPS
                vision_config.SetSensorSettings(sensor_settings, vision_device_id)
                assert (
                    False
                ), "Incorrect Kinova camera sensor settings detected, please restart the camera to apply new settings"

            # Disable autofocus and set manual focus to infinity
            # Note: This must be called after the OpenCV stream is created,
            # otherwise the camera will still have autofocus enabled
            sensor_focus_action = VisionConfig_pb2.SensorFocusAction()
            sensor_focus_action.sensor = VisionConfig_pb2.SENSOR_COLOR
            sensor_focus_action.focus_action = (
                VisionConfig_pb2.FOCUSACTION_SET_MANUAL_FOCUS
            )
            sensor_focus_action.manual_focus.value = 0
            vision_config.DoSensorFocusAction(sensor_focus_action, vision_device_id)


if __name__ == "__main__":
    base_camera = LogitechCamera(BASE_CAMERA_SERIAL)
    wrist_camera = KinovaCamera()
    try:
        while True:
            base_image = base_camera.get_image()
            wrist_image = wrist_camera.get_image()
            cv.imshow("base_image", cv.cvtColor(base_image, cv.COLOR_RGB2BGR))
            cv.imshow("wrist_image", cv.cvtColor(wrist_image, cv.COLOR_RGB2BGR))
            key = cv.waitKey(1)
            if key == ord("s"):  # Save image
                base_image_path = f"base-image-{int(10 * time.time()) % 100000000}.jpg"
                cv.imwrite(base_image_path, cv.cvtColor(base_image, cv.COLOR_RGB2BGR))
                print(f"Saved image to {base_image_path}")
                wrist_image_path = (
                    f"wrist-image-{int(10 * time.time()) % 100000000}.jpg"
                )
                cv.imwrite(wrist_image_path, cv.cvtColor(wrist_image, cv.COLOR_RGB2BGR))
                print(f"Saved image to {wrist_image_path}")
    finally:
        base_camera.close()
        wrist_camera.close()
        cv.destroyAllWindows()
