"""Base class for 6D object pose detectors in RGBD images."""

import abc
from typing import Collection

from prpl_perception_utils.structs import (
    CameraInfo,
    DetectedPose6D,
    ObjectDetectionID,
    RGBDImage,
)


class PoseDetector6D(abc.ABC):
    """Base class for 6D object pose detectors in RGBD images."""

    @abc.abstractmethod
    def detect(
        self,
        rgbds: dict[CameraInfo, RGBDImage],
        object_ids: Collection[ObjectDetectionID],
    ) -> dict[CameraInfo, list[DetectedPose6D]]:
        """Detect zero or one pose for each object ID in each RGBD image.

        Returns a list of detections for each image. The order of the detections in the
        image does not matter -- we just aren't using set() to avoid hashing.
        """
