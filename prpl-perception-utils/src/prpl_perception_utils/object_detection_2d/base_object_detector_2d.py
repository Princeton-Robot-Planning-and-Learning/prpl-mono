"""Base class for 2D object detectors."""

import abc
from typing import Collection

from prpl_perception_utils.structs import DetectedObject2D, ObjectDetectionID, RGBImage


class ObjectDetector2D(abc.ABC):
    """Base class for 2D object detectors."""

    @abc.abstractmethod
    def detect(
        self, rgbs: list[RGBImage], object_ids: Collection[ObjectDetectionID]
    ) -> list[list[DetectedObject2D]]:
        """Detect zero or one instances of each object ID in each RGB image.

        Returns a list of detections for each image. The order of the detections in the
        image does not matter -- we just aren't using set() to avoid hashing.
        """
