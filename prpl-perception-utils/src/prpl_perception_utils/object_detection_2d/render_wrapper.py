"""A wrapper for ObjectDetector2D that saves detections to disk."""

import logging
import time
from pathlib import Path
from typing import Collection

import imageio.v2 as iio

from prpl_perception_utils.object_detection_2d.base_object_detector_2d import (
    ObjectDetector2D,
)
from prpl_perception_utils.structs import (
    DetectedObject2D,
    ObjectDetectionID,
    RGBImage,
)
from prpl_perception_utils.utils import visualize_detections_2d


class RenderWrapperObjectDetector2D(ObjectDetector2D):
    """A wrapper for ObjectDetector2D that saves detections to disk."""

    def __init__(
        self,
        base_detector: ObjectDetector2D,
        outdir: Path,
    ) -> None:
        self._base_detector = base_detector
        self._outdir = outdir

    def detect(
        self, rgbs: list[RGBImage], object_ids: Collection[ObjectDetectionID]
    ) -> list[list[DetectedObject2D]]:
        # Make the subdirectory where the annotated images will go.
        subdir = self._outdir / str(time.time_ns())
        subdir.mkdir(exist_ok=True, parents=True)
        # Run detection.
        detections = self._base_detector.detect(rgbs, object_ids)
        # Visualize the detections.
        for i, (rgb, rgb_detection) in enumerate(zip(rgbs, detections, strict=True)):
            img = visualize_detections_2d(rgb, rgb_detection)
            img_file = subdir / f"{i}.png"
            iio.imsave(img_file, img)
        logging.info(f"Saved detections to {subdir}")
        # Return the detections.
        return detections
