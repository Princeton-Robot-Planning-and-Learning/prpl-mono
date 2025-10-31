"""Tests for gemini_object_detector_2d.py."""

from pathlib import Path

import imageio.v2 as iio
import pytest
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache

from prpl_perception_utils.object_detection_2d.gemini_object_detector_2d import (
    GeminiObjectDetector2D,
)
from prpl_perception_utils.object_detection_2d.render_wrapper import (
    RenderWrapperObjectDetector2D,
)
from prpl_perception_utils.structs import LanguageObjectDetectionID

runllms = pytest.mark.skipif("not config.getoption('runllms')")
ASSETS_PATH = Path(__file__).parent / "assets"


@runllms
def test_real_gemini_object_detector_2d():
    """Tests object detection with Gemini, for real.

    Add --runllms to your pytest command to run.
    """
    cache = SQLite3PretrainedLargeModelCache(Path(__file__).parent / "_test_cache.db")
    detector = GeminiObjectDetector2D(cache=cache)

    # Wrap the detector to visualize the results.
    detector = RenderWrapperObjectDetector2D(detector, outdir=Path("gemini_unit_test"))

    # Test with simple image that has one airplane.
    airplane_img = iio.imread(ASSETS_PATH / "airplane.jpg")
    airplane_id = LanguageObjectDetectionID("airplane")
    detections = detector.detect([airplane_img], [airplane_id])
    assert len(detections) == 1  # only one RGB image given
    assert len(detections[0]) == 1  # should just detect the airplane
    detection = detections[0][0]
    assert detection.object_id == airplane_id

    # Test with image that has an apple, orange, and banana, and also the first image.
    fruit_img = iio.imread(ASSETS_PATH / "apple-orange-banana.jpg")
    apple_id = LanguageObjectDetectionID("apple")
    orange_id = LanguageObjectDetectionID("orange")
    monkey_id = LanguageObjectDetectionID("monkey")  # should not get detected
    detections = detector.detect(
        [airplane_img, fruit_img], [apple_id, orange_id, monkey_id, airplane_id]
    )
    assert len(detections) == 2  # two image
    assert len(detections[0]) == 1  # should just detect the airplane
    assert len(detections[1]) == 2  # should just detect the apple and orange
    detection_ids = {d.object_id for d in detections[1]}
    assert detection_ids == {apple_id, orange_id}
