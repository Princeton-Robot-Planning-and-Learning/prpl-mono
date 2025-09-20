"""Script to test 2D object detection with Gemini.

Example usage:
    python scripts/gemini_object_detection_2d.py \
        --labels apple orange monkey \
        --input tests/object_detection_2d/assets/apple-orange-banana.jpg \
        --output fruit_detections.png
"""

from pathlib import Path

import imageio.v2 as iio

from prpl_perception_utils.object_detection_2d.gemini_object_detector_2d import (
    GeminiObjectDetector2D,
)
from prpl_perception_utils.structs import LanguageObjectDetectionID
from prpl_perception_utils.utils import visualize_detections_2d


def _main(
    object_labels: list[str], input_img_path: Path, output_img_path: Path
) -> None:
    detector = GeminiObjectDetector2D()
    img = iio.imread(input_img_path)
    obj_ids = [LanguageObjectDetectionID(l) for l in object_labels]
    detections = detector.detect([img], obj_ids)
    assert len(detections) == 1  # only one RGB image given
    out_img = visualize_detections_2d(img, detections[0])
    iio.imsave(output_img_path, out_img)
    print(f"Wrote out to {output_img_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test 2D object detection with Gemini")
    parser.add_argument(
        "--labels", nargs="+", required=True, help="Object labels to detect"
    )
    parser.add_argument("--input", type=Path, required=True, help="Input image path")
    parser.add_argument("--output", type=Path, required=True, help="Output image path")

    args = parser.parse_args()

    _main(args.labels, args.input, args.output)
