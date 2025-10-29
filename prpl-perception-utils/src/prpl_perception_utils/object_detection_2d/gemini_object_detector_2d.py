"""Uses a Gemini model for 2D object detection."""

import base64
import io
import json
import logging
import tempfile
from pathlib import Path
from string import Template
from typing import Collection

import numpy as np
from PIL import Image
from prpl_llm_utils.cache import (
    PretrainedLargeModelCache,
    SQLite3PretrainedLargeModelCache,
)
from prpl_llm_utils.models import GeminiModel
from prpl_llm_utils.reprompting import (
    FunctionalRepromptCheck,
    create_reprompt_from_error_message,
    query_with_reprompts,
)
from prpl_llm_utils.structs import Query, Response

from prpl_perception_utils.object_detection_2d.base_object_detector_2d import (
    ObjectDetector2D,
)
from prpl_perception_utils.structs import (
    BoundingBox,
    DetectedObject2D,
    LanguageObjectDetectionID,
    ObjectDetectionID,
    RGBImage,
)

_PROMPT = Template(
    """
Give the segmentation masks for the following objects if they appear to be
present in the image:

$joined_object_descriptions

Output a JSON list where each entry contains:
{
    "box_2d": the 2D bounding box of the object.
    "mask": a complete pixel-level segmentation tightly following the
        visible boundaries of the object (take note of figure-ground effects
        to keep track of the boundaries of the figure), ensuring the entire
        object is included, even if elongated or partially occluded.
    "label": the object label (use the same labels as listed above).
    "confidence": a calibrated confidence score between 0.1 and 1 indicating how
        likely it is that the object is correctly detected in that location. Set
        confidence according to how much you would be willing to bet that the
        object you masked is the correct one, with the correct COLOR, SHAPE, and
        other distinctive features. Never mask an image with confidence <0.1,
        and be confident if you know you masked the right object.
}
"""
)


def check_detection(query: Query, response: Response) -> Query | None:
    """
    Validate that the response `r` is a JSON list of dicts with the required fields:
      - "box_2d"
      - "mask"
      - "label"
      - "confidence" in [0, 1]
    Returns None if valid, otherwise a reprompt message.
    """

    try:
        data = json.loads(_parse_json(response.text))
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        return create_reprompt_from_error_message(
            query, response, f"JSON parsing error: {e}"
        )

    if not isinstance(data, list):
        return create_reprompt_from_error_message(query, response, "Not a JSON list.")

    required_keys = {"box_2d", "mask", "label", "confidence"}

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            return create_reprompt_from_error_message(
                query, response, f"Entry {i}: Not a valid JSON object."
            )

        missing = required_keys - set(item.keys())
        if missing:
            return create_reprompt_from_error_message(
                query, response, f"Entry {i}: Missing keys {', '.join(missing)}."
            )

        if not isinstance(item["box_2d"], (list, dict)):
            return create_reprompt_from_error_message(
                query, response, f"Entry {i}: Invalid format for box_2d."
            )

        if not isinstance(item["mask"], (str, list, dict)):
            return create_reprompt_from_error_message(
                query, response, f"Entry {i}: Invalid format for mask."
            )

        if not isinstance(item["label"], str):
            return create_reprompt_from_error_message(
                query, response, f"Entry {i}: Invalid format for label."
            )

        conf = item["confidence"]
        if not (isinstance(conf, (int, float)) and 0 <= conf <= 1):
            return create_reprompt_from_error_message(
                query,
                response,
                f"Entry {i}: 'confidence' must be a number between 0 and 1.",
            )

    # Valid response format (no reprompt)
    return None


def _parse_json(json_output: str) -> str:
    # Parsing out the markdown fencing, keeping only the first one.
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i + 1 :])
            json_output = json_output.split("```", maxsplit=1)[0]
            break
    return json_output


class GeminiObjectDetector2D(ObjectDetector2D):
    """Uses a Gemini model for 2D object detection."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        thumbnail_size: int = 1024,
        min_mask_value: int = 100,
        cache: PretrainedLargeModelCache | None = None,
    ) -> None:

        if cache is None:
            logging.info("Cache does not exist, a default one will be created")
            cache_path = Path(tempfile.gettempdir()) / "detection.db"
            cache = SQLite3PretrainedLargeModelCache(cache_path)

        self._gemini_model = GeminiModel(model, cache)
        self._thumbnail_size = thumbnail_size
        self._min_mask_value = min_mask_value

    def detect(
        self, rgbs: list[RGBImage], object_ids: Collection[ObjectDetectionID]
    ) -> list[list[DetectedObject2D]]:
        # Finalize prompt.
        object_descriptions: list[str] = []
        for object_id in object_ids:
            assert isinstance(object_id, LanguageObjectDetectionID)
            object_descriptions.append(object_id.language_id)
        joined_object_descriptions = "\n".join(object_descriptions)
        prompt = _PROMPT.substitute(
            joined_object_descriptions=joined_object_descriptions
        )
        object_description_to_object_id = dict(zip(object_descriptions, object_ids))
        # NOTE: should eventually figure out batching over images.
        detections: list[list[DetectedObject2D]] = []
        for rgb in rgbs:
            # Convert RGB to PIL and resize.
            im = Image.fromarray(rgb)
            im.thumbnail(
                (self._thumbnail_size, self._thumbnail_size), Image.Resampling.LANCZOS
            )
            # Run Gemini query.
            # NOTE: validates and reprompts, according to check_fn.
            logging.info("Sending query to Gemini")
            query = Query(prompt, [im], {"temperature": 0.0})
            checker = FunctionalRepromptCheck(check_detection)
            response = query_with_reprompts(self._gemini_model, query, [checker])
            logging.info("Received response from Gemini")
            assert response.text is not None
            json_str = _parse_json(response.text)
            try:
                items = json.loads(json_str)
            except json.decoder.JSONDecodeError:
                # This can happen for any number of reasons, including when no object
                # is detected.
                items = []
            # Convert to DetectedObject2D.
            rgb_detections: list[DetectedObject2D] = []
            for item in items:

                # Extract the object_id.
                label = item["label"]
                if label not in object_description_to_object_id:
                    continue
                object_id = object_description_to_object_id[label]

                # Get bounding box coordinates.
                box = item["box_2d"]
                norm_y1 = int(box[0] / 1000 * im.size[1])
                norm_x1 = int(box[1] / 1000 * im.size[0])
                norm_y2 = int(box[2] / 1000 * im.size[1])
                norm_x2 = int(box[3] / 1000 * im.size[0])

                # Skip invalid boxes.
                if norm_y1 >= norm_y2 or norm_x1 >= norm_x2:
                    continue

                # Rescale the bounding box back to the original image size.
                x_scale = rgb.shape[1] / im.size[0]
                y_scale = rgb.shape[0] / im.size[1]
                x1 = int(norm_x1 * x_scale)
                x2 = int(norm_x2 * x_scale)
                y1 = int(norm_y1 * y_scale)
                y2 = int(norm_y2 * y_scale)

                # Finalize the bounding box.
                bounding_box = BoundingBox(x1, y1, x2, y2)

                # Process mask.
                png_str = item["mask"]
                if not png_str.startswith("data:image/png;base64,"):
                    continue

                # Remove prefix.
                png_str = png_str.removeprefix("data:image/png;base64,")
                mask_data = base64.b64decode(png_str)
                mask: Image.Image = Image.open(io.BytesIO(mask_data))

                # Resize mask to match bounding box.
                mask = mask.resize(
                    (bounding_box.width, bounding_box.height), Image.Resampling.BILINEAR
                )

                # Convert mask to numpy array.
                mask_array = np.array(mask) > self._min_mask_value

                # Extract the confidence
                confidence_score = item["confidence"]

                # Finish the object detection.
                detection = DetectedObject2D(
                    object_id, bounding_box, mask_array, confidence_score
                )
                rgb_detections.append(detection)

            detections.append(rgb_detections)

        return detections
