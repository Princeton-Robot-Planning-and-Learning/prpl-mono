# PRPL Perception Utils

![workflow](https://github.com/Princeton-Robot-Planning-and-Learning/prpl-perception-utils/actions/workflows/ci.yml/badge.svg)

Perception utilities from the Princeton Robot Planning and Learning group.

## Requirements

- Python 3.10+
- Tested on MacOS Monterey and Ubuntu 22.04

## Installation

1. Recommended: create and source a virtualenv.
2. `pip install -e ".[develop]"`

## Usage

### Object Detection with Gemini

To use the Gemini detector, you'll need a Google API key for Gemini stored in `$GOOGLE_API_KEY`.

```python
from pathlib import Path
import imageio.v2 as iio
from prpl_perception_utils.object_detection_2d.gemini_object_detector_2d import GeminiObjectDetector2D
from prpl_perception_utils.object_detection_2d.render_wrapper import RenderWrapperObjectDetector2D
from prpl_perception_utils.structs import LanguageObjectDetectionID

# Initialize the detector
detector = GeminiObjectDetector2D()

# Optionally wrap with render wrapper for visualization (examine output/ afterwards)
detector = RenderWrapperObjectDetector2D(detector, outdir=Path("output"))

# Load an image
image = iio.imread("tests/object_detection_2d/assets/apple-orange-banana.jpg")

# Define objects to detect
object_ids = [
    LanguageObjectDetectionID("airplane"),
    LanguageObjectDetectionID("apple"),
    LanguageObjectDetectionID("orange"),
]

# Run detection
detections = detector.detect([image], object_ids)

# Process results
for image_detections in detections:
    for detection in image_detections:
        print(f"Detected {detection.object_id} at {detection.bounding_box}")
        print(f"  with {int((detection.mask > 0).sum())} pixels in the mask")
```

Saved to `output/1755970040387329000/0.png`:
<img width="1200" height="798" alt="0" src="https://github.com/user-attachments/assets/c68653ba-0e95-42d8-b416-3e65a1110e69" />


### Pose Detection with Gemini

This is not really pose detection, because we are not detecting orientations, but this will detect 3D positions and can be good enough for some applications.

```python
from prpl_perception_utils.object_detection_2d.gemini_object_detector_2d import GeminiObjectDetector2D
from prpl_perception_utils.pose_detection_6d.simple_2d_pose_detector_6d import Simple2DPoseDetector6D
from prpl_perception_utils.structs import LanguageObjectDetectionID, CameraInfo, RGBD

# Initialize your camera with intrinsics and extrinsics
camera: CameraInfo = ...  # depends on user

# Assume we have some RGBD image, taken by the camera
image: RGBD = ... # depends on user

# Initialize the pose detector
detector = Simple2DPoseDetector6D(GeminiObjectDetector2D())

# Define objects to detect
object_ids = [
    LanguageObjectDetectionID("airplane"),
    LanguageObjectDetectionID("apple"),
    LanguageObjectDetectionID("orange"),
]

# Run detection
detections = detector.detect({camera: image}, object_ids)

# Process results
for image_detections in detections:
    for detection in image_detections:
        print(f"Detected {detection.object_id} at {detection.pose}")
```

## Check Installation

Run `./run_ci_checks.sh`. It should complete with all green successes in 5-10 seconds.
