"""Tests for simple_2d_pose_detector_6d.py."""

import json
from pathlib import Path
from unittest.mock import Mock

import imageio.v2 as iio
import numpy as np
import pytest
from spatialmath import SE3

from prpl_perception_utils.object_detection_2d.base_object_detector_2d import (
    ObjectDetector2D,
)
from prpl_perception_utils.object_detection_2d.gemini_object_detector_2d import (
    GeminiObjectDetector2D,
)
from prpl_perception_utils.object_detection_2d.render_wrapper import (
    RenderWrapperObjectDetector2D,
)
from prpl_perception_utils.pose_detection_6d.simple_2d_pose_detector_6d import (
    Simple2DPoseDetector6D,
)
from prpl_perception_utils.structs import (
    BoundingBox,
    CameraInfo,
    DetectedObject2D,
    DetectedPose6D,
    LanguageObjectDetectionID,
    RGBDImage,
)

runllms = pytest.mark.skipif("not config.getoption('runllms')")
ASSETS_PATH = Path(__file__).parent / "assets" / "tyo-l"


class MockObjectDetector2D(ObjectDetector2D):
    """Mock object detector for testing."""

    def __init__(self, detections_to_return):
        self.detections_to_return = detections_to_return

    def detect(self, rgbs, object_ids):
        return self.detections_to_return


def test_simple_2d_pose_detector_6d_creation():
    """Test creating a Simple2DPoseDetector6D."""
    # pylint: disable=protected-access
    mock_detector = Mock(spec=ObjectDetector2D)
    detector = Simple2DPoseDetector6D(mock_detector)
    assert detector._object_detector == mock_detector
    assert detector._min_depth_value == 2  # default value


def test_detect_empty_input():
    """Test detect method with empty input."""
    mock_detector = MockObjectDetector2D([])
    detector = Simple2DPoseDetector6D(mock_detector)

    result = detector.detect({}, [])
    assert not result


def test_detect_no_detections():
    """Test detect method when 2D detector finds no objects."""
    mock_detector = MockObjectDetector2D([[]])  # Empty detections for one image
    detector = Simple2DPoseDetector6D(mock_detector)

    # Create test data
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    depth = np.ones((100, 100), dtype=np.uint8) * 100
    rgbd = RGBDImage(rgb, depth)
    camera = CameraInfo(
        name="test_camera",
        focal_length=(50.0, 50.0),
        principal_point=(50.0, 50.0),
        depth_scale=1000.0,
        world_tform_camera=SE3(),
    )
    object_ids = [LanguageObjectDetectionID("test_object")]

    result = detector.detect({camera: rgbd}, object_ids)
    assert len(result) == 1
    assert camera in result
    assert len(result[camera]) == 0


def test_detect_successful_detection():
    """Test detect method with successful object detection and pose estimation."""
    # Create a test detection
    obj_id = LanguageObjectDetectionID("test_object")
    bbox = BoundingBox(25, 25, 75, 75)
    mask = np.ones((50, 50), dtype=np.bool_)
    detection = DetectedObject2D(obj_id, bbox, mask, 0.9)

    mock_detector = MockObjectDetector2D([[detection]])
    detector = Simple2DPoseDetector6D(mock_detector, min_depth_value=1)

    # Create test RGBD with meaningful depth values
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    depth = np.ones((100, 100), dtype=np.uint8) * 100  # depth value of 100
    rgbd = RGBDImage(rgb, depth)

    camera = CameraInfo(
        name="test_camera",
        focal_length=(50.0, 50.0),
        principal_point=(50.0, 50.0),
        depth_scale=1000.0,
        world_tform_camera=SE3(),
    )
    object_ids = [obj_id]

    result = detector.detect({camera: rgbd}, object_ids)

    assert len(result) == 1
    assert camera in result
    assert len(result[camera]) == 1

    detected_pose = result[camera][0]
    assert isinstance(detected_pose, DetectedPose6D)
    assert detected_pose.object_id == obj_id
    assert isinstance(detected_pose.pose, SE3)


def test_detect_multiple_cameras():
    """Test detect method with multiple cameras."""
    obj_id = LanguageObjectDetectionID("test_object")
    bbox = BoundingBox(25, 25, 75, 75)
    mask = np.ones((50, 50), dtype=np.bool_)
    detection = DetectedObject2D(obj_id, bbox, mask, 0.9)

    mock_detector = MockObjectDetector2D([[detection], [detection]])
    detector = Simple2DPoseDetector6D(mock_detector, min_depth_value=1)

    # Create test data for two cameras
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    depth = np.ones((100, 100), dtype=np.uint8) * 100
    rgbd = RGBDImage(rgb, depth)

    camera1 = CameraInfo(
        name="camera1",
        focal_length=(50.0, 50.0),
        principal_point=(50.0, 50.0),
        depth_scale=1000.0,
        world_tform_camera=SE3(),
    )
    camera2 = CameraInfo(
        name="camera2",
        focal_length=(60.0, 60.0),
        principal_point=(60.0, 60.0),
        depth_scale=1000.0,
        world_tform_camera=SE3.Trans(1, 0, 0),  # Translated camera
    )

    rgbds = {camera1: rgbd, camera2: rgbd}
    object_ids = [obj_id]

    result = detector.detect(rgbds, object_ids)

    assert len(result) == 2
    assert camera1 in result
    assert camera2 in result
    assert len(result[camera1]) == 1
    assert len(result[camera2]) == 1


def test_object_detection_to_pose_valid_depth():
    """Test _object_detection_to_pose with valid depth values."""
    # pylint: disable=protected-access
    mock_detector = Mock(spec=ObjectDetector2D)
    detector = Simple2DPoseDetector6D(mock_detector, min_depth_value=1)

    obj_id = LanguageObjectDetectionID("test_object")
    bbox = BoundingBox(40, 40, 60, 60)  # 20x20 box centered at (50, 50)
    mask = np.ones((20, 20), dtype=np.bool_)
    detection = DetectedObject2D(obj_id, bbox, mask, 0.9)

    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    depth = np.ones((100, 100), dtype=np.uint8) * 100
    rgbd = RGBDImage(rgb, depth)

    camera = CameraInfo(
        name="test_camera",
        focal_length=(50.0, 50.0),
        principal_point=(50.0, 50.0),
        depth_scale=1000.0,
        world_tform_camera=SE3(),
    )

    result = detector._object_detection_to_pose(detection, rgbd, camera)

    assert result is not None
    assert isinstance(result, SE3)

    # Check that the pose is at the expected position
    # Object center is at (50, 50) in image coordinates
    # With principal point at (50, 50), this should be at origin in camera frame
    # With depth of 100 and depth_scale of 1000, z should be 0.1
    expected_z = 100 / 1000.0  # 0.1
    expected_x = 0.0  # Center aligned with principal point
    expected_y = 0.0  # Center aligned with principal point

    translation = result.t
    assert abs(translation[0] - expected_x) < 1e-6
    assert abs(translation[1] - expected_y) < 1e-6
    assert abs(translation[2] - expected_z) < 1e-6


def test_object_detection_to_pose_with_camera_transform():
    """Test _object_detection_to_pose with non-identity camera transform."""
    # pylint: disable=protected-access
    mock_detector = Mock(spec=ObjectDetector2D)
    detector = Simple2DPoseDetector6D(mock_detector, min_depth_value=1)

    obj_id = LanguageObjectDetectionID("test_object")
    bbox = BoundingBox(40, 40, 60, 60)
    mask = np.ones((20, 20), dtype=np.bool_)
    detection = DetectedObject2D(obj_id, bbox, mask, 0.9)

    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    depth = np.ones((100, 100), dtype=np.uint8) * 100
    rgbd = RGBDImage(rgb, depth)

    # Camera translated by (1, 2, 3) in world frame
    camera_transform = SE3.Trans(1, 2, 3)
    camera = CameraInfo(
        name="test_camera",
        focal_length=(50.0, 50.0),
        principal_point=(50.0, 50.0),
        depth_scale=1000.0,
        world_tform_camera=camera_transform,
    )

    result = detector._object_detection_to_pose(detection, rgbd, camera)

    assert result is not None
    assert isinstance(result, SE3)

    # The world frame pose should be the camera transform applied to camera frame pose
    camera_frame_pose = SE3((0.0, 0.0, 0.1))  # Object at camera center, depth 0.1
    expected_world_pose = camera_transform * camera_frame_pose

    translation = result.t
    expected_translation = expected_world_pose.t

    assert abs(translation[0] - expected_translation[0]) < 1e-6
    assert abs(translation[1] - expected_translation[1]) < 1e-6
    assert abs(translation[2] - expected_translation[2]) < 1e-6


def test_object_detection_to_pose_off_center():
    """Test _object_detection_to_pose with object not at image center."""
    # pylint: disable=protected-access
    mock_detector = Mock(spec=ObjectDetector2D)
    detector = Simple2DPoseDetector6D(mock_detector, min_depth_value=1)

    obj_id = LanguageObjectDetectionID("test_object")
    # Object at (75, 25) center, which is offset from principal point (50, 50)
    bbox = BoundingBox(65, 15, 85, 35)  # 20x20 box
    mask = np.ones((20, 20), dtype=np.bool_)
    detection = DetectedObject2D(obj_id, bbox, mask, 0.9)

    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    depth = np.ones((100, 100), dtype=np.uint8) * 100
    rgbd = RGBDImage(rgb, depth)

    camera = CameraInfo(
        name="test_camera",
        focal_length=(50.0, 50.0),
        principal_point=(50.0, 50.0),
        depth_scale=1000.0,
        world_tform_camera=SE3(),
    )

    result = detector._object_detection_to_pose(detection, rgbd, camera)

    assert result is not None
    assert isinstance(result, SE3)

    # Calculate expected position
    # Object center: (75, 25)
    # Principal point: (50, 50)
    # Focal length: (50, 50)
    # Depth: 100, scale: 1000, so camera_z = 0.1
    expected_camera_x = 0.1 * (75 - 50) / 50  # 0.1 * 25 / 50 = 0.05
    expected_camera_y = 0.1 * (25 - 50) / 50  # 0.1 * -25 / 50 = -0.05
    expected_camera_z = 0.1

    translation = result.t
    assert abs(translation[0] - expected_camera_x) < 1e-6
    assert abs(translation[1] - expected_camera_y) < 1e-6
    assert abs(translation[2] - expected_camera_z) < 1e-6


def test_simple_2d_pose_detector_6d_with_real_data():
    """Test Simple2DPoseDetector6D with real tyo-l dataset."""
    # Load the real data
    with open(ASSETS_PATH / "scene_camera.json", encoding="utf-8") as f:
        camera_data = json.load(f)
    with open(ASSETS_PATH / "scene_gt.json", encoding="utf-8") as f:
        gt_data = json.load(f)
    with open(ASSETS_PATH / "scene_gt_info.json", encoding="utf-8") as f:
        gt_info_data = json.load(f)

    # Use image ID "1" which corresponds to the available files
    image_id = "1"

    # Extract camera parameters for this image
    cam_info = camera_data[image_id]
    cam_k = cam_info["cam_K"]  # 3x3 matrix flattened
    depth_scale = cam_info["depth_scale"]

    # Camera intrinsics: K matrix is [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    fx, fy = cam_k[0], cam_k[4]
    cx, cy = cam_k[2], cam_k[5]

    # Extract ground truth pose for object ID 1
    gt_pose = gt_data[image_id][0]  # First (and only) object in this image
    assert gt_pose["obj_id"] == 1

    # Ground truth rotation and translation
    gt_rotation = np.array(gt_pose["cam_R_m2c"]).reshape(3, 3)
    gt_translation = np.array(gt_pose["cam_t_m2c"])
    expected_pose = SE3.Rt(gt_rotation, gt_translation, check=False)

    # Extract bounding box info
    bbox_info = gt_info_data[image_id][0]
    bbox = bbox_info["bbox_obj"]  # [x, y, width, height]
    x, y, width, height = bbox

    # Load the real images
    rgb_img = iio.imread(ASSETS_PATH / "rgb" / "000001.png")
    depth_img = iio.imread(ASSETS_PATH / "depth" / "000001.png")
    mask_img = iio.imread(ASSETS_PATH / "mask" / "000001_000000.png")

    # Convert mask to bounding box coordinates
    # The mask image is the same size as RGB/depth, but we need the mask over
    # the bounding box
    mask_bbox = mask_img[y : y + height, x : x + width]
    # Convert to boolean mask
    mask_bool = mask_bbox > 0

    # Create the structures
    obj_id = LanguageObjectDetectionID("tyo-l_object_1")
    bounding_box = BoundingBox(x, y, x + width, y + height)
    detection = DetectedObject2D(obj_id, bounding_box, mask_bool.astype(np.uint8))

    # Create RGBD image
    rgbd = RGBDImage(rgb_img, depth_img)

    # Create camera info (assuming identity world transform for simplicity)
    camera = CameraInfo(
        name="tyo-l_camera",
        focal_length=(fx, fy),
        principal_point=(cx, cy),
        depth_scale=depth_scale,
        world_tform_camera=SE3(),  # Identity - poses will be in camera frame
    )

    # Create mock detector that returns our real detection
    mock_detector = MockObjectDetector2D([[detection]])
    detector = Simple2DPoseDetector6D(mock_detector, min_depth_value=1)

    # Run the detection
    result = detector.detect({camera: rgbd}, [obj_id])

    # Verify we got a result
    assert len(result) == 1
    assert camera in result
    assert len(result[camera]) == 1

    detected_pose = result[camera][0]
    assert isinstance(detected_pose, DetectedPose6D)
    assert detected_pose.object_id == obj_id

    # The detected pose should be reasonably close to ground truth
    # Note: Simple2DPoseDetector6D only estimates position, not full 6D pose
    # and uses median depth, so we expect some difference
    detected_translation = detected_pose.pose.t
    expected_translation = expected_pose.t

    # Check that the detected pose is in a reasonable ballpark
    # We use loose tolerances since this is a very simple pose estimator
    position_error = np.linalg.norm(detected_translation - expected_translation)

    # The position should be somewhat close (within reasonable bounds)
    # This is more of a smoke test to ensure the pipeline works with real data
    assert position_error < 100.0  # Very loose tolerance - this is a basic method

    # Check that the depth is reasonable (should be positive and in expected range)
    assert detected_translation[2] > 0  # Depth should be positive
    assert (
        100 < detected_translation[2] < 2000
    )  # Reasonable depth range for this dataset


@runllms
def test_simple_2d_pose_detector_6d_with_gemini():
    """Test Simple2DPoseDetector6D with real tyo-l dataset and Gemini."""
    # Load the real data
    with open(ASSETS_PATH / "scene_camera.json", encoding="utf-8") as f:
        camera_data = json.load(f)
    with open(ASSETS_PATH / "scene_gt.json", encoding="utf-8") as f:
        gt_data = json.load(f)

    # Use image ID "1" which corresponds to the available files
    image_id = "1"

    # Extract camera parameters for this image
    cam_info = camera_data[image_id]
    cam_k = cam_info["cam_K"]  # 3x3 matrix flattened
    depth_scale = cam_info["depth_scale"]

    # Camera intrinsics: K matrix is [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    fx, fy = cam_k[0], cam_k[4]
    cx, cy = cam_k[2], cam_k[5]

    # Extract ground truth pose for object ID 1
    gt_pose = gt_data[image_id][0]  # First (and only) object in this image
    assert gt_pose["obj_id"] == 1

    # Ground truth rotation and translation
    gt_rotation = np.array(gt_pose["cam_R_m2c"]).reshape(3, 3)
    gt_translation = np.array(gt_pose["cam_t_m2c"])
    expected_pose = SE3.Rt(gt_rotation, gt_translation, check=False)

    # Load the real images
    rgb_img = iio.imread(ASSETS_PATH / "rgb" / "000001.png")
    depth_img = iio.imread(ASSETS_PATH / "depth" / "000001.png")

    # Create the object label
    obj_id = LanguageObjectDetectionID("remote control")

    # Create RGBD image
    rgbd = RGBDImage(rgb_img, depth_img)

    # Create camera info (assuming identity world transform for simplicity)
    camera = CameraInfo(
        name="tyo-l_camera",
        focal_length=(fx, fy),
        principal_point=(cx, cy),
        depth_scale=depth_scale,
        world_tform_camera=SE3(),  # Identity - poses will be in camera frame
    )

    # Create real detector
    object_detector = GeminiObjectDetector2D()
    object_detector = RenderWrapperObjectDetector2D(
        object_detector, Path("gemini-inner-detection")
    )
    detector = Simple2DPoseDetector6D(
        object_detector,
    )

    # Run the detection
    result = detector.detect({camera: rgbd}, [obj_id])

    # Verify we got a result
    assert len(result) == 1
    assert camera in result
    assert len(result[camera]) == 1

    detected_pose = result[camera][0]
    assert isinstance(detected_pose, DetectedPose6D)
    assert detected_pose.object_id == obj_id

    # The detected pose should be reasonably close to ground truth
    # Note: Simple2DPoseDetector6D only estimates position, not full 6D pose
    # and uses median depth, so we expect some difference
    detected_translation = detected_pose.pose.t
    expected_translation = expected_pose.t

    # Check that the detected pose is in a reasonable ballpark
    # We use loose tolerances since this is a very simple pose estimator
    position_error = np.linalg.norm(detected_translation - expected_translation)

    # The position should be somewhat close (within reasonable bounds)
    # This is more of a smoke test to ensure the pipeline works with real data
    assert position_error < 100.0  # Very loose tolerance - this is a basic method

    # Check that the depth is reasonable (should be positive and in expected range)
    assert detected_translation[2] > 0  # Depth should be positive
    assert (
        100 < detected_translation[2] < 2000
    )  # Reasonable depth range for this dataset
