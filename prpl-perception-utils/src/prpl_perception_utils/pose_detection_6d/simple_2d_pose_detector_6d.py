"""A simple pose detector that uses 2D object detection and median mask depth."""

import logging
from typing import Collection

import numpy as np
from spatialmath import SE3

from prpl_perception_utils.object_detection_2d.base_object_detector_2d import (
    ObjectDetector2D,
)
from prpl_perception_utils.pose_detection_6d.base_pose_detector_6d import PoseDetector6D
from prpl_perception_utils.structs import (
    CameraInfo,
    DetectedObject2D,
    DetectedPose6D,
    ObjectDetectionID,
    RGBDImage,
)


class Simple2DPoseDetector6D(PoseDetector6D):
    """A simple pose detector that uses 2D object detection and median mask depth.

    NOTE: this is a very, very imprecise approach. For example, there is NO attempt to
    determine a consistent frame of reference for objects. Furthermore, there is NO
    attempt to estimate the angle of the object whatsoever. This is really a 3D pose
    detector pretending to be a 6D one. But it can work well enough in some cases.
    """

    def __init__(
        self,
        object_detector: ObjectDetector2D,
        min_depth_value: int = 2,
    ) -> None:
        self._object_detector = object_detector
        self._min_depth_value = min_depth_value

    def detect(
        self,
        rgbds: dict[CameraInfo, RGBDImage],
        object_ids: Collection[ObjectDetectionID],
    ) -> dict[CameraInfo, list[DetectedPose6D]]:
        # First run 2D object detection.
        rgbs = [img.rgb for img in rgbds.values()]
        object_detections = self._object_detector.detect(rgbs, object_ids)
        # Now use depth to estimate Z and finish the pose.
        pose_detections: dict[CameraInfo, list[DetectedPose6D]] = {}
        for camera, detections in zip(rgbds, object_detections, strict=True):
            rgbd = rgbds[camera]
            camera_pose_detections: list[DetectedPose6D] = []
            for object_detection in detections:
                pose = self._object_detection_to_pose(object_detection, rgbd, camera)
                if pose is None:
                    continue
                pose_detection = DetectedPose6D(
                    object_detection.object_id,
                    pose,
                    confidence_score=object_detection.confidence_score,
                )
                camera_pose_detections.append(pose_detection)
            pose_detections[camera] = camera_pose_detections
        return pose_detections

    def _object_detection_to_pose(
        self, object_detection: DetectedObject2D, rgbd: RGBDImage, camera: CameraInfo
    ) -> SE3 | None:
        # Get the median depth value of segmented points.
        full_mask = object_detection.get_image_mask(rgbd.height, rgbd.width)
        seg_mask = full_mask & (rgbd.depth > self._min_depth_value)
        segmented_depth = rgbd.depth[seg_mask]

        # Uncomment to debug.
        # import imageio.v2 as iio
        # iio.imsave("debug/rgb.png", rgbd.rgb)
        # iio.imsave(
        #     "debug/depth.png", (255 * rgbd.depth / rgbd.depth.max()).astype(np.uint8)
        # )
        # iio.imsave("debug/full_mask.png", 255 * full_mask.astype(np.uint8))
        # iio.imsave("debug/seg_mask.png", 255 * seg_mask.astype(np.uint8))
        # seg_depth = np.zeros(rgbd.depth.shape, dtype=np.uint8)
        # seg_depth[seg_mask] = 255
        # iio.imsave("debug/seg_depth.png", seg_depth)
        # import ipdb; ipdb.set_trace()

        if len(segmented_depth) == 0:
            logging.warning("WARNING: depth reading failed.")
            return None
        depth_value = np.median(segmented_depth)
        # Convert to camera frame.
        x_center, y_center = object_detection.bounding_box.center
        fx, fy = camera.focal_length
        cx, cy = camera.principal_point
        depth_scale = camera.depth_scale
        camera_z = depth_value / depth_scale
        camera_x = np.multiply(camera_z, (x_center - cx)) / fx
        camera_y = np.multiply(camera_z, (y_center - cy)) / fy
        camera_frame_pose = SE3((camera_x, camera_y, camera_z))
        # Convert to world frame.
        world_frame_pose = camera.world_tform_camera * camera_frame_pose
        return world_frame_pose
