"""Data structures."""

import abc
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from spatialmath import SE3

RGBImage: TypeAlias = NDArray[np.uint8]  # must have shape (H, W, 3)
DepthImage: TypeAlias = NDArray[np.uint16]  # must have shape (H, W)


@dataclass(frozen=True)
class RGBDImage:
    """An RGB and depth image."""

    rgb: RGBImage
    depth: DepthImage

    def __post_init__(self) -> None:
        height, width, channels = self.rgb.shape
        assert channels == 3, "RGB images must have 3 channels"
        assert self.depth.shape == (height, width), "RGB and depth shapes do not match"

    @property
    def height(self) -> int:
        """The height of the image."""
        return self.rgb.shape[0]

    @property
    def width(self) -> int:
        """The width of the image."""
        return self.rgb.shape[1]


@dataclass(frozen=True)
class CameraInfo:
    """Camera information, including intrinsics and extrinsics."""

    name: str
    focal_length: tuple[float, float]
    principal_point: tuple[float, float]
    depth_scale: float
    world_tform_camera: SE3

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"CameraInfo({self.name})"

    def __hash__(self) -> int:
        return hash(repr(self))

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, CameraInfo)
        return self.name == other.name


class ObjectDetectionID(abc.ABC):
    """A unique identifier for an object that is to be detected."""


@dataclass(frozen=True)
class LanguageObjectDetectionID(ObjectDetectionID):
    """An ID for an object to be detected with a vision-language model."""

    language_id: str

    def __str__(self) -> str:
        return self.language_id

    def __repr__(self) -> str:
        return f"LanguageID({self.language_id})"

    def __hash__(self) -> int:
        return hash(repr(self))

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, LanguageObjectDetectionID)
        return self.language_id == other.language_id


@dataclass(frozen=True)
class BoundingBox:
    """A 2D bounding box, where (x=0, y=0) is the top left corner of the image."""

    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        assert self.x1 <= self.x2
        assert self.y1 <= self.y2

    @property
    def width(self) -> int:
        """The width of the bounding box."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """The height of the bounding box."""
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        """The center of the bounding box."""
        x_center = (self.x1 + self.x2) / 2
        y_center = (self.y1 + self.y2) / 2
        return (x_center, y_center)


@dataclass(frozen=True)
class DetectedObject2D:
    """A bounding box, mask, and confidence score for an object in an image."""

    object_id: ObjectDetectionID
    bounding_box: BoundingBox
    mask: NDArray[np.bool_]  # over the bounding box
    confidence_score: float = 1.0  # between 0 and 1

    def __post_init__(self) -> None:
        assert self.mask.shape == (self.bounding_box.height, self.bounding_box.width)
        assert 0 <= self.confidence_score <= 1

    def get_image_mask(self, height: int, width: int) -> NDArray[np.bool_]:
        """Get a full image mask given the height and width of the image."""
        full_mask = np.zeros((height, width), dtype=np.bool_)
        x1 = self.bounding_box.x1
        x2 = self.bounding_box.x2
        y1 = self.bounding_box.y1
        y2 = self.bounding_box.y2
        full_mask[y1:y2, x1:x2] = self.mask
        return full_mask


@dataclass(frozen=True)
class DetectedPose6D:
    """A detected 6D pose and confidence score for an object in an image.

    Note that the pose is in the world frame.
    """

    object_id: ObjectDetectionID
    pose: SE3
    confidence_score: float = 1.0  # between 0 and 1

    def __post_init__(self) -> None:
        assert 0 <= self.confidence_score <= 1
