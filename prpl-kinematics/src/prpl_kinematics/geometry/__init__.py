"""Geometry: conversions between spatialmath ``SE3`` and external conventions.

Poses are spatialmath ``SE3``/``SE2`` everywhere; this package only bridges the
PyBullet boundary. Import ``SE3``, ``SE2``, ``SO3``, ``UnitQuaternion`` directly
from ``spatialmath``.
"""

from prpl_kinematics.geometry.transforms import (
    pose_from_pybullet,
    pose_to_pybullet,
)

__all__ = [
    "pose_from_pybullet",
    "pose_to_pybullet",
]
