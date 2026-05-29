"""Conversions between spatialmath ``SE3`` and external pose conventions.

Internally, prpl_kinematics represents every rigid-body pose as a spatialmath
``SE3`` (or ``SE2`` for planar quantities) and every rotation as ``SO3`` /
``UnitQuaternion``. We deliberately do *not* define our own Pose or Quaternion
classes -- spatialmath already ships battle-tested ones, and elsewhere in the
codebase a roll-pitch-yaw pose is just ``SE3(x, y, z) * SE3.RPY([r, p, y])``.

The only conversions we need are at the boundary with tools that use a different
convention. PyBullet represents a pose as a position ``(x, y, z)`` plus a
quaternion in ``(x, y, z, w)`` order, whereas spatialmath's ``UnitQuaternion``
uses ``(w, x, y, z)`` order; these helpers bridge that gap.
"""

from __future__ import annotations

from collections.abc import Sequence

from scipy.spatial.transform import Rotation
from spatialmath import SE3


def pose_from_pybullet(position: Sequence[float], orientation: Sequence[float]) -> SE3:
    """Build an ``SE3`` from a PyBullet ``(position, xyzw-quaternion)`` pose."""
    rotation = Rotation.from_quat(list(orientation)).as_matrix()
    return SE3.Rt(rotation, list(position))


def pose_to_pybullet(
    pose: SE3,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Convert an ``SE3`` to a PyBullet ``(position, xyzw-quaternion)`` pose."""
    quat = Rotation.from_matrix(pose.R).as_quat()
    translation = pose.t
    position = (
        float(translation[0]),
        float(translation[1]),
        float(translation[2]),
    )
    orientation = (
        float(quat[0]),
        float(quat[1]),
        float(quat[2]),
        float(quat[3]),
    )
    return position, orientation
