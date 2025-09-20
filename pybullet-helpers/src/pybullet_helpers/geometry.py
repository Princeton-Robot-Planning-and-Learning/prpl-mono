"""PyBullet helper class for geometry utilities."""

from __future__ import annotations

from typing import Iterator, NamedTuple

import numpy as np
import numpy.typing as npt
import pybullet as p
from pybullet_utils.transformations import (
    euler_from_quaternion,
    quaternion_from_euler,
    quaternion_from_matrix,
)
from scipy.spatial.transform import Rotation as ScipyRotation
from scipy.spatial.transform import Slerp

Pose3D = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]
RollPitchYaw = tuple[float, float, float]


class Pose(NamedTuple):
    """Pose which is a position (translation) and rotation.

    We use a NamedTuple as it supports retrieving by integer indexing
    and most closely follows the PyBullet API.
    """

    # Cartesian (x, y, z) position
    position: Pose3D
    # Quaternion in (x, y, z, w) representation
    orientation: Quaternion = (0.0, 0.0, 0.0, 1.0)

    @classmethod
    def from_rpy(cls, translation: Pose3D, rpy: RollPitchYaw) -> Pose:
        """Create a Pose from translation and Euler roll-pitch-yaw angles."""
        return cls(translation, quaternion_from_euler(*rpy))

    @classmethod
    def from_matrix(cls, matrix: npt.NDArray) -> Pose:
        """Create a Pose from a 4x4 homogeneous matrix."""
        return Pose(
            position=tuple(matrix[:3, 3]), orientation=quat_from_matrix(matrix[:3, :3])
        )

    @property
    def rpy(self) -> RollPitchYaw:
        """Get the Euler roll-pitch-yaw representation."""
        return euler_from_quaternion(self.orientation)

    @classmethod
    def identity(cls) -> Pose:
        """Unit pose."""
        return cls((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

    def to_matrix(self) -> npt.NDArray:
        """Get the 4x4 homogenous matrix representation."""
        matrix = np.eye(4)
        matrix[:3, :3] = matrix_from_quat(self.orientation)
        matrix[:3, 3] = self.position
        return matrix

    def multiply(self, *poses: Pose) -> Pose:
        """Multiplies poses (i.e., rigid transforms) together."""
        return multiply_poses(self, *poses)

    def invert(self) -> Pose:
        """Invert the pose (i.e., transform)."""
        pos, quat = p.invertTransform(self.position, self.orientation)
        return Pose(pos, quat)

    def allclose(self, other: Pose, atol: float = 1e-6) -> bool:
        """Return whether this pose is close enough to another pose."""
        return np.allclose(
            self.position, other.position, atol=atol
        ) and orientations_allclose(self.orientation, other.orientation, atol=atol)


def multiply_poses(*poses: Pose) -> Pose:
    """Multiplies poses (which are essentially transforms) together."""
    pose = poses[0]
    for next_pose in poses[1:]:
        pybullet_pose = p.multiplyTransforms(
            pose.position, pose.orientation, next_pose.position, next_pose.orientation
        )
        pose = Pose(pybullet_pose[0], pybullet_pose[1])
    return pose


def orientations_allclose(
    quat1: Quaternion, quat2: Quaternion, atol: float = 1e-6
) -> bool:
    """Check whether two quaternion orientations are close, accounting for
    double coverage.

    Note that this should not be used to check if two rotations are
    equal, e.g., in the context of slerp.

    For example, see
    https://gamedev.stackexchange.com/questions/75072/.
    """
    return np.allclose(quat1, quat2, atol=atol) or np.allclose(
        quat1, -1 * np.array(quat2), atol=atol
    )


def matrix_from_quat(quat: Quaternion) -> npt.NDArray[np.float64]:
    """Get 3x3 rotation matrix from quaternion (xyzw)."""
    return np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)


def quat_from_matrix(matrix: npt.NDArray[np.float64]) -> Quaternion:
    """Get quaternion (xyzw) from 3x3 rotation matrix."""
    M = np.identity(4)
    M[:3, :3] = matrix
    return tuple(quaternion_from_matrix(M))


def rotate_pose(
    pose: Pose, roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0
) -> Pose:
    """Rotate a pose by the given rpy to make a new pose."""
    current_orn = pose.orientation
    rot_orn = quaternion_from_euler(roll, pitch, yaw)
    current_mat = matrix_from_quat(current_orn)
    rot_mat = matrix_from_quat(rot_orn)
    new_mat = current_mat @ rot_mat
    new_orn = quat_from_matrix(new_mat)
    return Pose(pose.position, new_orn)


def get_pose(body: int, physics_client_id: int) -> Pose:
    """Get the pose of a body."""
    pybullet_pose = p.getBasePositionAndOrientation(
        body, physicsClientId=physics_client_id
    )
    return Pose(pybullet_pose[0], pybullet_pose[1])


def set_pose(body: int, pose: Pose, physics_client_id: int) -> None:
    """Set the pose of a body."""
    p.resetBasePositionAndOrientation(
        body,
        pose.position,
        pose.orientation,
        physicsClientId=physics_client_id,
    )


def iter_between_quats(
    q1: Quaternion,
    q2: Quaternion,
    num_interp: int = 10,
    include_start: bool = True,
) -> Iterator[Quaternion]:
    """Interpolate quaternions using slerp."""
    slerp = Slerp([0, num_interp], ScipyRotation.from_quat([q1, q2]))
    time_start = 0 if include_start else 1
    times = list(range(time_start, num_interp + 1))
    for t in times:
        yield tuple(slerp(t).as_quat())


def iter_between_pose3ds(
    p1: Pose3D,
    p2: Pose3D,
    num_interp: int = 10,
    include_start: bool = True,
) -> Iterator[Pose3D]:
    """Interpolate positions."""
    time_start = 0 if include_start else 1
    times = list(range(time_start, num_interp + 1))
    positions = np.linspace(p1, p2, num=(num_interp + 1), endpoint=True)
    for t in times:
        yield tuple(positions[t])


def iter_between_poses(
    p1: Pose,
    p2: Pose,
    num_interp: int = 10,
    include_start: bool = True,
) -> Iterator[Pose]:
    """Interpolate between two poses in pose space."""
    # Determine the number of interpolation steps.
    pose3d_gen = iter_between_pose3ds(
        p1.position, p2.position, num_interp=num_interp, include_start=include_start
    )
    quat_gen = iter_between_quats(
        p1.orientation,
        p2.orientation,
        num_interp=num_interp,
        include_start=include_start,
    )
    for position, orientation in zip(pose3d_gen, quat_gen, strict=True):
        yield Pose(position, orientation)


def interpolate_quats(q1: Quaternion, q2: Quaternion, t: float) -> Quaternion:
    """Interpolate between q1 and q2 given 0 <= t <= 1."""
    assert 0 <= t <= 1
    slerp = Slerp([0, 1], ScipyRotation.from_quat([q1, q2]))
    return tuple(slerp(t).as_quat())


def interpolate_pose3d(p1: Pose3D, p2: Pose3D, t: float) -> Pose3D:
    """Interpolate between p1 and p2 given 0 <= t <= 1."""
    dists_arr = np.subtract(p2, p1)
    return tuple(np.add(p1, t * dists_arr))


def interpolate_poses(
    p1: Pose,
    p2: Pose,
    t: float,
) -> Pose:
    """Interpolate between p1 and p2 given 0 <= t <= 1."""
    assert 0 <= t <= 1
    position = interpolate_pose3d(p1.position, p2.position, t)
    quat = interpolate_quats(p1.orientation, p2.orientation, t)
    return Pose(position, quat)


def get_half_extents_from_aabb(
    body_id: int,
    physics_client_id: int,
    link_id: int | None = None,
    rotation_okay: bool = False,
) -> tuple[float, float, float]:
    """Get box half extents based on AABB."""
    if not rotation_okay:
        pose = get_pose(body_id, physics_client_id)
        if not pose.allclose(Pose(pose.position)):
            raise NotImplementedError("This is too confusing when objects are rotated.")
    if link_id is None:
        aabb_min, aabb_max = p.getAABB(body_id, physicsClientId=physics_client_id)
    else:
        aabb_min, aabb_max = p.getAABB(
            body_id, linkIndex=link_id, physicsClientId=physics_client_id
        )
    return (
        (aabb_max[0] - aabb_min[0]) / 2,
        (aabb_max[1] - aabb_min[1]) / 2,
        (aabb_max[2] - aabb_min[2]) / 2,
    )
