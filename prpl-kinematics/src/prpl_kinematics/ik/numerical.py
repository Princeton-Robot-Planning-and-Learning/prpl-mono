"""Numerical inverse kinematics via Jacobian damped least squares.

``NumericalIK`` solves for a configuration whose end-effector frame reaches a
target pose by stepping differentially from a seed configuration:
``dq = J^T (J J^T + lambda^2 I)^-1 e``, with the step clipped and clamped to the
joint limits each iteration. Seeding from the previous solution (see
:func:`~prpl_kinematics.ik.follow.follow_end_effector_path`) keeps a redundant
arm on a single, smooth IK branch instead of flipping between solutions.

The Jacobian is computed by finite-differencing the tree's forward kinematics,
so it works for any joint group without an analytic derivative.

This is a local solver: it converges when the seed is within the target's basin
of attraction (a nearby configuration), and returns ``None`` otherwise. Global,
seedless solutions are the job of an analytic solver (a later addition).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
from spatialmath import SE3

from prpl_kinematics.planning.joint_space import JointSpace
from prpl_kinematics.tree.kinematic_tree import Configuration, KinematicTree


def _pose_twist(current: SE3, target: SE3) -> np.ndarray:
    """6-vector ``[dx, dy, dz, rx, ry, rz]`` carrying ``current`` to ``target``."""
    position = np.asarray(target.t) - np.asarray(current.t)
    rotation = Rotation.from_matrix(np.asarray(target.R) @ np.asarray(current.R).T)
    return np.concatenate([position, rotation.as_rotvec()])


class NumericalIK:
    """Damped-least-squares differential IK over a JointSpace."""

    def __init__(
        self,
        tree: KinematicTree,
        space: JointSpace,
        ee_frame: str,
        position_tolerance: float = 1e-3,
        orientation_tolerance: float = 1e-2,
        max_iters: int = 100,
        damping: float = 0.05,
        step_clip: float = 0.2,
        jacobian_epsilon: float = 1e-6,
    ) -> None:
        self._tree = tree
        self._space = space
        self._ee_frame = ee_frame
        self._position_tolerance = position_tolerance
        self._orientation_tolerance = orientation_tolerance
        self._max_iters = max_iters
        self._damping = damping
        self._step_clip = step_clip
        self._jacobian_epsilon = jacobian_epsilon

    def solve(self, target_pose: SE3, seed: Configuration) -> Configuration | None:
        """A configuration reaching ``target_pose``, or ``None`` if not converged.

        The returned configuration carries every joint of ``seed``, with the
        space's joints driven toward the target.
        """
        base = dict(seed)
        q = self._space.to_vector(base)
        for _ in range(self._max_iters):
            config = {**base, **self._space.to_configuration(q)}
            current = self._tree.forward_kinematics(self._ee_frame, config)
            error = _pose_twist(current, target_pose)
            if self._within_tolerance(error):
                return config
            jacobian = self._jacobian(q, base, current)
            damped = jacobian @ jacobian.T + self._damping**2 * np.eye(6)
            dq = jacobian.T @ np.linalg.solve(damped, error)
            step = float(np.linalg.norm(dq))
            scale = self._step_clip / step if step > self._step_clip else 1.0
            q = self._space.clamp(q + dq * scale)
        config = {**base, **self._space.to_configuration(q)}
        current = self._tree.forward_kinematics(self._ee_frame, config)
        if self._within_tolerance(_pose_twist(current, target_pose)):
            return config
        return None

    def _within_tolerance(self, error: np.ndarray) -> bool:
        return bool(
            np.linalg.norm(error[:3]) < self._position_tolerance
            and np.linalg.norm(error[3:]) < self._orientation_tolerance
        )

    def _jacobian(self, q: np.ndarray, base: Configuration, current: SE3) -> np.ndarray:
        columns = []
        for i in range(q.size):
            perturbed = q.copy()
            perturbed[i] += self._jacobian_epsilon
            config = {**base, **self._space.to_configuration(perturbed)}
            pose = self._tree.forward_kinematics(self._ee_frame, config)
            columns.append(_pose_twist(current, pose) / self._jacobian_epsilon)
        return np.column_stack(columns)
