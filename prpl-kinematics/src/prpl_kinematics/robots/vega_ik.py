"""Analytic inverse kinematics for a Dexmate Vega arm.

Vega's wrist joints are not spherical, so no closed-form 6R IK applies. With two
joints locked (the elbow j4 and the wrist roll j7) the residual 5R chain matches
EAIK's catalog of solvable classes; to recover the full 7-DOF reach we search
the two locked values -- a coarse grid refined with Nelder-Mead -- driving
EAIK's least-squares pose residual to zero on the 1-D solvability manifold.

The chain's product-of-exponentials parameters (axes ``H``, offsets ``P``) are
extracted from the tree's forward kinematics, so the same class serves either
arm. It conforms to the ``InverseKinematics`` protocol.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from eaik.pybindings import EAIK
from scipy.optimize import minimize
from spatialmath import SE3

from prpl_kinematics.tree.joints import RevoluteJoint
from prpl_kinematics.tree.kinematic_tree import Configuration, KinematicTree

# Lock the elbow (index 3, j4) and the final wrist roll (index 6, j7).
_LOCK_A = 3
_LOCK_B = 6


class VegaArmIK:
    """EAIK-based IK for one 7-DOF Vega arm, with two joints locked and searched."""

    def __init__(
        self,
        tree: KinematicTree,
        arm_joints: Sequence[str],
        ee_link: str,
        tool_frame: str | None = None,
        grid_size: int = 20,
        refine_seeds: int = 5,
        tolerance: float = 1e-4,
    ) -> None:
        self._tree = tree
        self._arm_joints = list(arm_joints)
        self._ee_link = ee_link
        self._grid_size = grid_size
        self._refine_seeds = refine_seeds
        self._tolerance = tolerance

        edges = [e for e in tree.path_from_root(ee_link) if e.joint.name in arm_joints]
        self._base_frame = edges[0].parent
        self._lower = np.array([tree.joint(n).lower_limits[0] for n in arm_joints])
        self._upper = np.array([tree.joint(n).upper_limits[0] for n in arm_joints])

        # Product-of-exponentials parameters in the arm-base frame at zero config.
        offsets: list[np.ndarray] = []
        axes: list[np.ndarray] = []
        previous = np.zeros(3)
        for edge in edges:
            joint = edge.joint
            assert isinstance(joint, RevoluteJoint)
            frame = tree.relative_pose(self._base_frame, edge.child, {})
            axes.append(np.asarray(frame.R) @ np.array(joint.axis, dtype=float))
            offsets.append(np.asarray(frame.t) - previous)
            previous = np.asarray(frame.t)
        ee_position = np.asarray(tree.relative_pose(self._base_frame, ee_link, {}).t)
        offsets.append(ee_position - previous)
        self._h = np.array(axes).T
        self._p = np.array(offsets).T

        tool = tool_frame if tool_frame is not None else ee_link
        self._ee_from_tool = tree.relative_pose(ee_link, tool, {})

    def solve(self, target_pose: SE3, seed: Configuration) -> Configuration | None:
        """A configuration reaching ``target_pose`` (the tool frame), or ``None``."""
        ee_target = target_pose * self._ee_from_tool.inv()
        in_base = (
            self._tree.forward_kinematics(self._base_frame, seed).inv() * ee_target
        )
        target = np.asarray(in_base.A)

        a_lo, a_hi = self._lower[_LOCK_A] + 0.05, self._upper[_LOCK_A] - 0.05
        b_lo, b_hi = self._lower[_LOCK_B] + 0.05, self._upper[_LOCK_B] - 0.05
        grid = sorted(
            (
                (*self._best_for_lock(qa, qb, target)[::-1], qa, qb)
                for qa in np.linspace(a_lo, a_hi, self._grid_size)
                for qb in np.linspace(b_lo, b_hi, self._grid_size)
            ),
            key=lambda item: item[0],
        )
        best_residual, best_q, _, _ = grid[0]
        for _, _, qa_seed, qb_seed in grid[: self._refine_seeds]:
            if best_residual < self._tolerance:
                break
            opt = minimize(
                lambda x: self._best_for_lock(x[0], x[1], target)[1],
                x0=[qa_seed, qb_seed],
                method="Nelder-Mead",
                options={"xatol": 1e-5, "fatol": 1e-6, "maxiter": 200},
            )
            q, residual = self._best_for_lock(opt.x[0], opt.x[1], target)
            if residual < best_residual:
                best_residual, best_q = residual, q

        if best_q is None or best_residual >= self._tolerance:
            return None
        values = {name: [float(v)] for name, v in zip(self._arm_joints, best_q)}
        return {**dict(seed), **values}

    def _best_for_lock(
        self, qa: float, qb: float, target: np.ndarray
    ) -> tuple[np.ndarray | None, float]:
        """Solve EAIK with the two joints locked; return the best in-limits solution.

        Some locked values make the residual 5R chain degenerate (e.g. two axes
        parallel), which EAIK reports by raising; treat those as unsolvable.
        """
        try:
            robot = EAIK.Robot(
                self._h,
                self._p,
                np.eye(3),
                [(_LOCK_A, float(qa)), (_LOCK_B, float(qb))],
                True,
            )
            candidates = np.asarray(robot.calculate_IK(target).Q)
        except RuntimeError:
            return None, np.inf
        if candidates.size == 0:
            return None, np.inf
        if candidates.ndim == 1:
            candidates = candidates[None, :]
        unlocked = np.ones(len(self._arm_joints), dtype=bool)
        unlocked[[_LOCK_A, _LOCK_B]] = False
        best_q: np.ndarray | None = None
        best_residual = np.inf
        for q in candidates:
            if np.any(q[unlocked] < self._lower[unlocked] - 1e-6):
                continue
            if np.any(q[unlocked] > self._upper[unlocked] + 1e-6):
                continue
            pose = robot.fwdkin(q)
            position_error = float(np.linalg.norm(pose[:3, 3] - target[:3, 3]))
            rotation = pose[:3, :3].T @ target[:3, :3]
            angle_error = float(
                np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
            )
            residual = position_error + angle_error
            if residual < best_residual:
                best_residual, best_q = residual, np.asarray(q, dtype=float)
        return best_q, best_residual
