"""EAIK-based inverse kinematics for the Dexmate Vega left arm.

Vega's wrist joints (L_arm_j5, L_arm_j6, L_arm_j7) are not spherical (their axes are
several cm apart), so traditional closed-form 6R IK does not apply. With two joints (the
elbow L_arm_j4 and the wrist roll L_arm_j7) locked, the residual 5R chain matches EAIK's
catalog of analytically-solvable kinematic classes. To recover the full 7-DOF reach, we
wrap a 2-D refinement search over the locked values around EAIK's 5R closed-form solver:
starting from a coarse grid, we use Nelder-Mead to drive EAIK's least-squares pose
residual to zero, landing on the 1-D solvability manifold.

EAIK is an optional dependency; if it isn't importable, EAIK_AVAILABLE is False and
callers should fall back to a different IK method.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

try:
    import eaik.pybindings.EAIK as _EAIK  # type: ignore[import-not-found]

    EAIK_AVAILABLE = True
except ImportError:
    EAIK_AVAILABLE = False


# Joint axes for the 7 left-arm joints, taken directly from the vega_1u URDF's
# <axis xyz="..."> entries (all parent rpy values are zero, so URDF axes are
# already expressed in EAIK's convention). Column i is the axis of joint i+1.
_H = np.array(
    [
        [0, 1, 0],  # L_arm_j1
        [0, 0, 1],  # L_arm_j2
        [1, 0, 0],  # L_arm_j3
        [0, 1, 0],  # L_arm_j4
        [1, 0, 0],  # L_arm_j5
        [0, 1, 0],  # L_arm_j6
        [0, 0, 1],  # L_arm_j7
    ],
    dtype=np.float64,
).T

# Joint position offsets (column i is the translation from joint i's frame to
# joint i+1's frame). Last column is the offset from joint 7 to the IK
# end-effector frame, which we take to be L_arm_l7 itself (zero offset). Values
# match the vega_1u URDF's <origin xyz="..."> entries on the left arm.
_P = np.array(
    [
        [0.0, 0.16946, 0.0],  # arm_center -> L_arm_l1
        [0.04, 0.06, 0.0454],
        [0.1644, 0.0, -0.043],
        [0.113, 0.0433, 0.06],
        [0.1938, -0.0434, -0.04],
        [0.0762, 0.0319, 0.0],
        [0.065, -0.032, 0.0319],  # L_arm_l6 -> L_arm_l7
        [0.0, 0.0, 0.0],  # L_arm_l7 -> EE (identity)
    ],
    dtype=np.float64,
).T

_R6T = np.eye(3)

# Indices into the 7-vector q for the two joints we lock during the EAIK call.
# L_arm_j4 is the elbow; L_arm_j7 is the final wrist roll. Locking both yields
# a 5R chain whose structure EAIK identifies as
# "5R-FOURTH_FITH_INTERSECTING_SECOND_THIRD_INTERSECTING", which is in its
# catalog of closed-form-solvable classes.
_LOCK_A = 3
_LOCK_B = 6

# Joint limits from the URDF, used to clamp the grid and to reject IK
# candidates that leave the legal range.
_LOWER = np.array([-3.071, -0.453, -3.071, -3.071, -3.071, -1.396, -1.378])
_UPPER = np.array([3.071, 1.553, 3.071, 0.244, 3.071, 1.396, 1.117])


def _best_for_lock(
    qa: float, qb: float, target_pose: np.ndarray
) -> tuple[np.ndarray | None, float]:
    """Run EAIK with (L_arm_j4=qa, L_arm_j7=qb) locked and return the best solution's
    joint vector and its pose residual to the target."""
    robot = _EAIK.Robot(
        _H, _P, _R6T, [(_LOCK_A, float(qa)), (_LOCK_B, float(qb))], True
    )
    sol = robot.calculate_IK(target_pose)
    candidates = np.asarray(sol.Q)
    if candidates.size == 0:
        return None, np.inf
    if candidates.ndim == 1:
        candidates = candidates[None, :]

    unlock_mask = np.ones(7, dtype=bool)
    unlock_mask[_LOCK_A] = False
    unlock_mask[_LOCK_B] = False

    best_q: np.ndarray | None = None
    best_res = np.inf
    for q in candidates:
        if np.any(q[unlock_mask] < _LOWER[unlock_mask] - 1e-6):
            continue
        if np.any(q[unlock_mask] > _UPPER[unlock_mask] + 1e-6):
            continue
        pose = robot.fwdkin(q)
        pos_err = float(np.linalg.norm(pose[:3, 3] - target_pose[:3, 3]))
        rot_diff = pose[:3, :3].T @ target_pose[:3, :3]
        rot_err = float(np.arccos(np.clip((np.trace(rot_diff) - 1.0) / 2.0, -1.0, 1.0)))
        res = pos_err + rot_err
        if res < best_res:
            best_res = res
            best_q = np.asarray(q, dtype=float)
    return best_q, best_res


def solve_left_arm_ik(
    target_pose: np.ndarray,
    *,
    seed: np.ndarray | None = None,
    n_grid: int = 20,
    n_refine_seeds: int = 5,
    tol: float = 1e-4,
) -> np.ndarray | None:
    """Return joint angles (length 7) reaching target_pose, or None if not found.

    target_pose: 4x4 homogeneous transform of L_arm_l7 in arm_center frame.
    seed: ignored for the grid sweep; future versions may bias the grid here.
    """
    if not EAIK_AVAILABLE:
        return None

    target_pose = np.asarray(target_pose, dtype=float)

    # Coarse sweep over (q_elbow, q_wrist_roll). Stay just inside the limits so
    # we don't waste samples on infeasible boundaries.
    a_grid = np.linspace(_LOWER[_LOCK_A] + 0.05, _UPPER[_LOCK_A] - 0.05, n_grid)
    b_grid = np.linspace(_LOWER[_LOCK_B] + 0.05, _UPPER[_LOCK_B] - 0.05, n_grid)
    grid_results: list[tuple[float, float, float, np.ndarray | None]] = []
    for qa in a_grid:
        for qb in b_grid:
            q, res = _best_for_lock(qa, qb, target_pose)
            grid_results.append((res, float(qa), float(qb), q))
    grid_results.sort(key=lambda item: item[0])

    best_res, _, _, best_q = grid_results[0]
    if best_res < tol:
        return best_q

    # Local refinement on EAIK's LS residual. The residual is smooth in
    # (qa, qb) and zero on the 1-D solvability manifold; multi-starting from
    # the best grid seeds raises the success rate substantially over a single
    # start.
    bounds = [(_LOWER[_LOCK_A], _UPPER[_LOCK_A]), (_LOWER[_LOCK_B], _UPPER[_LOCK_B])]

    def objective(x: np.ndarray) -> float:
        _, residual = _best_for_lock(x[0], x[1], target_pose)
        return residual

    for _, qa_seed, qb_seed, _ in grid_results[:n_refine_seeds]:
        opt = minimize(
            objective,
            x0=[qa_seed, qb_seed],
            method="Nelder-Mead",
            bounds=bounds,
            options={"xatol": 1e-5, "fatol": 1e-6, "maxiter": 200},
        )
        candidate_q, candidate_res = _best_for_lock(opt.x[0], opt.x[1], target_pose)
        if candidate_res < best_res:
            best_res = candidate_res
            best_q = candidate_q
            if best_res < tol:
                return best_q

    if best_q is not None and best_res < tol:
        return best_q
    return None
