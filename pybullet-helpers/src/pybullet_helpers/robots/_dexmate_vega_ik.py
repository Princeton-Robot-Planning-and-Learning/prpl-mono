"""EAIK-based inverse kinematics for the Dexmate Vega arms.

Vega's wrist joints (j5, j6, j7) are not spherical (their axes are several cm apart), so
traditional closed-form 6R IK does not apply. With two joints (the elbow j4 and the wrist
roll j7) locked, the residual 5R chain matches EAIK's catalog of analytically-solvable
kinematic classes. To recover the full 7-DOF reach, we wrap a 2-D refinement search over
the locked values around EAIK's 5R closed-form solver: starting from a coarse grid, we use
Nelder-Mead to drive EAIK's least-squares pose residual to zero, landing on the 1-D
solvability manifold.

The two arms share this structure but have different (mirrored) geometry, so each arm has
its own ``ArmIKParams``. Parameters are extracted directly from the vega_1u URDF -- a single
source of truth that avoids hand-transcribing the mirrored right-arm values.

EAIK is an optional dependency; if it isn't importable, EAIK_AVAILABLE is False and callers
should fall back to a different IK method.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from dexmate_urdf import get_robot_path
from scipy.optimize import minimize

try:
    import eaik.pybindings.EAIK as _EAIK  # type: ignore[import-not-found]

    EAIK_AVAILABLE = True
except ImportError:
    EAIK_AVAILABLE = False


# Indices into the 7-vector q for the two joints we lock during the EAIK call.
# Joint 4 is the elbow; joint 7 is the final wrist roll. Locking both yields a 5R
# chain whose structure EAIK identifies as
# "5R-FOURTH_FITH_INTERSECTING_SECOND_THIRD_INTERSECTING", which is in its catalog
# of closed-form-solvable classes.
_LOCK_A = 3
_LOCK_B = 6


@dataclass(frozen=True)
class ArmIKParams:
    """Kinematic parameters for one Vega arm in EAIK's product-of-exponentials form.

    H: 3x7, column i is the rotation axis of joint i+1 (in arm_center's frame).
    P: 3x8, column i is the translation from joint i's frame to joint i+1's frame;
        the last column is the offset from joint 7 to the IK end-effector frame
        (taken to be the arm's l7 link, i.e. zero offset).
    R6T: 3x3 end-effector orientation offset (identity here).
    lower/upper: length-7 joint limits, used to clamp the search grid and reject
        out-of-range IK candidates.
    """

    H: np.ndarray
    P: np.ndarray
    R6T: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    lock_a: int = _LOCK_A
    lock_b: int = _LOCK_B


def _joint_block(urdf_str: str, joint_name: str) -> str:
    match = re.search(
        r'<joint name="' + re.escape(joint_name) + r'".*?</joint>', urdf_str, re.S
    )
    assert match is not None, f"joint {joint_name} not found in URDF"
    return match.group(0)


@lru_cache(maxsize=2)
def get_arm_ik_params(prefix: str) -> ArmIKParams:
    """Extract EAIK kinematic parameters for the given arm ("L" or "R") from the URDF.

    EAIK expects all joint axes expressed in a common frame. The Vega arm joints all
    have zero rpy on their <origin>, so the URDF <axis> entries are already in that
    convention; we assert this to guard against silent breakage if the URDF changes.
    """
    assert prefix in ("L", "R"), f"Unknown arm prefix {prefix}"
    robot_dir = get_robot_path("humanoid", "vega_1u")
    urdf_str = (robot_dir / "vega_1u.urdf").read_text(encoding="utf-8")

    axes: list[list[float]] = []
    offsets: list[list[float]] = []
    lower: list[float] = []
    upper: list[float] = []
    for i in range(1, 8):
        block = _joint_block(urdf_str, f"{prefix}_arm_j{i}")
        axis = [
            float(v) for v in re.search(r'<axis xyz="([^"]+)"', block).group(1).split()
        ]
        origin = re.search(r'<origin xyz="([^"]+)"[^>]*?(?:rpy="([^"]+)")?', block)
        xyz = [float(v) for v in origin.group(1).split()]
        rpy = origin.group(2)
        if rpy is not None:
            assert np.allclose(
                [float(v) for v in rpy.split()], 0.0
            ), f"{prefix}_arm_j{i} has nonzero rpy; EAIK axis convention assumption broken"
        limits = re.search(r'lower="([^"]+)" upper="([^"]+)"', block).groups()
        axes.append(axis)
        offsets.append(xyz)
        lower.append(float(limits[0]))
        upper.append(float(limits[1]))
    # Final column: joint 7 -> end-effector frame (l7 itself), zero offset.
    offsets.append([0.0, 0.0, 0.0])

    return ArmIKParams(
        H=np.array(axes, dtype=np.float64).T,
        P=np.array(offsets, dtype=np.float64).T,
        R6T=np.eye(3),
        lower=np.array(lower, dtype=np.float64),
        upper=np.array(upper, dtype=np.float64),
    )


def _best_for_lock(
    qa: float, qb: float, target_pose: np.ndarray, params: ArmIKParams
) -> tuple[np.ndarray | None, float]:
    """Run EAIK with the two lock joints fixed at (qa, qb) and return the best
    solution's joint vector and its pose residual to the target."""
    robot = _EAIK.Robot(
        params.H,
        params.P,
        params.R6T,
        [(params.lock_a, float(qa)), (params.lock_b, float(qb))],
        True,
    )
    sol = robot.calculate_IK(target_pose)
    candidates = np.asarray(sol.Q)
    if candidates.size == 0:
        return None, np.inf
    if candidates.ndim == 1:
        candidates = candidates[None, :]

    unlock_mask = np.ones(7, dtype=bool)
    unlock_mask[params.lock_a] = False
    unlock_mask[params.lock_b] = False

    best_q: np.ndarray | None = None
    best_res = np.inf
    for q in candidates:
        if np.any(q[unlock_mask] < params.lower[unlock_mask] - 1e-6):
            continue
        if np.any(q[unlock_mask] > params.upper[unlock_mask] + 1e-6):
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


def solve_arm_ik(
    target_pose: np.ndarray,
    params: ArmIKParams,
    *,
    n_grid: int = 20,
    n_refine_seeds: int = 5,
    tol: float = 1e-4,
) -> np.ndarray | None:
    """Return joint angles (length 7) reaching target_pose for the given arm, or None.

    target_pose: 4x4 homogeneous transform of the arm's l7 link in arm_center frame.
    """
    if not EAIK_AVAILABLE:
        return None

    target_pose = np.asarray(target_pose, dtype=float)

    # Coarse sweep over (q_elbow, q_wrist_roll). Stay just inside the limits so we
    # don't waste samples on infeasible boundaries.
    a_grid = np.linspace(
        params.lower[params.lock_a] + 0.05, params.upper[params.lock_a] - 0.05, n_grid
    )
    b_grid = np.linspace(
        params.lower[params.lock_b] + 0.05, params.upper[params.lock_b] - 0.05, n_grid
    )
    grid_results: list[tuple[float, float, float, np.ndarray | None]] = []
    for qa in a_grid:
        for qb in b_grid:
            q, res = _best_for_lock(qa, qb, target_pose, params)
            grid_results.append((res, float(qa), float(qb), q))
    grid_results.sort(key=lambda item: item[0])

    best_res, _, _, best_q = grid_results[0]
    if best_res < tol:
        return best_q

    # Local refinement on EAIK's LS residual. The residual is smooth in (qa, qb) and
    # zero on the 1-D solvability manifold; multi-starting from the best grid seeds
    # raises the success rate substantially over a single start.
    bounds = [
        (params.lower[params.lock_a], params.upper[params.lock_a]),
        (params.lower[params.lock_b], params.upper[params.lock_b]),
    ]

    def objective(x: np.ndarray) -> float:
        _, residual = _best_for_lock(x[0], x[1], target_pose, params)
        return residual

    for _, qa_seed, qb_seed, _ in grid_results[:n_refine_seeds]:
        opt = minimize(
            objective,
            x0=[qa_seed, qb_seed],
            method="Nelder-Mead",
            bounds=bounds,
            options={"xatol": 1e-5, "fatol": 1e-6, "maxiter": 200},
        )
        candidate_q, candidate_res = _best_for_lock(
            opt.x[0], opt.x[1], target_pose, params
        )
        if candidate_res < best_res:
            best_res = candidate_res
            best_q = candidate_q
            if best_res < tol:
                return best_q

    if best_q is not None and best_res < tol:
        return best_q
    return None
