"""Unit tests for IKFast-backed analytic inverse kinematics.

The Panda IKFast module is compiled on demand the first time a solver is built; this
requires a C++ toolchain and LAPACK/BLAS (provided in CI).
"""

import numpy as np
from scipy.spatial.transform import Rotation
from spatialmath import SE3

from prpl_kinematics.ik import IKFastInfo, IKFastSolver, InverseKinematics
from prpl_kinematics.loading import load_urdf
from prpl_kinematics.utils import get_assets_path

ARM = [f"panda_joint{i}" for i in range(1, 8)]
EE = "panda_link8"
INFO = IKFastInfo(
    module_dir="panda_arm",
    module_name="ikfast_panda_arm",
    base_link="panda_link0",
    ee_link=EE,
    free_joints=["panda_joint7"],
)


def _panda():
    return load_urdf(str(get_assets_path() / "urdf" / "panda_arm_hand.urdf"))


def _solver(tree):
    return IKFastSolver(tree, INFO, ARM, np.random.default_rng(0))


def _config(values: list[float]) -> dict[str, list[float]]:
    return {name: [value] for name, value in zip(ARM, values)}


def _pose_error(tree, config, target) -> tuple[float, float]:
    reached = tree.forward_kinematics(EE, config)
    position = float(np.linalg.norm(np.asarray(target.t) - np.asarray(reached.t)))
    rotation = Rotation.from_matrix(np.asarray(target.R) @ np.asarray(reached.R).T)
    return position, float(np.linalg.norm(rotation.as_rotvec()))


def test_ikfast_solver_conforms_to_interface():
    """The solver satisfies the InverseKinematics protocol."""
    assert isinstance(_solver(_panda()), InverseKinematics)


def test_ikfast_solves_globally_from_distant_seed():
    """IKFast reaches a pose even when seeded from an unrelated configuration."""
    tree = _panda()
    solver = _solver(tree)
    truth = _config([0.2, -0.4, 0.1, -1.8, 0.0, 1.4, 0.5])
    target = tree.forward_kinematics(EE, truth)
    seed = _config([0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])
    solution = solver.solve(target, seed)
    assert solution is not None
    position_error, orientation_error = _pose_error(tree, solution, target)
    assert position_error < 1e-6 and orientation_error < 1e-6


def test_ikfast_selects_branch_closest_to_seed():
    """Seeding at a known solution returns that branch, not a far one."""
    tree = _panda()
    solver = _solver(tree)
    truth = _config([0.2, -0.4, 0.1, -1.8, 0.0, 1.4, 0.5])
    target = tree.forward_kinematics(EE, truth)
    solution = solver.solve(target, truth)
    assert solution is not None
    vector = np.array([solution[name][0] for name in ARM])
    assert np.allclose(vector, [truth[name][0] for name in ARM], atol=1e-3)


def test_ikfast_returns_none_when_unreachable():
    """A target far outside the workspace yields no solution."""
    tree = _panda()
    solver = _solver(tree)
    seed = _config([0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])
    assert solver.solve(SE3(5.0, 5.0, 5.0), seed) is None
