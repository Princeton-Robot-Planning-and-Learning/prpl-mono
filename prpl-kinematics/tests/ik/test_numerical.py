"""Unit tests for numerical inverse kinematics."""

import numpy as np
from scipy.spatial.transform import Rotation
from spatialmath import SE3

from prpl_kinematics.ik import InverseKinematics, NumericalIK
from prpl_kinematics.loading import load_urdf
from prpl_kinematics.planning import JointSpace
from prpl_kinematics.utils import get_assets_path

ARM = [f"panda_joint{i}" for i in range(1, 8)]


def _panda():
    path = str(get_assets_path() / "urdf" / "panda_arm_hand.urdf")
    tree = load_urdf(path)
    return tree, JointSpace(tree, ARM)


def _comfortable() -> dict[str, list[float]]:
    config = {name: [0.0] for name in ARM}
    config["panda_joint2"] = [-0.5]
    config["panda_joint4"] = [-1.8]
    config["panda_joint6"] = [1.5]
    return config


def _pose_error(tree, ee, config, target) -> tuple[float, float]:
    reached = tree.forward_kinematics(ee, config)
    position = float(np.linalg.norm(np.asarray(target.t) - np.asarray(reached.t)))
    rotation = Rotation.from_matrix(np.asarray(target.R) @ np.asarray(reached.R).T)
    return position, float(np.linalg.norm(rotation.as_rotvec()))


def test_numerical_ik_conforms_to_interface():
    """NumericalIK satisfies the InverseKinematics protocol."""
    tree, space = _panda()
    assert isinstance(NumericalIK(tree, space, "panda_hand"), InverseKinematics)


def test_solve_reaches_nearby_pose():
    """IK recovers a configuration reaching a pose from a nearby seed."""
    tree, space = _panda()
    ik = NumericalIK(tree, space, "panda_hand")
    truth = _comfortable()
    target = tree.forward_kinematics("panda_hand", truth)
    seed = {name: [value[0] + 0.3] for name, value in truth.items()}
    solution = ik.solve(target, seed)
    assert solution is not None
    position_error, orientation_error = _pose_error(
        tree, "panda_hand", solution, target
    )
    assert position_error < 1e-3 and orientation_error < 1e-2
    vector = space.to_vector(solution)
    assert np.allclose(vector, space.clamp(vector))


def test_solve_returns_none_when_unreachable():
    """A target far outside the workspace yields no solution."""
    tree, space = _panda()
    ik = NumericalIK(tree, space, "panda_hand")
    assert ik.solve(SE3(5.0, 5.0, 5.0), _comfortable()) is None
