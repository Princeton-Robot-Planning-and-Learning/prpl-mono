"""Unit tests for numerical inverse kinematics and end-effector following."""

import os

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from spatialmath import SE3

from prpl_kinematics.ik import (
    InverseKinematics,
    NumericalIK,
    follow_end_effector_path,
)
from prpl_kinematics.loading import load_urdf
from prpl_kinematics.planning import JointSpace
from prpl_kinematics.utils import get_assets_path
from prpl_kinematics.visualization import (
    CameraParams,
    PyBulletRenderer,
    render_configurations,
    save_video,
)

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


def test_follow_end_effector_path_is_smooth():
    """Following a Cartesian path warm-started stays on one smooth IK branch."""
    tree, space = _panda()
    ik = NumericalIK(tree, space, "panda_hand")
    start = _comfortable()
    origin = tree.forward_kinematics("panda_hand", start)
    # A 15 cm straight line in world +y, sampled finely.
    poses = [SE3(0.0, 0.15 * k / 40, 0.0) * origin for k in range(41)]
    path = follow_end_effector_path(ik, poses, start)
    assert path is not None and len(path) == len(poses)
    for config, target in zip(path, poses):
        position_error, orientation_error = _pose_error(
            tree, "panda_hand", config, target
        )
        assert position_error < 1e-3 and orientation_error < 1e-2
    steps = [
        space.distance(space.to_vector(path[i]), space.to_vector(path[i + 1]))
        for i in range(len(path) - 1)
    ]
    assert max(steps) < 0.1  # no IK-branch flips


def test_follow_returns_none_when_a_pose_is_unreachable():
    """The follower reports failure if any pose along the path is unreachable."""
    tree, space = _panda()
    ik = NumericalIK(tree, space, "panda_hand")
    start = _comfortable()
    origin = tree.forward_kinematics("panda_hand", start)
    poses = [origin, SE3(5.0, 5.0, 5.0)]
    assert follow_end_effector_path(ik, poses, start) is None


def test_ik_follow_video(physics_client_id, make_videos):
    """With --make-videos, render an EE circle traced via warm-started IK."""
    if not make_videos:
        pytest.skip("pass --make-videos to render the video")
    tree, space = _panda()
    ik = NumericalIK(tree, space, "panda_hand")
    start = _comfortable()
    origin = tree.forward_kinematics("panda_hand", start)
    poses = []
    for k in range(80):
        angle = 2 * np.pi * k / 80
        offset = SE3(0.0, 0.08 * np.sin(angle), 0.08 * (1 - np.cos(angle)))
        poses.append(offset * origin)
    path = follow_end_effector_path(ik, poses, start)
    assert path is not None
    renderer = PyBulletRenderer(physics_client_id)
    renderer.load(tree)
    camera = CameraParams(target=(0.3, 0.0, 0.6), distance=1.5, yaw=90.0, pitch=-15.0)
    frames = render_configurations(renderer, path, camera)
    save_video(frames, "panda_ik_follow.mp4", fps=20)
    assert os.path.exists("panda_ik_follow.mp4")
