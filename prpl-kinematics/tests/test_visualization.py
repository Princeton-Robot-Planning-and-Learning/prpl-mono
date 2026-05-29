"""Render a Panda joint sweep, exercising the --make-videos pattern."""

import os

import numpy as np
import pybullet_data
import pytest

from prpl_kinematics.loading import load_urdf
from prpl_kinematics.tree.kinematic_tree import KinematicTree
from prpl_kinematics.visualization import (
    CameraParams,
    load_urdf_for_rendering,
    render_configurations,
    save_video,
)


def _panda_path() -> str:
    return os.path.join(pybullet_data.getDataPath(), "franka_panda", "panda.urdf")


def _sweep_configs(tree: KinematicTree, num_steps: int) -> list[dict[str, list[float]]]:
    """A gentle sinusoidal sweep of every actuated joint within its limits."""
    names = tree.actuated_joint_names()
    configs = []
    for step in range(num_steps):
        phase = 2 * np.pi * step / num_steps
        config = {}
        for k, name in enumerate(names):
            joint = tree.joint(name)
            lo = max(joint.lower_limits[0], -2.5)
            hi = min(joint.upper_limits[0], 2.5)
            mid = 0.5 * (lo + hi)
            amplitude = 0.4 * (hi - lo)
            config[name] = [mid + amplitude * np.sin(phase + k)]
        configs.append(config)
    return configs


def test_panda_sweep_renders_frames(physics_client_id):
    """Rendering yields correctly shaped RGB frames (runs without --make-videos)."""
    tree = load_urdf(_panda_path())
    body, joint_index = load_urdf_for_rendering(physics_client_id, _panda_path())
    camera = CameraParams(distance=1.6, width=320, height=240)
    frames = render_configurations(
        physics_client_id, body, joint_index, _sweep_configs(tree, 4), camera
    )
    assert len(frames) == 4
    assert frames[0].shape == (240, 320, 3)
    assert frames[0].dtype == np.uint8


def test_panda_sweep_video(physics_client_id, make_videos):
    """With --make-videos, render the full sweep to panda_sweep.mp4 in the cwd."""
    if not make_videos:
        pytest.skip("pass --make-videos to render the video")
    tree = load_urdf(_panda_path())
    body, joint_index = load_urdf_for_rendering(physics_client_id, _panda_path())
    camera = CameraParams(distance=1.6, width=480, height=360)
    frames = render_configurations(
        physics_client_id, body, joint_index, _sweep_configs(tree, 48), camera
    )
    save_video(frames, "panda_sweep.mp4", fps=20)
    assert os.path.exists("panda_sweep.mp4")
