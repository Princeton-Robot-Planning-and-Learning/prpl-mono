"""Unit tests for the PyBullet rendering backend."""

import os

import numpy as np
import pytest
from spatialmath import SE3

from prpl_kinematics.geometry.shapes import BoxShape
from prpl_kinematics.loading import load_urdf
from prpl_kinematics.tree.joints import FixedJoint
from prpl_kinematics.tree.kinematic_tree import Edge, KinematicTree, Node
from prpl_kinematics.utils import get_assets_path
from prpl_kinematics.visualization import (
    DEFAULT_BACKGROUND_COLOR,
    CameraParams,
    PyBulletRenderer,
    render_configurations,
    save_video,
)


def _panda_path() -> str:
    return str(get_assets_path() / "urdf" / "panda_arm_hand.urdf")


def _sweep_configs(tree: KinematicTree, num_steps: int) -> list[dict[str, list[float]]]:
    """A gentle sinusoidal sweep of every actuated joint within its limits."""
    names = tree.actuated_joint_names()
    configs = []
    for step in range(num_steps):
        phase = 2 * np.pi * step / num_steps
        config: dict[str, list[float]] = {}
        for k, name in enumerate(names):
            joint = tree.joint(name)
            lo = max(joint.lower_limits[0], -2.5)
            hi = min(joint.upper_limits[0], 2.5)
            mid = 0.5 * (lo + hi)
            amplitude = 0.4 * (hi - lo)
            config[name] = [mid + amplitude * float(np.sin(phase + k))]
        configs.append(config)
    return configs


def test_panda_renders_frames(physics_client_id):
    """The shape-soup renderer yields correctly shaped RGB frames."""
    tree = load_urdf(_panda_path())
    renderer = PyBulletRenderer(physics_client_id)
    renderer.load(tree)
    camera = CameraParams(distance=1.6, width=320, height=240)
    frames = render_configurations(renderer, _sweep_configs(tree, 4), camera)
    assert len(frames) == 4
    assert frames[0].shape == (240, 320, 3)
    assert frames[0].dtype == np.uint8
    # The robot occupies a non-trivial portion of the frame (not blank).
    assert float((frames[0].sum(axis=2) < 720).mean()) > 0.01


def test_panda_sweep_video(physics_client_id, make_videos):
    """With --make-videos, render the full sweep to panda_sweep.mp4 in the cwd."""
    if not make_videos:
        pytest.skip("pass --make-videos to render the video")
    tree = load_urdf(_panda_path())
    renderer = PyBulletRenderer(physics_client_id)
    renderer.load(tree)
    camera = CameraParams(distance=1.6, width=480, height=360)
    frames = render_configurations(renderer, _sweep_configs(tree, 48), camera)
    save_video(frames, "panda_sweep.mp4", fps=20)
    assert os.path.exists("panda_sweep.mp4")


def test_pybullet_renders_shape_color(physics_client_id):
    """A red box renders with a red-dominant region (the color reaches PyBullet)."""
    tree = KinematicTree(root="base")
    box = BoxShape(size=(0.6, 0.6, 0.6), color=(1.0, 0.0, 0.0, 1.0))
    tree.add_node(Node("box", visuals=[box]))
    tree.add_edge(Edge("base", "box", FixedJoint(name="f", origin=SE3())))
    renderer = PyBulletRenderer(physics_client_id)
    renderer.load(tree)
    frame = renderer.render_frames([{}], CameraParams(target=(0, 0, 0), distance=2.0))[
        0
    ]
    red_over_blue = frame[:, :, 0].astype(int) - frame[:, :, 2].astype(int)
    assert float((red_over_blue > 40).mean()) > 0.02  # a clearly red region exists


def test_pybullet_background_default_and_disable(physics_client_id):
    """A background pixel is the soft-purple default, or white when disabled."""
    tree = KinematicTree(root="base")
    box = BoxShape(size=(0.2, 0.2, 0.2))
    tree.add_node(Node("box", visuals=[box]))
    tree.add_edge(Edge("base", "box", FixedJoint(name="f", origin=SE3())))
    camera = CameraParams(target=(0, 0, 0), distance=2.0)

    renderer = PyBulletRenderer(physics_client_id)  # default soft-purple background
    renderer.load(tree)
    purple = [round(c * 255) for c in DEFAULT_BACKGROUND_COLOR]
    assert renderer.render_frames([{}], camera)[0][0, 0].tolist() == purple

    plain = PyBulletRenderer(physics_client_id, background_color=None)
    plain.load(tree)
    assert plain.render_frames([{}], camera)[0][0, 0].tolist() == [255, 255, 255]
