"""Unit tests for the Blender rendering backend."""

import os

import numpy as np
import pytest

from prpl_kinematics.geometry.shapes import BoxShape
from prpl_kinematics.loading import load_urdf
from prpl_kinematics.tree.kinematic_tree import KinematicTree
from prpl_kinematics.utils import get_assets_path
from prpl_kinematics.visualization import (
    BlenderRenderer,
    CameraParams,
    render_configurations,
    save_video,
)
from prpl_kinematics.visualization.blender_renderer import _shape_spec


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


def test_blender_shape_spec_covers_primitives_and_meshes():
    """The Blender job spec captures each shape kind without launching Blender."""
    tree = load_urdf(_panda_path())
    specs = []
    index = 0
    for name, node in tree.nodes.items():
        for shape in node.visuals:
            specs.append(_shape_spec(index, name, shape))
            index += 1
    assert specs and all(s["kind"] == "mesh" for s in specs)  # Panda is all meshes
    assert all(len(s["origin"]) == 7 for s in specs)  # [x,y,z, qx,qy,qz,qw]


def test_blender_spec_includes_color_only_when_set():
    """The Blender job spec carries an explicit color and omits it otherwise."""
    plain = _shape_spec(0, "n", BoxShape(size=(1.0, 1.0, 1.0)))
    colored = _shape_spec(
        0, "n", BoxShape(size=(1.0, 1.0, 1.0), color=(1.0, 0.0, 0.0, 1.0))
    )
    assert "color" not in plain
    assert colored["color"] == [1.0, 0.0, 0.0, 1.0]


def test_blender_executable_honors_env(monkeypatch):
    """The Blender backend resolves its executable from $PRPL_BLENDER first."""
    monkeypatch.setenv("PRPL_BLENDER", "/custom/blender")
    assert BlenderRenderer().blender_executable == "/custom/blender"


def test_panda_blender_video(make_videos):
    """With --make-videos, render the sweep through Blender (skips if absent)."""
    if not make_videos:
        pytest.skip("pass --make-videos to render the video")
    try:
        renderer = BlenderRenderer(samples=48)
    except FileNotFoundError:
        pytest.skip("Blender executable not found")
    tree = load_urdf(_panda_path())
    renderer.load(tree)
    camera = CameraParams(
        target=(0.15, 0.0, 0.45),
        distance=1.3,
        yaw=55.0,
        pitch=-18.0,
        fov=50.0,
        width=480,
        height=360,
    )
    frames = render_configurations(renderer, _sweep_configs(tree, 24), camera)
    assert len(frames) == 24
    assert frames[0].shape == (360, 480, 3)
    save_video(frames, "panda_blender_sweep.mp4", fps=20)
    assert os.path.exists("panda_blender_sweep.mp4")
