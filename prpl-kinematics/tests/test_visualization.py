"""Shape-soup rendering tests, exercising the --make-videos pattern."""

import os

import numpy as np
import pybullet as p
import pybullet_data
import pytest
import trimesh

from prpl_kinematics.loading import load_urdf
from prpl_kinematics.tree.kinematic_tree import KinematicTree
from prpl_kinematics.visualization import (
    CameraParams,
    PyBulletRenderer,
    render_configurations,
    save_video,
    to_pybullet_mesh,
)


def _panda_path() -> str:
    return os.path.join(pybullet_data.getDataPath(), "franka_panda", "panda.urdf")


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


def test_native_mesh_passthrough():
    """Native mesh formats are returned unchanged, without conversion."""
    assert to_pybullet_mesh("/some/dir/link.obj") == "/some/dir/link.obj"
    assert to_pybullet_mesh("/some/dir/link.STL") == "/some/dir/link.STL"


def test_glb_converted_to_loadable_obj(physics_client_id, tmp_path):
    """A non-native .glb mesh is converted to a .obj that PyBullet can load."""
    glb = tmp_path / "box.glb"
    trimesh.creation.box((0.2, 0.2, 0.2)).export(str(glb))
    obj = to_pybullet_mesh(str(glb))
    assert obj.endswith(".obj")
    assert os.path.exists(obj)
    shape = p.createVisualShape(
        p.GEOM_MESH, fileName=obj, physicsClientId=physics_client_id
    )
    assert shape >= 0


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
