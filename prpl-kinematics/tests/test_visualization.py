"""Shape-soup rendering tests, exercising the --make-videos pattern."""

import os

import numpy as np
import pybullet as p
import pytest
import trimesh
from spatialmath import SE3

from prpl_kinematics.geometry.shapes import BoxShape
from prpl_kinematics.loading import load_urdf
from prpl_kinematics.meshes import to_pybullet_mesh
from prpl_kinematics.robots import make_panda
from prpl_kinematics.tree.joints import FixedJoint
from prpl_kinematics.tree.kinematic_tree import Configuration, Edge, KinematicTree, Node
from prpl_kinematics.tree.state import KinematicState
from prpl_kinematics.utils import get_assets_path
from prpl_kinematics.visualization import (
    DEFAULT_BACKGROUND_COLOR,
    BlenderRenderer,
    CameraParams,
    PyBulletRenderer,
    Renderer,
    render_configurations,
    render_states,
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


def test_renderers_conform_to_interface(physics_client_id):
    """Both backends are Renderers (the protocol the helpers consume)."""
    assert isinstance(PyBulletRenderer(physics_client_id), Renderer)
    assert isinstance(BlenderRenderer(blender_executable="/usr/bin/false"), Renderer)


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


def test_shape_color_defaults_to_none():
    """Shapes are uncolored by default, so backends apply their own defaults."""
    assert BoxShape(size=(1.0, 1.0, 1.0)).color is None


def test_blender_spec_includes_color_only_when_set():
    """The Blender job spec carries an explicit color and omits it otherwise."""
    plain = _shape_spec(0, "n", BoxShape(size=(1.0, 1.0, 1.0)))
    colored = _shape_spec(
        0, "n", BoxShape(size=(1.0, 1.0, 1.0), color=(1.0, 0.0, 0.0, 1.0))
    )
    assert "color" not in plain
    assert colored["color"] == [1.0, 0.0, 0.0, 1.0]


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


def test_blender_executable_honors_env(monkeypatch):
    """The Blender backend resolves its executable from $PRPL_BLENDER first."""
    monkeypatch.setenv("PRPL_BLENDER", "/custom/blender")
    assert BlenderRenderer().blender_executable == "/custom/blender"


class _RecordingRenderer(Renderer):
    """A renderer that records each render_frames batch and the cube's parent."""

    def __init__(self, tree: KinematicTree) -> None:
        self._tree = tree
        self.batch_sizes: list[int] = []
        self.parents_at_call: list[str] = []

    def load(self, tree: KinematicTree) -> None:
        pass

    def render(self, config: Configuration) -> None:
        pass

    def capture_image(self, camera: CameraParams = CameraParams()) -> np.ndarray:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    def render_frames(self, configs, camera=CameraParams()):
        config_list = list(configs)
        self.batch_sizes.append(len(config_list))
        self.parents_at_call.append(self._tree.edges["cube"].parent)
        return [np.zeros((1, 1, 3), dtype=np.uint8) for _ in config_list]


def test_render_states_batches_by_structure():
    """render_states groups consecutive same-structure states into one batch."""
    robot = make_panda()
    tree = robot.tree
    cube = BoxShape(size=(0.05, 0.05, 0.05))
    tree.add_node(Node("cube", visuals=[cube], collisions=[cube]))
    tree.add_edge(
        Edge(tree.root, "cube", FixedJoint(name="cf", origin=SE3(0.4, 0.0, 0.2)))
    )
    on_table = KinematicState.from_tree(tree, robot.home)
    tree.attach("cube", "tool_link", SE3(0.0, 0.0, 0.1))
    held = KinematicState.from_tree(tree, robot.home)
    tree.set_edge(
        Edge(tree.root, "cube", FixedJoint(name="cf2", origin=SE3(0.5, 0.0, 0.2)))
    )
    placed = KinematicState.from_tree(tree, robot.home)

    plan = [on_table, on_table, held, held, held, placed, placed]
    recorder = _RecordingRenderer(tree)
    frames = render_states(recorder, plan, tree)

    assert len(frames) == len(plan)
    assert recorder.batch_sizes == [2, 3, 2]  # one batch per structural segment
    assert recorder.parents_at_call == [tree.root, "tool_link", tree.root]


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
