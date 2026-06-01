"""Backend-agnostic rendering interface tests."""

import numpy as np
from spatialmath import SE3

from prpl_kinematics.geometry.shapes import BoxShape
from prpl_kinematics.robots import make_panda
from prpl_kinematics.tree.joints import FixedJoint
from prpl_kinematics.tree.kinematic_tree import Configuration, Edge, KinematicTree, Node
from prpl_kinematics.tree.state import KinematicState
from prpl_kinematics.visualization import (
    BlenderRenderer,
    CameraParams,
    PyBulletRenderer,
    Renderer,
    render_states,
)


def test_renderers_conform_to_interface(physics_client_id):
    """Both backends are Renderers (the protocol the helpers consume)."""
    assert isinstance(PyBulletRenderer(physics_client_id), Renderer)
    assert isinstance(BlenderRenderer(blender_executable="/usr/bin/false"), Renderer)


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
