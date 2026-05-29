"""Unit tests for joint-space BiRRT motion planning."""

import math
import os

import numpy as np
import pytest
from spatialmath import SE3

from prpl_kinematics.collision import PyBulletCollisionChecker
from prpl_kinematics.geometry.shapes import BoxShape
from prpl_kinematics.loading import load_urdf
from prpl_kinematics.planning import (
    BiRRTPlanner,
    ConfigurationSpace,
    JointSpace,
    SE2Space,
)
from prpl_kinematics.tree.joints import FixedJoint, PrismaticJoint, RevoluteJoint
from prpl_kinematics.tree.kinematic_tree import Edge, KinematicTree, Node
from prpl_kinematics.utils import get_assets_path
from prpl_kinematics.visualization import (
    CameraParams,
    PyBulletRenderer,
    render_configurations,
    save_video,
)


def _gantry_tree() -> KinematicTree:
    """An XY gantry: a small box robot that slides in x then y, plus a central
    block obstacle it must steer around."""
    tree = KinematicTree()
    tree.add_node(Node("jx"))
    tree.add_node(Node("robot", collisions=[BoxShape(size=(0.2, 0.2, 0.2))]))
    tree.add_node(Node("obstacle", collisions=[BoxShape(size=(2.0, 2.0, 2.0))]))
    tree.add_edge(
        Edge(
            "world",
            "jx",
            PrismaticJoint(name="jx_joint", axis=(1, 0, 0), lower=-1, upper=5),
        )
    )
    tree.add_edge(
        Edge(
            "jx",
            "robot",
            PrismaticJoint(name="jy_joint", axis=(0, 1, 0), lower=-1, upper=5),
        )
    )
    tree.add_edge(
        Edge("world", "obstacle", FixedJoint(name="ofix", origin=SE3(2.5, 2.5, 0)))
    )
    return tree


def test_joint_space_geometry():
    """A JointSpace samples within bounds and converts vectors round-trip."""
    space = JointSpace(_gantry_tree(), ["jx_joint", "jy_joint"])
    assert space.dimension == 2
    rng = np.random.default_rng(0)
    for _ in range(50):
        sample = space.sample(rng)
        assert np.all(sample >= -1) and np.all(sample <= 5)
    config = {"jx_joint": [1.5], "jy_joint": [-0.5]}
    assert space.to_configuration(space.to_vector(config)) == config
    assert space.distance(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == pytest.approx(
        5.0
    )
    assert np.allclose(space.clamp(np.array([-3.0, 7.0])), [-1.0, 5.0])


def _continuous_space() -> JointSpace:
    tree = KinematicTree()
    tree.add_node(Node("a"))
    tree.add_edge(
        Edge("world", "a", RevoluteJoint(name="cont", lower=-math.inf, upper=math.inf))
    )
    return JointSpace(tree, ["cont"])


def test_continuous_joint_distance_wraps_around():
    """A continuous joint measures the shorter way around 2*pi."""
    space = _continuous_space()
    assert space.distance(np.array([3.0]), np.array([-3.0])) == pytest.approx(
        2 * math.pi - 6.0
    )


def test_continuous_joint_samples_within_pi():
    """A continuous joint (infinite limits) samples over [-pi, pi]."""
    space = _continuous_space()
    rng = np.random.default_rng(0)
    for _ in range(50):
        value = space.sample(rng)[0]
        assert -math.pi <= value <= math.pi


def test_continuous_joint_interpolates_short_way():
    """Interpolation crosses the +-pi seam instead of unwinding the long way."""
    space = _continuous_space()
    waypoints = [
        w[0] for w in space.interpolate(np.array([3.0]), np.array([-3.0]), 0.1)
    ]
    steps = np.diff([3.0] + waypoints)
    assert np.all(np.abs(steps) <= 0.1 + 1e-9)
    assert (waypoints[-1] - (-3.0)) % (2 * math.pi) == pytest.approx(0.0, abs=1e-9)


def test_finite_revolute_distance_is_euclidean():
    """A limited revolute joint does not wrap; distance stays Euclidean."""
    tree = KinematicTree()
    tree.add_node(Node("a"))
    tree.add_edge(
        Edge("world", "a", RevoluteJoint(name="r", lower=-math.pi, upper=math.pi))
    )
    space = JointSpace(tree, ["r"])
    assert space.distance(np.array([3.0]), np.array([-3.0])) == pytest.approx(6.0)


def test_joint_and_se2_spaces_conform_to_protocol():
    """Both space types satisfy the ConfigurationSpace protocol."""
    assert isinstance(_continuous_space(), ConfigurationSpace)
    assert isinstance(SE2Space("base", (-1, 1), (-1, 1)), ConfigurationSpace)


def test_se2_space_sampling_and_distance():
    """SE2Space samples within the box and measures yaw the short way around."""
    space = SE2Space("base", (-2.0, 2.0), (-1.0, 1.0))
    assert space.dimension == 3
    rng = np.random.default_rng(0)
    for _ in range(50):
        x, y, yaw = space.sample(rng)
        assert -2.0 <= x <= 2.0 and -1.0 <= y <= 1.0 and -math.pi <= yaw <= math.pi
    config = {"base": [1.0, 0.5, 0.3]}
    assert space.to_configuration(space.to_vector(config)) == config
    # Pure translation is Euclidean; pure yaw wraps the short way.
    assert space.distance(
        np.array([0, 0, 0.0]), np.array([3, 4, 0.0])
    ) == pytest.approx(5.0)
    assert space.distance(
        np.array([0, 0, 3.0]), np.array([0, 0, -3.0])
    ) == pytest.approx(2 * math.pi - 6.0)
    assert np.allclose(space.clamp(np.array([5.0, -5.0, 0.0]))[:2], [2.0, -1.0])


def test_se2_space_interpolates_yaw_short_way():
    """SE2 interpolation crosses the +-pi seam in yaw."""
    space = SE2Space("base", (-5.0, 5.0), (-5.0, 5.0))
    yaws = [
        w[2]
        for w in space.interpolate(np.array([0, 0, 3.0]), np.array([0, 0, -3.0]), 0.1)
    ]
    assert np.all(np.abs(np.diff([3.0] + yaws)) <= 0.1 + 1e-9)
    assert (yaws[-1] - (-3.0)) % (2 * math.pi) == pytest.approx(0.0, abs=1e-9)


def test_interpolate_resolution_and_endpoint():
    """Interpolation steps stay within resolution and end exactly at the target."""
    space = JointSpace(_gantry_tree(), ["jx_joint", "jy_joint"])
    a, b = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    waypoints = list(space.interpolate(a, b, resolution=0.1))
    assert len(waypoints) == 10
    assert np.allclose(waypoints[-1], b)
    steps = np.diff([a] + waypoints, axis=0)
    assert np.all(np.linalg.norm(steps, axis=1) <= 0.1 + 1e-9)


def test_birrt_solves_around_obstacle(physics_client_id):
    """BiRRT finds a collision-free path when the straight line is blocked."""
    tree = _gantry_tree()
    checker = PyBulletCollisionChecker(physics_client_id)
    checker.load(tree)
    space = JointSpace(tree, ["jx_joint", "jy_joint"])
    start = {"jx_joint": [0.0], "jy_joint": [0.0]}
    goal = {"jx_joint": [5.0], "jy_joint": [5.0]}
    # The straight-line interpolation passes through the central obstacle.
    direct = space.interpolate(space.to_vector(start), space.to_vector(goal), 0.05)
    assert any(
        checker.in_collision({**start, **space.to_configuration(v)}) for v in direct
    )
    planner = BiRRTPlanner(
        space, checker.in_collision, np.random.default_rng(0), num_iters=500
    )
    path = planner.plan(start, goal)
    assert path is not None
    assert path[0] == start and path[-1] == goal
    assert all(not checker.in_collision(config) for config in path)


def test_birrt_returns_none_when_start_in_collision(physics_client_id):
    """Planning from a colliding configuration yields no path."""
    tree = _gantry_tree()
    checker = PyBulletCollisionChecker(physics_client_id)
    checker.load(tree)
    space = JointSpace(tree, ["jx_joint", "jy_joint"])
    inside = {"jx_joint": [2.5], "jy_joint": [2.5]}
    goal = {"jx_joint": [5.0], "jy_joint": [5.0]}
    planner = BiRRTPlanner(space, checker.in_collision, np.random.default_rng(0))
    assert planner.plan(inside, goal) is None


def _panda_around_obstacle():
    """A Panda with a block obstacle placed on the arm's straight-line sweep."""
    path = str(get_assets_path() / "urdf" / "panda_arm_hand.urdf")
    tree = load_urdf(path)
    block = BoxShape(size=(0.12, 0.12, 0.5))
    tree.add_node(Node("obstacle", visuals=[block], collisions=[block]))
    tree.add_edge(
        Edge(
            tree.root, "obstacle", FixedJoint(name="ofix", origin=SE3(0.2, 0.21, 0.82))
        )
    )
    arm = [f"panda_joint{i}" for i in range(1, 8)]
    start = {name: [0.0] for name in tree.actuated_joint_names()}
    start["panda_joint2"] = [-0.5]
    start["panda_joint4"] = [-1.5]
    start["panda_joint6"] = [1.0]
    goal = dict(start)
    goal["panda_joint1"] = [1.6]
    return tree, arm, start, goal


def test_birrt_plans_panda_around_obstacle(
    physics_client_id, render_client_id, make_videos
):
    """BiRRT steers the Panda's arm around a block; --make-videos renders it."""
    tree, arm, start, goal = _panda_around_obstacle()
    checker = PyBulletCollisionChecker(physics_client_id)
    checker.load(tree)
    # The robot's rest-overlapping pairs are intrinsic to the robot; discovering
    # them must not absorb any arm-vs-obstacle overlap, or the obstacle would be
    # silently ignored for the rest of planning.
    allowed = {
        pair for pair in checker.pairs_in_collision(start) if "obstacle" not in pair
    }
    checker.ignore(allowed)
    assert not checker.in_collision(start)
    assert not checker.in_collision(goal)
    space = JointSpace(tree, arm)
    margin = 0.01

    def collision_with_margin(config):
        return checker.in_collision(config, max_distance=margin)

    planner = BiRRTPlanner(
        space, collision_with_margin, np.random.default_rng(0), num_iters=1500
    )
    path = planner.plan(start, goal)
    assert path is not None
    assert all(not checker.in_collision(config) for config in path)
    if make_videos:
        renderer = PyBulletRenderer(render_client_id)
        renderer.load(tree)
        camera = CameraParams(
            target=(0.1, 0.12, 0.8), distance=1.4, yaw=180.0, pitch=-10.0
        )
        frames = render_configurations(renderer, path, camera)
        save_video(frames, "panda_birrt.mp4", fps=20)
        assert os.path.exists("panda_birrt.mp4")
