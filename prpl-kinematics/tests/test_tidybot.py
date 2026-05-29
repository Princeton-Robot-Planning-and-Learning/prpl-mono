"""Unit tests for TidyBot, a Kinova Gen3 on an SE(2) mobile base.

TidyBot exercises mobile manipulation: an ``SE2Space`` base group and a
``JointSpace`` arm group, planned by the same ``BiRRTPlanner``. The kortex
IKFast module compiles on first use (needs a C++ toolchain and LAPACK/BLAS,
present in CI).
"""

import os

import numpy as np
import pytest
from spatialmath import SE3

from prpl_kinematics.collision import PyBulletCollisionChecker
from prpl_kinematics.geometry.shapes import BoxShape
from prpl_kinematics.ik import InverseKinematics
from prpl_kinematics.planning import BiRRTPlanner, SE2Space
from prpl_kinematics.robots import Robot, make_tidybot
from prpl_kinematics.tree.joints import FixedJoint, PlanarJoint
from prpl_kinematics.tree.kinematic_tree import Configuration, Edge, Node
from prpl_kinematics.visualization import (
    CameraParams,
    PyBulletRenderer,
    render_configurations,
    save_video,
)

_ARM = [f"joint_{i}" for i in range(1, 8)]


def _pillar(origin: SE3, size=(0.3, 0.3, 1.2)) -> tuple[Node, Edge]:
    block = BoxShape(size=size)
    node = Node("pillar", visuals=[block], collisions=[block])
    return node, Edge("world", "pillar", FixedJoint(name="pf", origin=origin))


def test_tidybot_assembly_structure():
    """TidyBot exposes an SE(2) base group and a 7-DOF arm group."""
    robot = make_tidybot()
    assert isinstance(robot, Robot)
    assert robot.name == "tidybot"
    assert isinstance(robot.groups["base"], SE2Space)
    assert robot.groups["base"].dimension == 3
    assert robot.groups["arm"].dimension == 7
    assert isinstance(robot.tree.joint("base"), PlanarJoint)
    assert robot.manipulators["arm"].ee_frame == "tool_frame"
    assert isinstance(robot.manipulators["arm"].ik, InverseKinematics)
    assert robot.allowed_collision_pairs


def test_tidybot_base_moves_whole_arm():
    """Driving the base translates the end effector by the same amount."""
    robot = make_tidybot()
    home_ee = robot.tree.forward_kinematics(
        robot.manipulators["arm"].ee_frame, robot.home
    ).t
    moved = {**dict(robot.home), "base": [1.0, 0.5, 0.0]}
    moved_ee = robot.tree.forward_kinematics(
        robot.manipulators["arm"].ee_frame, moved
    ).t
    assert np.allclose(
        np.asarray(moved_ee) - np.asarray(home_ee), [1.0, 0.5, 0.0], atol=1e-6
    )


def test_tidybot_base_plans_around_obstacle(physics_client_id):
    """BiRRT plans the SE(2) base around a floor pillar that blocks the straight
    line."""
    robot = make_tidybot(base_bounds=(-1.0, 3.0))
    node, edge = _pillar(SE3(0.9, 0.0, 0.6))
    robot.tree.add_node(node)
    robot.tree.add_edge(edge)
    checker = PyBulletCollisionChecker(physics_client_id)
    checker.load(robot.tree)
    checker.ignore(robot.allowed_collision_pairs)
    start = robot.home
    goal = {**dict(start), "base": [1.8, 0.0, 0.0]}
    assert not checker.in_collision(start) and not checker.in_collision(goal)
    planner = BiRRTPlanner(
        robot.groups["base"],
        checker.in_collision,
        np.random.default_rng(0),
        num_iters=800,
    )
    path = planner.plan(start, goal)
    assert path is not None
    assert all(not checker.in_collision(config) for config in path)


def test_tidybot_arm_ik_through_robot():
    """The robot's IK solver reaches a pose with the base at home."""
    robot = make_tidybot()
    truth = {
        **dict(robot.home),
        **{name: [v] for name, v in zip(_ARM, [0.3, -0.5, 1.2, -1.0, 0.4, 0.8, -0.6])},
    }
    target = robot.tree.forward_kinematics(robot.manipulators["arm"].ee_frame, truth)
    solution = robot.manipulators["arm"].ik.solve(target, robot.home)
    assert solution is not None
    reached = robot.tree.forward_kinematics(
        robot.manipulators["arm"].ee_frame, solution
    )
    assert np.linalg.norm(np.asarray(reached.t) - np.asarray(target.t)) < 1e-4


def test_tidybot_base_then_arm_video(physics_client_id, render_client_id, make_videos):
    """With --make-videos: drive the base around a pillar, then the arm around a box."""
    if not make_videos:
        pytest.skip("pass --make-videos to render the video")
    robot = make_tidybot(base_bounds=(-1.0, 3.0))
    arm = robot.groups["arm"]
    base_goal = [1.8, 0.0, 0.0]
    at_goal: Configuration = {**dict(robot.home), "base": base_goal}
    arm_goal = {**dict(at_goal), "joint_1": [at_goal["joint_1"][0] + 2.0]}

    # A floor pillar the base must steer around, and a box on the arm's sweep.
    pillar_node, pillar_edge = _pillar(SE3(0.9, 0.0, 0.6))
    robot.tree.add_node(pillar_node)
    robot.tree.add_edge(pillar_edge)
    mid = {**dict(at_goal), "joint_1": [at_goal["joint_1"][0] + 1.0]}
    box_at = np.asarray(
        robot.tree.forward_kinematics(robot.manipulators["arm"].ee_frame, mid).t
    )
    armbox = BoxShape(size=(0.1, 0.1, 0.4))
    robot.tree.add_node(Node("armbox", visuals=[armbox], collisions=[armbox]))
    robot.tree.add_edge(
        Edge("world", "armbox", FixedJoint(name="af", origin=SE3(*box_at)))
    )

    checker = PyBulletCollisionChecker(physics_client_id)
    checker.load(robot.tree)
    checker.ignore(robot.allowed_collision_pairs)
    base_path = BiRRTPlanner(
        robot.groups["base"],
        checker.in_collision,
        np.random.default_rng(0),
        num_iters=800,
    ).plan(robot.home, at_goal)
    arm_path = BiRRTPlanner(
        arm, checker.in_collision, np.random.default_rng(0), num_iters=1200
    ).plan(at_goal, arm_goal)
    assert base_path is not None and arm_path is not None

    renderer = PyBulletRenderer(render_client_id)
    renderer.load(robot.tree)
    camera = CameraParams(
        target=(1.0, 0.0, 0.5),
        distance=3.4,
        yaw=55.0,
        pitch=-28.0,
        width=640,
        height=480,
    )
    frames = render_configurations(renderer, base_path + arm_path, camera)
    save_video(frames, "tidybot_base_then_arm.mp4", fps=20)
    assert os.path.exists("tidybot_base_then_arm.mp4")
