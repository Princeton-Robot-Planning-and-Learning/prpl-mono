"""Unit tests for the bimanual Dexmate Vega 1U robot.

Vega is the multi-manipulator case: two arm groups, each with its own
``VegaArmIK`` solver (EAIK with two joints locked and searched). The IK matches
the implementation in pybullet-helpers.
"""

import os

import numpy as np
import pytest

from prpl_kinematics.collision import PyBulletCollisionChecker
from prpl_kinematics.ik import InverseKinematics
from prpl_kinematics.robots import Robot, make_vega
from prpl_kinematics.visualization import (
    CameraParams,
    PyBulletRenderer,
    render_configurations,
    save_video,
)

_LEFT = [f"L_arm_j{i}" for i in range(1, 8)]
_RIGHT = [f"R_arm_j{i}" for i in range(1, 8)]


def test_vega_assembly_structure():
    """Vega exposes two 7-DOF arm groups and two manipulators."""
    robot = make_vega()
    assert isinstance(robot, Robot)
    assert robot.name == "vega-1u"
    assert robot.groups["left_arm"].dimension == 7
    assert robot.groups["right_arm"].dimension == 7
    assert robot.groups["left_gripper"].dimension == 2
    assert robot.groups["right_gripper"].dimension == 2
    assert set(robot.manipulators) == {"left", "right"}
    assert robot.manipulators["left"].ee_frame == "L_ee"
    assert robot.manipulators["right"].ee_frame == "R_ee"
    assert isinstance(robot.manipulators["left"].ik, InverseKinematics)
    assert robot.allowed_collision_pairs


def test_vega_both_arms_solve_ik():
    """Each arm's EAIK solver reaches a reachable pose."""
    robot = make_vega()
    for side, joints in [("left", _LEFT), ("right", _RIGHT)]:
        manipulator = robot.manipulators[side]
        truth = {
            **dict(robot.home),
            **{n: [v] for n, v in zip(joints, [0.3, -0.4, 0.2, -1.0, 0.1, 0.6, -0.3])},
        }
        target = robot.tree.forward_kinematics(manipulator.ee_frame, truth)
        solution = manipulator.ik.solve(target, robot.home)
        assert solution is not None
        reached = robot.tree.forward_kinematics(manipulator.ee_frame, solution)
        assert np.linalg.norm(np.asarray(reached.t) - np.asarray(target.t)) < 1e-3


def test_vega_home_is_self_collision_free(physics_client_id):
    """Home is collision-free once the robot's intrinsic ACM is allowed."""
    robot = make_vega()
    checker = PyBulletCollisionChecker(physics_client_id)
    checker.load(robot.tree)
    checker.ignore(robot.allowed_collision_pairs)
    assert not checker.in_collision(robot.home)


def test_vega_bimanual_reach_video(physics_client_id, make_videos):
    """With --make-videos: both arms reach IK-solved targets simultaneously."""
    if not make_videos:
        pytest.skip("pass --make-videos to render the video")
    robot = make_vega()
    targets = {
        "left": [0.3, -0.4, 0.2, -1.0, 0.1, 0.6, -0.3],
        "right": [-0.3, 0.4, -0.2, -1.0, -0.1, 0.6, 0.3],
    }
    solutions = {}
    for side, joints in [("left", _LEFT), ("right", _RIGHT)]:
        manipulator = robot.manipulators[side]
        truth = {**dict(robot.home), **{n: [v] for n, v in zip(joints, targets[side])}}
        target = robot.tree.forward_kinematics(manipulator.ee_frame, truth)
        solution = manipulator.ik.solve(target, robot.home)
        assert solution is not None
        solutions[side] = solution

    left_space, right_space = robot.groups["left_arm"], robot.groups["right_arm"]
    home_l, home_r = left_space.to_vector(robot.home), right_space.to_vector(robot.home)
    goal_l, goal_r = left_space.to_vector(solutions["left"]), right_space.to_vector(
        solutions["right"]
    )
    steps = 40
    configs = []
    for k in range(steps + 1):
        fraction = k / steps
        configs.append(
            {
                **dict(robot.home),
                **left_space.to_configuration(home_l + (goal_l - home_l) * fraction),
                **right_space.to_configuration(home_r + (goal_r - home_r) * fraction),
            }
        )

    renderer = PyBulletRenderer(physics_client_id)
    renderer.load(robot.tree)
    camera = CameraParams(
        target=(0.4, 0.0, 1.0),
        distance=2.4,
        yaw=50.0,
        pitch=-12.0,
        width=640,
        height=480,
    )
    frames = render_configurations(renderer, configs, camera)
    save_video(frames, "vega_bimanual.mp4", fps=20)
    assert os.path.exists("vega_bimanual.mp4")
