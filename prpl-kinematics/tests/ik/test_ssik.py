"""Unit tests for the ssik-backed analytic IK solver.

ssik is an optional dependency, so the whole module is skipped when it is not installed.
It solves Vega's non-SRS 7R arm -- the case VegaArmIK works around -- analytically (with
a Newton polish), matching the tree's forward kinematics.
"""

import numpy as np
import pytest
from spatialmath import SE3

pytest.importorskip("ssik")

# Imports follow the optional-dependency skip guard above.
# pylint: disable=wrong-import-position
from prpl_kinematics.ik import InverseKinematics, SSIKSolver
from prpl_kinematics.robots import make_vega

_LEFT = [f"L_arm_j{i}" for i in range(1, 8)]
_RIGHT = [f"R_arm_j{i}" for i in range(1, 8)]
_TRUTH = [0.3, -0.4, 0.2, -1.0, 0.1, 0.6, -0.3]


def test_ssik_solver_conforms_to_protocol():
    """SSIKSolver satisfies the InverseKinematics protocol."""
    robot = make_vega(ik="ssik")
    assert isinstance(robot.manipulators["left"].ik, SSIKSolver)
    assert isinstance(robot.manipulators["left"].ik, InverseKinematics)


def test_ssik_solves_both_vega_arms():
    """Each arm's ssik solver reaches a reachable pose to machine precision."""
    robot = make_vega(ik="ssik")
    for side, joints in [("left", _LEFT), ("right", _RIGHT)]:
        manipulator = robot.manipulators[side]
        truth = {**dict(robot.home), **{n: [v] for n, v in zip(joints, _TRUTH)}}
        target = robot.tree.forward_kinematics(manipulator.ee_frame, truth)
        solution = manipulator.ik.solve(target, robot.home)
        assert solution is not None
        reached = robot.tree.forward_kinematics(manipulator.ee_frame, solution)
        assert np.linalg.norm(np.asarray(reached.t) - np.asarray(target.t)) < 1e-6


def test_ssik_solves_top_down_grasp():
    """A top-down grasp (the case that stresses EAIK) solves cleanly."""
    robot = make_vega(ik="ssik")
    manipulator = robot.manipulators["left"]
    target = SE3.Rt(SE3.Rx(np.pi).R, [0.35, 0.30, 0.85])  # gripper z-axis down
    solution = manipulator.ik.solve(target, robot.home)
    assert solution is not None
    reached = robot.tree.forward_kinematics(manipulator.ee_frame, solution)
    assert np.linalg.norm(np.asarray(reached.t) - np.asarray(target.t)) < 1e-2


def test_ssik_solution_respects_joint_limits():
    """The returned branch lies within the arm's joint limits."""
    robot = make_vega(ik="ssik")
    manipulator = robot.manipulators["left"]
    truth = {**dict(robot.home), **{n: [v] for n, v in zip(_LEFT, _TRUTH)}}
    target = robot.tree.forward_kinematics(manipulator.ee_frame, truth)
    solution = manipulator.ik.solve(target, robot.home)
    assert solution is not None
    solved = np.array([solution[n][0] for n in _LEFT])
    lower = np.array([robot.tree.joint(n).lower_limits[0] for n in _LEFT])
    upper = np.array([robot.tree.joint(n).upper_limits[0] for n in _LEFT])
    assert np.all(solved >= lower - 1e-6) and np.all(solved <= upper + 1e-6)


def test_ssik_unreachable_target_returns_none():
    """A pose far outside the workspace yields no solution rather than raising."""
    robot = make_vega(ik="ssik")
    manipulator = robot.manipulators["left"]
    target = SE3(5.0, 5.0, 5.0)  # well outside the arm's reach
    assert manipulator.ik.solve(target, robot.home) is None
