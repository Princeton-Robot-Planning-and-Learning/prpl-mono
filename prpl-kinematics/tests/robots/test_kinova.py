"""Unit tests for the Kinova Gen3 robot assembly.

The Gen3 has continuous joints (1, 3, 5, 7), so it exercises JointSpace's wrap-around
handling. The kortex IKFast module is compiled on demand the first time the solver is
built (needs a C++ toolchain and LAPACK/BLAS, present in CI).
"""

import math

import numpy as np

from prpl_kinematics.collision import PyBulletCollisionChecker
from prpl_kinematics.ik import InverseKinematics
from prpl_kinematics.robots import Robot, make_kinova

ARM = [f"joint_{i}" for i in range(1, 8)]


def test_kinova_assembly_structure():
    """make_kinova exposes a 7-DOF arm with the expected continuous joints."""
    robot = make_kinova()
    assert isinstance(robot, Robot)
    assert robot.name == "kinova-gen3"
    assert robot.groups["arm"].dimension == 7
    assert robot.manipulators["arm"].ee_frame == "tool_frame"
    assert isinstance(robot.manipulators["arm"].ik, InverseKinematics)
    assert robot.allowed_collision_pairs
    # Joints 1, 3, 5, 7 are continuous (unlimited); 2, 4, 6 are limited.
    for i in (1, 3, 5, 7):
        assert not math.isfinite(robot.tree.joint(f"joint_{i}").upper_limits[0])
    for i in (2, 4, 6):
        assert math.isfinite(robot.tree.joint(f"joint_{i}").upper_limits[0])


def test_kinova_ik_through_robot():
    """The robot's IKFast solver reaches a reachable pose from the home seed."""
    robot = make_kinova()
    truth = {
        name: [value]
        for name, value in zip(ARM, [0.3, -0.5, 1.2, -1.0, 0.4, 0.8, -0.6])
    }
    target = robot.tree.forward_kinematics(robot.manipulators["arm"].ee_frame, truth)
    solution = robot.manipulators["arm"].ik.solve(target, robot.home)
    assert solution is not None
    reached = robot.tree.forward_kinematics(
        robot.manipulators["arm"].ee_frame, solution
    )
    assert np.linalg.norm(np.asarray(reached.t) - np.asarray(target.t)) < 1e-4


def test_kinova_home_is_self_collision_free(physics_client_id):
    """Home is collision-free once the robot's intrinsic ACM is allowed."""
    robot = make_kinova()
    checker = PyBulletCollisionChecker(physics_client_id)
    checker.load(robot.tree)
    checker.ignore(robot.allowed_collision_pairs)
    assert not checker.in_collision(robot.home)
