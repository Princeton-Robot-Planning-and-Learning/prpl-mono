"""Unit tests for robot assemblies (composition over a KinematicTree)."""

import numpy as np
from spatialmath import SE3

from prpl_kinematics.collision import PyBulletCollisionChecker
from prpl_kinematics.geometry.shapes import BoxShape
from prpl_kinematics.ik import InverseKinematics
from prpl_kinematics.planning import BiRRTPlanner
from prpl_kinematics.robots import Robot, make_panda
from prpl_kinematics.tree.joints import FixedJoint
from prpl_kinematics.tree.kinematic_tree import Edge, Node


def test_panda_assembly_structure():
    """make_panda exposes named groups, an EE, an IK solver, and a valid home."""
    robot = make_panda()
    assert isinstance(robot, Robot)
    assert robot.groups["arm"].dimension == 7
    assert robot.groups["gripper"].dimension == 2
    assert robot.manipulators["arm"].ee_frame == "tool_link"
    assert isinstance(robot.manipulators["arm"].ik, InverseKinematics)
    assert robot.allowed_collision_pairs  # intrinsic rest overlaps discovered
    arm = robot.groups["arm"]
    home_vector = arm.to_vector(robot.home)
    assert np.allclose(home_vector, arm.clamp(home_vector))  # home within limits
    # The exact Franka Emika joint limits.
    assert robot.tree.joint("panda_joint1").lower_limits[0] == -2.8973
    assert robot.tree.joint("panda_joint4").upper_limits[0] == -0.0698


def test_panda_ik_through_robot():
    """The robot's injected IK solver reaches a reachable pose."""
    robot = make_panda()
    target = robot.tree.forward_kinematics(
        robot.manipulators["arm"].ee_frame, robot.home
    )
    seed = {**dict(robot.home), "panda_joint4": [-1.5]}
    solution = robot.manipulators["arm"].ik.solve(target, seed)
    assert solution is not None
    reached = robot.tree.forward_kinematics(
        robot.manipulators["arm"].ee_frame, solution
    )
    assert np.linalg.norm(np.asarray(reached.t) - np.asarray(target.t)) < 1e-6


def test_panda_home_is_self_collision_free(physics_client_id):
    """Home is collision-free once the robot's intrinsic ACM is allowed."""
    robot = make_panda()
    checker = PyBulletCollisionChecker(physics_client_id)
    checker.load(robot.tree)
    checker.ignore(robot.allowed_collision_pairs)
    assert not checker.in_collision(robot.home)


def test_panda_arm_plans_around_obstacle(physics_client_id):
    """A scene checker built from the robot's tree + ACM drives BiRRT on the arm."""
    robot = make_panda()
    block = BoxShape(size=(0.12, 0.12, 0.5))
    robot.tree.add_node(Node("obstacle", collisions=[block]))
    robot.tree.add_edge(
        Edge(
            robot.tree.root,
            "obstacle",
            FixedJoint(name="ofix", origin=SE3(0.45, 0.0, 0.5)),
        )
    )
    checker = PyBulletCollisionChecker(physics_client_id)
    checker.load(robot.tree)
    checker.ignore(robot.allowed_collision_pairs)
    assert not checker.in_collision(robot.home)
    start = robot.home
    goal = {**dict(start), "panda_joint1": [1.0]}
    planner = BiRRTPlanner(
        robot.groups["arm"],
        checker.in_collision,
        np.random.default_rng(0),
        num_iters=800,
    )
    path = planner.plan(start, goal)
    assert path is not None
    assert path[0] == start and path[-1] == goal
    assert all(not checker.in_collision(config) for config in path)
