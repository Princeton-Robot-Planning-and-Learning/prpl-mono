"""Unit tests for OMPL-backed motion planning."""

import numpy as np
from spatialmath import SE3

from prpl_kinematics.collision import PyBulletCollisionChecker
from prpl_kinematics.geometry.shapes import BoxShape
from prpl_kinematics.planning import (
    BiRRTPlanner,
    JointSpace,
    MotionPlanner,
    OMPLPlanner,
)
from prpl_kinematics.tree.joints import FixedJoint, PrismaticJoint
from prpl_kinematics.tree.kinematic_tree import Edge, KinematicTree, Node


def _gantry_tree() -> KinematicTree:
    """An XY gantry box that must steer around a central block obstacle."""
    tree = KinematicTree()
    tree.add_node(Node("jx"))
    tree.add_node(Node("robot", collisions=[BoxShape(size=(0.2, 0.2, 0.2))]))
    tree.add_node(Node("obstacle", collisions=[BoxShape(size=(2.0, 2.0, 2.0))]))
    tree.add_edge(
        Edge(
            "world", "jx", PrismaticJoint(name="jx", axis=(1, 0, 0), lower=-1, upper=5)
        )
    )
    tree.add_edge(
        Edge(
            "jx", "robot", PrismaticJoint(name="jy", axis=(0, 1, 0), lower=-1, upper=5)
        )
    )
    tree.add_edge(
        Edge("world", "obstacle", FixedJoint(name="ofix", origin=SE3(2.5, 2.5, 0)))
    )
    return tree


def test_planners_conform_to_motion_planner(physics_client_id):
    """Both BiRRTPlanner and OMPLPlanner satisfy the MotionPlanner protocol."""
    checker = PyBulletCollisionChecker(physics_client_id)
    checker.load(_gantry_tree())
    space = JointSpace(_gantry_tree(), ["jx", "jy"])
    rng = np.random.default_rng(0)
    assert isinstance(OMPLPlanner(space, checker.in_collision, rng), MotionPlanner)
    assert isinstance(BiRRTPlanner(space, checker.in_collision, rng), MotionPlanner)


def test_ompl_solves_around_obstacle(physics_client_id):
    """OMPL finds a collision-free path around the blocking obstacle."""
    tree = _gantry_tree()
    checker = PyBulletCollisionChecker(physics_client_id)
    checker.load(tree)
    space = JointSpace(tree, ["jx", "jy"])
    start = {"jx": [0.0], "jy": [0.0]}
    goal = {"jx": [5.0], "jy": [5.0]}
    planner = OMPLPlanner(space, checker.in_collision, np.random.default_rng(0))
    path = planner.plan(start, goal)
    assert path is not None
    assert path[0] == start and path[-1] == goal
    assert all(not checker.in_collision(config) for config in path)


def test_ompl_returns_none_when_start_in_collision(physics_client_id):
    """Planning from a colliding start yields no path."""
    tree = _gantry_tree()
    checker = PyBulletCollisionChecker(physics_client_id)
    checker.load(tree)
    space = JointSpace(tree, ["jx", "jy"])
    planner = OMPLPlanner(
        space, checker.in_collision, np.random.default_rng(0), timeout=1.0
    )
    assert planner.plan({"jx": [2.5], "jy": [2.5]}, {"jx": [5.0], "jy": [5.0]}) is None
