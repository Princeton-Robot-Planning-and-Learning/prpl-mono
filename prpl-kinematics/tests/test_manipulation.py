"""Unit tests for the Pick and Place manipulation primitives."""

import os

import numpy as np
import pytest
from spatialmath import SE3

from prpl_kinematics.collision import PyBulletCollisionChecker
from prpl_kinematics.geometry.shapes import BoxShape
from prpl_kinematics.manipulation import Pick, Place, Primitive
from prpl_kinematics.planning import BiRRTPlanner
from prpl_kinematics.robots import make_panda
from prpl_kinematics.tree.joints import FixedJoint
from prpl_kinematics.tree.kinematic_tree import Edge, Node
from prpl_kinematics.tree.state import KinematicState
from prpl_kinematics.visualization import (
    CameraParams,
    PyBulletRenderer,
    capture_image,
    save_video,
)

_GRASP = SE3.Rx(np.pi)  # top-down: gripper z points down at the cube
_PLACEMENT = SE3(0.5, 0.2, 0.25)  # move the cube 20 cm along +y


def _scene(physics_client_id):
    robot = make_panda()
    table = BoxShape(size=(0.4, 0.6, 0.02))
    robot.tree.add_node(Node("table", visuals=[table], collisions=[table]))
    robot.tree.add_edge(
        Edge(robot.tree.root, "table", FixedJoint(name="tf", origin=SE3(0.5, 0.0, 0.2)))
    )
    cube = BoxShape(size=(0.05, 0.05, 0.08))
    robot.tree.add_node(Node("cube", visuals=[cube], collisions=[cube]))
    robot.tree.add_edge(
        Edge(robot.tree.root, "cube", FixedJoint(name="cf", origin=SE3(0.5, 0.0, 0.25)))
    )
    checker = PyBulletCollisionChecker(physics_client_id)
    checker.load(robot.tree)
    checker.ignore(robot.allowed_collision_pairs)
    checker.ignore([("cube", "table")])  # the cube rests on the table
    # The gripper grasps (contacts) the cube; allow that contact.
    gripper = ["panda_hand", "panda_leftfinger", "panda_rightfinger"]
    checker.ignore([(link, "cube") for link in gripper])
    planner = BiRRTPlanner(
        robot.groups["arm"],
        checker.in_collision,
        np.random.default_rng(0),
        num_iters=1000,
    )
    state = KinematicState.from_tree(robot.tree, robot.home)
    return robot, checker, planner, state


def test_primitives_conform_to_protocol(physics_client_id):
    """Pick and Place satisfy the Primitive protocol."""
    robot, checker, planner, _ = _scene(physics_client_id)
    pick = Pick(robot, checker, planner, "cube", "table", [_GRASP])
    place = Place(robot, checker, planner, "cube", "table", [_PLACEMENT])
    assert isinstance(pick, Primitive) and isinstance(place, Primitive)


def test_pick_attaches_object_to_gripper(physics_client_id):
    """Picking re-parents the cube onto the gripper: its edge flips at the grasp."""
    robot, checker, planner, state = _scene(physics_client_id)
    plan = Pick(robot, checker, planner, "cube", "table", [_GRASP]).plan(state)
    assert plan is not None
    assert plan[0].edges["cube"][0] == robot.tree.root  # starts on the table/base
    assert plan[-1].edges["cube"][0] == "tool_link"  # ends held by the gripper


def test_place_releases_object_onto_surface(physics_client_id):
    """Placing detaches the held cube back to the base at the target pose."""
    robot, checker, planner, state = _scene(physics_client_id)
    picked = Pick(robot, checker, planner, "cube", "table", [_GRASP]).plan(state)
    assert picked is not None
    placed = Place(robot, checker, planner, "cube", "table", [_PLACEMENT]).plan(
        picked[-1]
    )
    assert placed is not None
    assert placed[-1].edges["cube"][0] == robot.tree.root
    cube_pose = robot.tree.forward_kinematics("cube", placed[-1].apply(robot.tree))
    assert np.allclose(np.asarray(cube_pose.t), [0.5, 0.2, 0.25], atol=1e-6)


def test_pick_and_place_video(physics_client_id, render_client_id, make_videos):
    """With --make-videos: render the cube being picked and placed."""
    if not make_videos:
        pytest.skip("pass --make-videos to render the video")
    robot, checker, planner, state = _scene(physics_client_id)
    pick_plan = Pick(robot, checker, planner, "cube", "table", [_GRASP]).plan(state)
    assert pick_plan is not None
    place_plan = Place(robot, checker, planner, "cube", "table", [_PLACEMENT]).plan(
        pick_plan[-1]
    )
    assert place_plan is not None

    renderer = PyBulletRenderer(render_client_id)
    renderer.load(robot.tree)
    camera = CameraParams(target=(0.5, 0.1, 0.3), distance=1.4, yaw=55.0, pitch=-25.0)
    # Apply each state's edges (so the grasp follows the gripper) then render.
    frames = []
    for plan_state in pick_plan + place_plan:
        renderer.render(plan_state.apply(robot.tree))
        frames.append(capture_image(render_client_id, camera))
    save_video(frames, "panda_pick_place.mp4", fps=20)
    assert os.path.exists("panda_pick_place.mp4")
