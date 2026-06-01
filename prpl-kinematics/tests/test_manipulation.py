"""Unit tests for the Pick and Place manipulation primitives."""

import os

import numpy as np
import pytest
from spatialmath import SE3

from prpl_kinematics.collision import PyBulletCollisionChecker
from prpl_kinematics.geometry.shapes import BoxShape
from prpl_kinematics.manipulation import Handover, Pick, Place, Primitive
from prpl_kinematics.planning import BiRRTPlanner
from prpl_kinematics.robots import make_panda, make_vega
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


# A bimanual left->right handover on Vega: the giving (left) arm picks a bar near
# one end, carries it to a handover pose where it lies horizontal between the arms,
# the receiving (right) arm takes the *other* end, the left arm withdraws, and the
# right arm lays the bar flat on the other side of the table. Grasping opposite ends
# (and resting the jaws open) keeps the two grippers clear of each other -- their
# mutual collisions are NOT masked here, so the planner rejects any interpenetration.
# The geometry is tuned so the left arm reaches the centre without its elbow fouling
# the torso; the primitives still search their candidate lists.
_BAR_LENGTH = 0.22  # the grasped object is a long bar (its local z axis)
_GIVE_END = 0.07  # the left gripper grasps this far toward the +z end of the bar
_TAKE_END = -0.07  # the right gripper takes the opposite (-z) end
# The bar lies horizontal along world Y at the handover, so each arm keeps to its
# own side (left holds the +y end, right the -y end) instead of crossing over.
_HANDOVER_POSES = [SE3(0.66, 0.0, 0.80) * SE3.Rx(-np.pi / 2)]
_RECEIVE_GRASPS = [
    SE3.Rt((SE3.Ry(np.pi / 2) * SE3.Rz(np.pi / 2)).R, [0, 0, _TAKE_END]),
    SE3.Rt((SE3.Ry(np.pi / 2) * SE3.Rz(-np.pi / 2)).R, [0, 0, _TAKE_END]),
]


def _vega_handover_scene(physics_client_id):
    """A Vega scene with a wide table and a bar the left arm can pick by one end.

    Returns the robot, checker, a per-group planner factory, the start state, the left-
    arm pick grasp, and the table's top surface height.
    """
    robot = make_vega()
    robot_links = set(robot.tree.nodes)  # everything not added below is the robot
    left = robot.manipulators["left"]
    home_ee = robot.tree.forward_kinematics(left.ee_frame, robot.home)
    rotation = np.asarray(home_ee.R)
    # Roll 90 deg about the approach axis so the jaws straddle the bar (fingers on
    # opposite faces) rather than splaying along it; grasp near the +z end.
    rolled = (SE3.Rt(rotation, [0, 0, 0]) * SE3.Rz(np.pi / 2)).R
    pick_grasp = SE3.Rt(rolled, [0, 0, _GIVE_END])
    # Place the standing bar so the left arm's end grasp lands at its home EE pose.
    grasp_world = np.asarray(home_ee.t) + 0.20 * rotation[:, 2]
    bar_center = grasp_world - np.array([0, 0, _GIVE_END])
    table_top = bar_center[2] - _BAR_LENGTH / 2  # the bar stands on the table

    table = BoxShape(size=(0.34, 0.95, 0.02))
    robot.tree.add_node(Node("table", visuals=[table], collisions=[table]))
    robot.tree.add_edge(
        Edge(
            robot.tree.root,
            "table",
            FixedJoint(name="tf", origin=SE3(bar_center[0], 0.05, table_top - 0.01)),
        )
    )
    bar_shape = BoxShape(size=(0.05, 0.05, _BAR_LENGTH))
    robot.tree.add_node(Node("cube", visuals=[bar_shape], collisions=[bar_shape]))
    robot.tree.add_edge(
        Edge(robot.tree.root, "cube", FixedJoint(name="cf", origin=SE3(*bar_center)))
    )

    checker = PyBulletCollisionChecker(physics_client_id)
    checker.load(robot.tree)
    checker.ignore(robot.allowed_collision_pairs)
    checker.ignore([("cube", "table")])  # the bar rests on the table
    checker.ignore([(link, "cube") for link in robot_links])  # a gripper grasps it

    def planner(group):
        return BiRRTPlanner(
            robot.groups[group],
            checker.in_collision,
            np.random.default_rng(0),
            num_iters=2000,
        )

    state = KinematicState.from_tree(robot.tree, robot.home)
    return robot, checker, planner, state, pick_grasp, table_top


def test_handover_conforms_to_protocol(physics_client_id):
    """Handover satisfies the Primitive protocol."""
    robot, checker, planner, _, _, _ = _vega_handover_scene(physics_client_id)
    handover = Handover(
        robot,
        checker,
        planner("left_arm"),
        planner("right_arm"),
        "cube",
        _HANDOVER_POSES,
        _RECEIVE_GRASPS,
    )
    assert isinstance(handover, Primitive)


def test_handover_transfers_object_between_grippers(physics_client_id):
    """A handover flips the cube's parent edge from the left gripper to the right."""
    robot, checker, planner, state, grasp, _ = _vega_handover_scene(physics_client_id)
    picked = Pick(
        robot,
        checker,
        planner("left_arm"),
        "cube",
        "table",
        [grasp],
        manipulator="left",
        approach_distance=0.20,
    ).plan(state)
    assert picked is not None
    assert picked[-1].edges["cube"][0] == "L_ee"  # held by the left gripper
    handed = Handover(
        robot,
        checker,
        planner("left_arm"),
        planner("right_arm"),
        "cube",
        _HANDOVER_POSES,
        _RECEIVE_GRASPS,
        from_manipulator="left",
        to_manipulator="right",
        approach_distance=0.08,
    ).plan(picked[-1])
    assert handed is not None
    assert handed[0].edges["cube"][0] == "L_ee"  # starts on the left gripper
    assert handed[-1].edges["cube"][0] == "R_ee"  # ends on the right gripper


# Frames the bimanual scene: the action runs from the table top up to the raised
# handover, so the camera looks slightly down on the whole spread.
_PLAN_CAMERA = CameraParams(
    target=(0.60, 0.0, 0.60),
    distance=1.95,
    yaw=80.0,
    pitch=-16.0,
    fov=50.0,
    width=640,
    height=480,
)


def _pick_handover_place(robot, checker, planner, state, grasp, table_top):
    """The full left-pick -> handover -> right-place plan for the Vega scene."""
    pick = Pick(
        robot,
        checker,
        planner("left_arm"),
        "cube",
        "table",
        [grasp],
        manipulator="left",
        approach_distance=0.20,
    ).plan(state)
    assert pick is not None
    handover = Handover(
        robot,
        checker,
        planner("left_arm"),
        planner("right_arm"),
        "cube",
        _HANDOVER_POSES,
        _RECEIVE_GRASPS,
        from_manipulator="left",
        to_manipulator="right",
        approach_distance=0.08,
    ).plan(pick[-1])
    assert handover is not None
    # Lay the bar flat on the table's right side (long axis along world x).
    bar_rest_z = table_top + 0.025
    placements = [
        SE3(px, py, bar_rest_z) * SE3.Ry(np.pi / 2)
        for px in (0.60, 0.66)
        for py in (-0.16, -0.22)
    ]
    place = Place(
        robot,
        checker,
        planner("right_arm"),
        "cube",
        "table",
        placements,
        manipulator="right",
        approach_distance=0.12,
    ).plan(handover[-1])
    assert place is not None
    return pick + handover + place


def test_vega_pick_handover_place_video(
    physics_client_id, render_client_id, make_videos
):
    """With --make-videos: left picks, hands the bar to right, right places it."""
    if not make_videos:
        pytest.skip("pass --make-videos to render the video")
    robot, checker, planner, state, grasp, table_top = _vega_handover_scene(
        physics_client_id
    )
    plan = _pick_handover_place(robot, checker, planner, state, grasp, table_top)

    renderer = PyBulletRenderer(render_client_id)
    renderer.load(robot.tree)
    frames = []
    for plan_state in plan:
        renderer.render(plan_state.apply(robot.tree))
        frames.append(capture_image(render_client_id, _PLAN_CAMERA))
    save_video(frames, "vega_pick_handover_place.mp4", fps=20)
    assert os.path.exists("vega_pick_handover_place.mp4")
    # A high-fidelity Blender render of this same plan is produced by the standalone
    # scripts/render_vega_handover_blender.py (kept out of the test suite because the
    # Blender render takes several minutes).
