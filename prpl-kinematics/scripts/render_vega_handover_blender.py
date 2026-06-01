"""Render the Vega pick -> handover -> place plan through Blender (high fidelity).

This is a standalone render script, deliberately NOT a pytest test: the Blender
render takes several minutes, so it must never run as part of the test suite / CI.
Run it directly to (re)generate ``vega_pick_handover_place_blender.mp4``::

    python scripts/render_vega_handover_blender.py

The scene and plan mirror ``tests/test_manipulation.py`` (the left arm picks a bar
by one end, hands the far end to the right arm, which lays it flat on the table);
the fast PyBullet preview of the same plan lives there as a ``--make-videos`` test.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pybullet as p
from spatialmath import SE3

from prpl_kinematics.collision import PyBulletCollisionChecker
from prpl_kinematics.geometry.shapes import BoxShape
from prpl_kinematics.manipulation import Handover, Pick, Place
from prpl_kinematics.planning import BiRRTPlanner
from prpl_kinematics.robots import Robot, make_vega
from prpl_kinematics.tree.joints import FixedJoint
from prpl_kinematics.tree.kinematic_tree import Edge, Node
from prpl_kinematics.tree.state import KinematicState
from prpl_kinematics.visualization import (
    BlenderRenderer,
    CameraParams,
    render_states,
    save_video,
)

_PlannerFactory = Callable[[str], BiRRTPlanner]
_Scene = tuple[
    Robot, PyBulletCollisionChecker, _PlannerFactory, KinematicState, SE3, float
]

_BAR_LENGTH = 0.22  # the grasped object is a long bar (its local z axis)
_GIVE_END = 0.07  # the left gripper grasps this far toward the +z end of the bar
_TAKE_END = -0.07  # the right gripper takes the opposite (-z) end
_HANDOVER_POSES = [SE3(0.66, 0.0, 0.80) * SE3.Rx(-np.pi / 2)]  # bar horizontal, world Y
_RECEIVE_GRASPS = [
    SE3.Rt((SE3.Ry(np.pi / 2) * SE3.Rz(np.pi / 2)).R, [0, 0, _TAKE_END]),
    SE3.Rt((SE3.Ry(np.pi / 2) * SE3.Rz(-np.pi / 2)).R, [0, 0, _TAKE_END]),
]
_CAMERA = CameraParams(
    target=(0.60, 0.0, 0.60),
    distance=1.95,
    yaw=80.0,
    pitch=-16.0,
    fov=50.0,
    width=640,
    height=480,
)


def _scene(physics_client_id: int) -> _Scene:
    """The Vega bar-on-table scene; returns robot, checker, planner factory, state."""
    robot = make_vega()
    robot_links = set(robot.tree.nodes)
    left = robot.manipulators["left"]
    home_ee = robot.tree.forward_kinematics(left.ee_frame, robot.home)
    rotation = np.asarray(home_ee.R)
    # Roll 90 deg about the approach axis so the jaws straddle the bar.
    rolled = (SE3.Rt(rotation, [0, 0, 0]) * SE3.Rz(np.pi / 2)).R
    pick_grasp = SE3.Rt(rolled, [0, 0, _GIVE_END])
    grasp_world = np.asarray(home_ee.t) + 0.20 * rotation[:, 2]
    bar_center = grasp_world - np.array([0, 0, _GIVE_END])
    table_top = bar_center[2] - _BAR_LENGTH / 2

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
    checker.ignore([("cube", "table")])
    checker.ignore([(link, "cube") for link in robot_links])

    def planner(group: str) -> BiRRTPlanner:
        return BiRRTPlanner(
            robot.groups[group],
            checker.in_collision,
            np.random.default_rng(0),
            num_iters=2000,
        )

    state = KinematicState.from_tree(robot.tree, robot.home)
    return robot, checker, planner, state, pick_grasp, table_top


def _plan(
    robot: Robot,
    checker: PyBulletCollisionChecker,
    planner: _PlannerFactory,
    state: KinematicState,
    pick_grasp: SE3,
    table_top: float,
) -> list[KinematicState]:
    """The full left-pick -> handover -> right-place plan."""
    pick = Pick(
        robot,
        checker,
        planner("left_arm"),
        "cube",
        "table",
        [pick_grasp],
        manipulator="left",
        approach_distance=0.20,
    ).plan(state)
    assert pick is not None, "pick failed"
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
    assert handover is not None, "handover failed"
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
    assert place is not None, "place failed"
    return pick + handover + place


def main() -> None:
    """Build the scene, plan, render through Blender, and write the video."""
    physics_client_id = p.connect(p.DIRECT)
    robot, checker, planner, state, pick_grasp, table_top = _scene(physics_client_id)
    plan = _plan(robot, checker, planner, state, pick_grasp, table_top)
    renderer = BlenderRenderer(samples=48)
    renderer.load(robot.tree)
    # render_states restores each state's grasp edges so the held bar follows the
    # gripper through the handover.
    frames = render_states(renderer, plan, robot.tree, _CAMERA)
    path = "vega_pick_handover_place_blender.mp4"
    save_video(frames, path, fps=20)
    print(f"wrote {path} ({len(frames)} frames)")
    p.disconnect(physics_client_id)


if __name__ == "__main__":
    main()
