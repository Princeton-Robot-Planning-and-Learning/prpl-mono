"""Pick and place primitives over the KinematicTree stack.

A grasp is just an edge: picking re-parents the object onto the gripper
(``tree.attach``), so the held object follows forward kinematics and a plan is a
``list[KinematicState]`` whose object edge flips at the grasp. Collision logic
follows pybullet-helpers: motion-plan to a pregrasp against all obstacles, then
descend and retreat while disregarding the target object and its support surface
(the gripper must contact the object and is near the surface).

The descend and retreat are straight Cartesian lines tracked by differential
``NumericalIK`` (via ``follow_end_effector_path``), seeded continuously so the arm
stays on one IK branch -- unlike joint-space interpolation between two independent
analytic solutions, which can swing the arm through obstacles when the solver picks
different branches for the pregrasp and grasp.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

import numpy as np
from spatialmath import SE3

from prpl_kinematics.collision import PyBulletCollisionChecker
from prpl_kinematics.ik.follow import follow_end_effector_path
from prpl_kinematics.ik.numerical import NumericalIK
from prpl_kinematics.planning.joint_space import JointSpace
from prpl_kinematics.planning.motion_planner import MotionPlanner
from prpl_kinematics.robots.robot import Manipulator, Robot
from prpl_kinematics.tree.kinematic_tree import Configuration
from prpl_kinematics.tree.state import KinematicState


@runtime_checkable
class Primitive(Protocol):
    """A manipulation primitive: turn a starting state into a plan, or fail."""

    def plan(self, state: KinematicState) -> list[KinematicState] | None:
        """A plan (sequence of states) achieving the primitive, or ``None``."""
        raise NotImplementedError


class Pick:
    """Pick an object off a surface with a generated grasp.

    ``grasps`` yields the gripper pose in the object frame; the first grasp whose
    pregrasp is reachable, descent is collision-free, and retreat lifts the held
    object clear is used.
    """

    def __init__(
        self,
        robot: Robot,
        checker: PyBulletCollisionChecker,
        planner: MotionPlanner,
        object_frame: str,
        surface_frame: str,
        grasps: Iterable[SE3],
        manipulator: str = "arm",
        approach_distance: float = 0.12,
        descend_resolution: float = 0.02,
    ) -> None:
        self._robot = robot
        self._checker = checker
        self._planner = planner
        self._object = object_frame
        self._surface = surface_frame
        self._grasps = grasps
        self._manipulator = robot.manipulators[manipulator]
        self._approach = approach_distance
        self._resolution = descend_resolution
        self._follow_ik = _follow_ik(robot, self._manipulator)

    def plan(self, state: KinematicState) -> list[KinematicState] | None:
        """A pick plan ending with the object grasped, or ``None``."""
        tree = self._robot.tree
        ee = self._manipulator.ee_frame
        ik = self._manipulator.ik
        ignore = {self._object, self._surface}
        for grasp in self._grasps:
            config = state.apply(tree)  # restore the un-grasped scene each attempt
            object_world = tree.forward_kinematics(self._object, config)
            grasp_world = object_world * grasp
            pregrasp_world = grasp_world * SE3(0.0, 0.0, -self._approach)

            pregrasp_cfg = ik.solve(pregrasp_world, config)
            if pregrasp_cfg is None:
                continue
            approach = self._planner.plan(config, pregrasp_cfg)
            if approach is None:
                continue
            descend = _cartesian_segment(
                self._follow_ik,
                self._checker,
                pregrasp_world,
                grasp_world,
                pregrasp_cfg,
                ignore,
                self._resolution,
            )
            if descend is None:
                continue
            grasp_cfg = descend[-1]

            plan = [
                KinematicState.from_tree(tree, c) for c in [config, *approach, *descend]
            ]
            tree.attach(
                self._object, ee, tree.relative_pose(ee, self._object, grasp_cfg)
            )
            retreat = _cartesian_segment(
                self._follow_ik,
                self._checker,
                grasp_world,
                pregrasp_world,
                grasp_cfg,
                ignore,
                self._resolution,
            )
            if retreat is None:
                continue
            plan += [KinematicState.from_tree(tree, c) for c in retreat]
            return plan
        return None


class Place:
    """Place a held object onto a surface at a target object pose.

    ``placements`` yields the object's target pose in the world frame.
    """

    def __init__(
        self,
        robot: Robot,
        checker: PyBulletCollisionChecker,
        planner: MotionPlanner,
        object_frame: str,
        surface_frame: str,
        placements: Iterable[SE3],
        manipulator: str = "arm",
        approach_distance: float = 0.12,
        descend_resolution: float = 0.02,
    ) -> None:
        self._robot = robot
        self._checker = checker
        self._planner = planner
        self._object = object_frame
        self._surface = surface_frame
        self._placements = placements
        self._manipulator = robot.manipulators[manipulator]
        self._approach = approach_distance
        self._resolution = descend_resolution
        self._follow_ik = _follow_ik(robot, self._manipulator)

    def plan(self, state: KinematicState) -> list[KinematicState] | None:
        """A place plan ending with the object released onto the surface, or
        ``None``."""
        tree = self._robot.tree
        ee = self._manipulator.ee_frame
        ik = self._manipulator.ik
        ignore = {self._object, self._surface}
        for placement in self._placements:
            config = state.apply(tree)  # restore the held-object scene each attempt
            ee_from_object = tree.relative_pose(ee, self._object, config)
            place_world = placement * ee_from_object.inv()
            preplace_world = place_world * SE3(0.0, 0.0, -self._approach)

            preplace_cfg = ik.solve(preplace_world, config)
            if preplace_cfg is None:
                continue
            approach = self._planner.plan(config, preplace_cfg)
            if approach is None:
                continue
            descend = _cartesian_segment(
                self._follow_ik,
                self._checker,
                preplace_world,
                place_world,
                preplace_cfg,
                ignore,
                self._resolution,
            )
            if descend is None:
                continue
            place_cfg = descend[-1]

            plan = [
                KinematicState.from_tree(tree, c) for c in [config, *approach, *descend]
            ]
            tree.attach(self._object, tree.root, placement)  # release onto the surface
            retreat = _cartesian_segment(
                self._follow_ik,
                self._checker,
                place_world,
                preplace_world,
                place_cfg,
                ignore,
                self._resolution,
            )
            if retreat is None:
                continue
            plan += [KinematicState.from_tree(tree, c) for c in retreat]
            return plan
        return None


def _follow_ik(robot: Robot, manipulator: Manipulator) -> NumericalIK:
    """A differential IK over the manipulator's arm group, for Cartesian following."""
    group = robot.groups[manipulator.group]
    assert isinstance(group, JointSpace), "Cartesian following needs a JointSpace arm"
    return NumericalIK(robot.tree, group, manipulator.ee_frame)


def _cartesian_segment(
    follow_ik: NumericalIK,
    checker: PyBulletCollisionChecker,
    start_pose: SE3,
    end_pose: SE3,
    seed: Configuration,
    ignore: set[str],
    resolution: float,
) -> list[Configuration] | None:
    """Follow a straight Cartesian line ``start_pose`` -> ``end_pose`` with numerical
    IK.

    The end-effector tracks evenly spaced poses, each solved seeded from the previous
    (the first from ``seed``), so the arm stays on one continuous branch instead of
    jumping between the analytic solver's solutions. Returns ``None`` if the follow
    fails or any configuration collides (ignoring ``ignore`` -- the grasped object and
    its support surface).
    """
    distance = float(np.linalg.norm(np.asarray(end_pose.t) - np.asarray(start_pose.t)))
    steps = max(1, int(np.ceil(distance / resolution)))
    poses = [start_pose.interp(end_pose, (i + 1) / steps) for i in range(steps)]
    configs = follow_end_effector_path(follow_ik, poses, seed)
    if configs is None:
        return None
    for config in configs:
        if checker.in_collision(config, ignored_nodes=ignore):
            return None
    return configs
