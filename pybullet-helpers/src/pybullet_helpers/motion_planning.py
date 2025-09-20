"""Motion Planning in PyBullet."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import partial
from typing import Callable, Collection, Iterable, Iterator, Optional

import numpy as np
from tomsutils.motion_planning import RRT, BiRRT

from pybullet_helpers.geometry import (
    Pose,
    get_pose,
    iter_between_poses,
    multiply_poses,
    set_pose,
)
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    check_body_collisions,
    check_collisions_with_held_object,
    filter_collision_free_joint_generator,
    pybullet_inverse_kinematics,
    sample_collision_free_inverse_kinematics,
)
from pybullet_helpers.joint import (
    JointInfo,
    JointPositions,
    get_joint_infos,
    get_jointwise_difference,
    interpolate_joints,
    iter_between_joint_positions,
)
from pybullet_helpers.math_utils import geometric_sequence
from pybullet_helpers.robots.single_arm import (
    FingeredSingleArmPyBulletRobot,
    SingleArmPyBulletRobot,
)
from pybullet_helpers.trajectory import (
    TrajectorySegment,
    concatenate_trajectories,
    iter_traj_with_max_distance,
)


@dataclass(frozen=True)
class MotionPlanningHyperparameters:
    """Hyperparameters for motion planning."""

    birrt_extend_num_interp: int = 10
    birrt_num_attempts: int = 10
    birrt_num_iters: int = 100
    birrt_smooth_amt: int = 50


def run_motion_planning(
    robot: SingleArmPyBulletRobot,
    initial_positions: JointPositions,
    target_positions: JointPositions,
    collision_bodies: Collection[int],
    seed: int,
    physics_client_id: int,
    held_object: int | None = None,
    base_link_to_held_obj: Pose | None = None,
    hyperparameters: MotionPlanningHyperparameters | None = None,
    additional_state_constraint_fn: Callable[[JointPositions], bool] | None = None,
    sampling_fn: Callable[[JointPositions], JointPositions] | None = None,
    direct_path_only: bool = False,
    distance_threshold: float = 1e-6,
) -> Optional[list[JointPositions]]:
    """Run BiRRT to find a collision-free sequence of joint positions.

    Note that this function changes the state of the robot.

    If additional_state_constraint_fn is provided, the collision
    checking is augmented so that additional_state_constraint_fn() =
    False behaves as if a collision check failed. For example, if you
    want to make sure that a held object is not rotated beyond some
    threshold, you could use additional_state_constraint_fn to enforce
    that. The additional state constraint function can assume that the
    robot is already in the given joint positions because it will be
    called right after collision checking, which sets the robot state.

    If sampling_fn is not provided, defaults to uniform joint space.
    """
    if hyperparameters is None:
        hyperparameters = MotionPlanningHyperparameters()

    rng = np.random.default_rng(seed)
    joint_space = robot.action_space
    joint_space.seed(seed)
    joint_infos = get_joint_infos(robot.robot_id, robot.arm_joints, physics_client_id)
    num_interp = hyperparameters.birrt_extend_num_interp

    _collision_fn: Callable[[JointPositions], bool] = partial(
        check_collisions_with_held_object,
        robot,
        collision_bodies,
        physics_client_id,
        held_object,
        base_link_to_held_obj,
        distance_threshold=distance_threshold,
    )
    if additional_state_constraint_fn is not None:
        _initial_collision_fn = _collision_fn
        _collision_fn = lambda x: _initial_collision_fn(
            x
        ) or not additional_state_constraint_fn(x)

    _distance_fn = partial(
        get_joint_positions_distance, robot, joint_infos, metric="end_effector"
    )

    def _joint_space_sample_fn(pt: JointPositions) -> JointPositions:
        new_pt: JointPositions = list(joint_space.sample())
        # Don't change the fingers.
        if isinstance(robot, FingeredSingleArmPyBulletRobot):
            for idx in robot.finger_joint_idxs:
                new_pt[idx] = pt[idx]
        return new_pt

    if sampling_fn is None:
        sampling_fn = _joint_space_sample_fn

    def _extend_fn(
        pt1: JointPositions, pt2: JointPositions
    ) -> Iterator[JointPositions]:
        yield from iter_between_joint_positions(
            joint_infos, pt1, pt2, num_interp_per_unit=num_interp, include_start=False
        )

    birrt = BiRRT(
        sampling_fn,
        _extend_fn,
        _collision_fn,
        _distance_fn,
        rng,
        num_attempts=hyperparameters.birrt_num_attempts,
        num_iters=hyperparameters.birrt_num_iters,
        smooth_amt=hyperparameters.birrt_smooth_amt,
    )

    if direct_path_only:
        return birrt.try_direct_path(initial_positions, target_positions)

    return birrt.query(initial_positions, target_positions)


def get_motion_plan_distance(
    motion_plan: list[JointPositions],
    joint_distance_fn: Callable[[JointPositions, JointPositions], float],
) -> float:
    """Get the total distance for a motion plan under a given metric."""
    mp_dist = 0.0
    for t in range(len(motion_plan) - 1):
        q1, q2 = motion_plan[t], motion_plan[t + 1]
        dist = joint_distance_fn(q2, q1)
        mp_dist += dist
    return mp_dist


def select_shortest_motion_plan(
    motion_plans: Iterable[list[JointPositions]],
    joint_distance_fn: Callable[[JointPositions, JointPositions], float],
) -> list[JointPositions]:
    """Return the motion plan that has the least cumulative distance."""

    shortest_motion_plan: list[JointPositions] | None = None
    shortest_length = np.inf

    for motion_plan in motion_plans:
        mp_dist = get_motion_plan_distance(motion_plan, joint_distance_fn)
        if mp_dist < shortest_length:
            shortest_motion_plan = motion_plan
            shortest_length = mp_dist

    assert shortest_motion_plan is not None, "motion_plans was empty"
    return shortest_motion_plan


def run_smooth_motion_planning_to_pose(
    target_pose: Pose | Callable[[], Pose],
    robot: SingleArmPyBulletRobot,
    collision_ids: set[int],
    end_effector_frame_to_plan_frame: Pose,
    seed: int,
    held_object: int | None = None,
    base_link_to_held_obj: Pose | None = None,
    max_time: float = np.inf,
    max_candidate_plans: int | None = None,
    joint_geometric_scalar: float = 0.9,
    birrt_num_attempts: int = 10,
    birrt_num_iters: int = 100,
    sampling_fn: Callable[[JointPositions], JointPositions] | None = None,
    distance_threshold: float = 1e-6,
) -> Optional[list[JointPositions]]:
    """A naive smooth motion planner that reruns motion planning multiple times
    and then picks the "smoothest" result according to a geometric weighting of
    the joints (so the lowest joint should move the least)."""
    assert (
        not np.isinf(max_time) or max_candidate_plans is not None
    ), "Must specify either max_time or max_candidate_plans"

    # Target poses can be sampled or singletons.
    if isinstance(target_pose, Pose):
        target_pose_sampler: Callable[[], Pose] = lambda: target_pose  # type: ignore
    else:
        target_pose_sampler = target_pose

    # Set up the geometrically weighted score function.
    def _score_motion_plan(plan: list[JointPositions]) -> float:
        weights = geometric_sequence(joint_geometric_scalar, len(robot.arm_joints))
        joint_infos = get_joint_infos(
            robot.robot_id, robot.arm_joints, robot.physics_client_id
        )
        dist_fn = partial(
            get_joint_positions_distance,
            robot,
            joint_infos,
            metric="weighted_joints",
            weights=weights,
        )
        return get_motion_plan_distance(plan, dist_fn)

    robot_initial_joints = robot.get_joint_positions()

    start_time = time.perf_counter()
    best_motion_plan: list[JointPositions] | None = None
    best_motion_plan_score: float = np.inf  # lower is better
    num_iters = 0
    iter_ub = max_candidate_plans if max_candidate_plans is not None else np.inf
    rng = np.random.default_rng(seed)

    while time.perf_counter() - start_time < max_time and num_iters < iter_ub:
        # Sample a target pose.
        target_pose = target_pose_sampler()
        # Transform to end effector space.
        end_effector_pose = multiply_poses(
            target_pose, end_effector_frame_to_plan_frame
        )
        # Sample a collision-free joint target. If none exist, we'll just
        # go back to sampling a different target pose.
        for target_joint_positions in sample_collision_free_inverse_kinematics(
            robot,
            end_effector_pose,
            collision_ids,
            rng,
            max_time=max_time,
            max_candidates=1,
        ):
            # Try motion planning to the target.
            robot.set_joints(robot_initial_joints)
            motion_plan = run_motion_planning(
                robot,
                robot_initial_joints,
                target_joint_positions,
                collision_ids,
                seed,
                robot.physics_client_id,
                held_object=held_object,
                base_link_to_held_obj=base_link_to_held_obj,
                sampling_fn=sampling_fn,
                hyperparameters=MotionPlanningHyperparameters(
                    birrt_num_attempts=birrt_num_attempts,
                    birrt_num_iters=birrt_num_iters,
                ),
                distance_threshold=distance_threshold,
            )
            # Score the motion plan.
            if motion_plan is not None:
                num_iters += 1
                score = _score_motion_plan(motion_plan)
                if score < best_motion_plan_score:
                    best_motion_plan = motion_plan
                    best_motion_plan_score = score

    return best_motion_plan


def smoothly_follow_end_effector_path(
    robot: SingleArmPyBulletRobot,
    end_effector_path: list[Pose],
    initial_joints: JointPositions,
    collision_ids: set[int],
    joint_distance_fn: Callable[[JointPositions, JointPositions], float],
    held_object: int | None = None,
    base_link_to_held_obj: Pose | None = None,
    max_time: float = 5.0,
    max_smoothing_iters_per_step: int = 1000000,
    include_start: bool = True,
    allow_skipping_intermediates: bool = True,
    seed: int = 0,
) -> list[JointPositions]:
    """Find a smooth (short) joint trajectory that follows the given end
    effector path while avoiding collisions.

    NOTE: if allow_skipping_intermediates is True, then some intermediate
    waypoints may be skipped if inverse kinematics fails.
    """

    joint_position_path: list[JointPositions] = []
    if include_start:
        joint_position_path.append(initial_joints)
    current_joints = initial_joints

    max_time_per_step = max_time / len(end_effector_path)
    rng = np.random.default_rng(seed)
    for i, end_effector_pose in enumerate(end_effector_path):
        # Get the closest neighbor in joint space.
        robot.set_joints(current_joints)  # for warm starting IK
        closest_neighbor: JointPositions | None = None
        closest_dist = np.inf
        # Try differential IK first because that often leads to closer solutions
        # when the target is close to the current end effector pose.
        remaining_candidates = max_smoothing_iters_per_step
        try:
            neighbor = pybullet_inverse_kinematics(
                robot,
                end_effector_pose,
                validate=True,
            )
            generator = iter([neighbor])
            for candidate in filter_collision_free_joint_generator(
                generator,
                robot,
                collision_ids,
                robot.physics_client_id,
                held_object,
                base_link_to_held_obj,
            ):
                closest_dist = joint_distance_fn(current_joints, candidate)
                closest_neighbor = candidate
                remaining_candidates -= 1
        except InverseKinematicsError:
            pass
        for neighbor in sample_collision_free_inverse_kinematics(
            robot,
            end_effector_pose,
            collision_ids,
            rng,
            held_object,
            base_link_to_held_obj,
            max_time=max_time_per_step,
            max_candidates=remaining_candidates,
        ):
            dist = joint_distance_fn(current_joints, neighbor)
            if dist < closest_dist:
                closest_dist = dist
                closest_neighbor = neighbor
        if closest_neighbor is None:
            if allow_skipping_intermediates and 0 < i < len(end_effector_path) - 1:
                continue
            raise InverseKinematicsError
        joint_position_path.append(closest_neighbor)
        current_joints = closest_neighbor

    return joint_position_path


def create_joint_distance_fn(
    robot: SingleArmPyBulletRobot,
    metric: str = "weighted_joints",
    weight_base: float = 0.9,
) -> Callable[[JointPositions, JointPositions], float]:
    """Helper for creating a joint distance function for a robot."""

    weights = geometric_sequence(weight_base, len(robot.arm_joint_names))
    joint_infos = get_joint_infos(
        robot.robot_id, robot.arm_joints, robot.physics_client_id
    )

    def _joint_distance_fn(pt1: JointPositions, pt2: JointPositions) -> float:
        return get_joint_positions_distance(
            robot,
            joint_infos,
            pt1,
            pt2,
            metric=metric,
            weights=weights,
        )

    return _joint_distance_fn


def get_joint_positions_distance(
    robot: SingleArmPyBulletRobot,
    joint_infos: list[JointInfo],
    q1: JointPositions,
    q2: JointPositions,
    metric: str = "end_effector",
    **kwargs,
):
    """Get the distance between two joint positions."""

    if metric == "end_effector":
        return _get_end_effector_joint_positions_distance(robot, q1, q2, **kwargs)

    if metric == "weighted_joints":
        return _get_weighted_joint_positions_distance(joint_infos, q1, q2, **kwargs)

    raise NotImplementedError(f"Unrecognized metric: {metric}")


def _get_end_effector_joint_positions_distance(
    robot: SingleArmPyBulletRobot,
    q1: JointPositions,
    q2: JointPositions,
) -> float:
    # NOTE: only using positions to calculate distance. Should use
    # orientations as well in the near future.
    from_ee = robot.forward_kinematics(q1).position
    to_ee = robot.forward_kinematics(q2).position
    return sum(np.subtract(from_ee, to_ee) ** 2)


def _get_weighted_joint_positions_distance(
    joint_infos: list[JointInfo],
    q1: JointPositions,
    q2: JointPositions,
    weights: list[float],
) -> float:
    assert len(joint_infos) == len(weights) == len(q1) == len(q2)
    diff = get_jointwise_difference(joint_infos, q2, q1)
    dist = np.abs(diff)
    return np.sum(weights * dist)


def remap_joint_position_plan_to_constant_distance(
    plan: list[JointPositions],
    robot: SingleArmPyBulletRobot,
    max_distance: float = 0.1,
    distance_fn: Callable[[JointPositions, JointPositions], float] | None = None,
) -> list[JointPositions]:
    """Re-interpolate a joint position plan so that it has constant distance
    with a max distance specified."""

    joint_infos = get_joint_infos(
        robot.robot_id, robot.arm_joints, robot.physics_client_id
    )

    def _interpolate_fn(q1, q2, t):
        return interpolate_joints(joint_infos, q1, q2, t)

    if distance_fn is None:
        distance_fn = create_joint_distance_fn(robot)

    # Use distances as times.
    distances = []
    for pt1, pt2 in zip(plan[:-1], plan[1:], strict=True):
        dist = distance_fn(pt1, pt2)
        distances.append(dist)

    segments = []
    for t in range(len(plan) - 1):
        seg = TrajectorySegment(
            plan[t],
            plan[t + 1],
            distances[t],
            interpolate_fn=_interpolate_fn,
            distance_fn=distance_fn,
        )
        segments.append(seg)
    continuous_time_trajectory = concatenate_trajectories(segments)
    remapped_plan = list(
        iter_traj_with_max_distance(continuous_time_trajectory, max_distance)
    )
    return remapped_plan


def run_base_motion_planning(
    robot: SingleArmPyBulletRobot,
    initial_pose: Pose,
    goal: Pose | Callable[[Pose], bool],
    position_lower_bounds: tuple[float, float],
    position_upper_bounds: tuple[float, float],
    collision_bodies: Collection[int],
    seed: int,
    physics_client_id: int,
    held_object: int | None = None,
    base_link_to_held_obj: Pose | None = None,
    platform: int | None = None,
    hyperparameters: MotionPlanningHyperparameters | None = None,
) -> Optional[list[Pose]]:
    """Run motion planning for the robot base in SE2."""
    if hyperparameters is None:
        hyperparameters = MotionPlanningHyperparameters()

    rng = np.random.default_rng(seed)
    num_interp = hyperparameters.birrt_extend_num_interp

    # Determine the transform between the platform and robot base if applicable.
    if platform is not None:
        world_to_base = robot.get_base_pose()
        world_to_platform = get_pose(platform, physics_client_id)
        base_to_platform = multiply_poses(world_to_base.invert(), world_to_platform)

    # The joint positions and z position of the robot won't change.
    base_z = robot.get_base_pose().position[2]
    joint_state = robot.get_joint_positions()

    def _set_robot(pt: Pose) -> None:
        robot.set_base(pt)
        if platform is not None:
            platform_pose = multiply_poses(pt, base_to_platform)
            set_pose(platform, platform_pose, physics_client_id)

    def _collision_fn(pt: Pose) -> bool:
        _set_robot(pt)
        if check_collisions_with_held_object(
            robot,
            collision_bodies,
            physics_client_id,
            held_object,
            base_link_to_held_obj,
            joint_state,
        ):
            return True
        if platform is not None:
            for collision_body in collision_bodies:
                if check_body_collisions(
                    platform,
                    collision_body,
                    physics_client_id,
                    perform_collision_detection=False,
                ):
                    return True
        return False

    def _distance_fn(pt1: Pose, pt2: Pose) -> float:
        return float(
            np.linalg.norm(np.subtract(pt1.position, pt2.position))
            + np.linalg.norm(np.subtract(pt1.orientation, pt2.orientation))
        )

    def _sampling_fn(pt: Pose) -> Pose:
        del pt  # not used
        x, y = rng.uniform(position_lower_bounds, position_upper_bounds)
        yaw = rng.uniform(-np.pi, np.pi)
        return Pose.from_rpy((x, y, base_z), rpy=(0, 0, yaw))

    def _extend_fn(pt1: Pose, pt2: Pose) -> Iterator[Pose]:
        yield from iter_between_poses(
            pt1, pt2, num_interp=num_interp, include_start=False
        )

    # Use BiRRT if a target base pose is given.
    if isinstance(goal, Pose):
        birrt = BiRRT(
            _sampling_fn,
            _extend_fn,
            _collision_fn,
            _distance_fn,
            rng,
            num_attempts=hyperparameters.birrt_num_attempts,
            num_iters=hyperparameters.birrt_num_iters,
            smooth_amt=hyperparameters.birrt_smooth_amt,
        )

        return birrt.query(initial_pose, goal)

    # Use RRT is the goal is a function.
    rrt = RRT(
        _sampling_fn,
        _extend_fn,
        _collision_fn,
        _distance_fn,
        rng,
        num_attempts=hyperparameters.birrt_num_attempts,
        num_iters=hyperparameters.birrt_num_iters,
        smooth_amt=hyperparameters.birrt_smooth_amt,
    )

    return rrt.query_to_goal_fn(initial_pose, goal)
