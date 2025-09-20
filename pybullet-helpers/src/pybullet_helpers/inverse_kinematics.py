"""Vanilla PyBullet Inverse Kinematics.

The IKFast solver is preferred over PyBullet IK, if available for the
given robot.
"""

import time
from dataclasses import dataclass
from functools import partial
from itertools import islice
from typing import Collection, Iterator

import numpy as np
import pybullet as p

from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.ikfast.utils import (
    ikfast_closest_inverse_kinematics,
    ikfast_inverse_kinematics,
)
from pybullet_helpers.joint import (
    JointPositions,
    get_joint_infos,
    get_joint_lower_limits,
    get_joint_positions,
    get_joint_upper_limits,
    get_joint_velocities,
    get_joints,
)
from pybullet_helpers.link import get_link_pose, get_link_state
from pybullet_helpers.robots.single_arm import (
    FingeredSingleArmPyBulletRobot,
    FingerState,
    SingleArmPyBulletRobot,
)
from pybullet_helpers.utils import get_closest_points_with_optional_links


class InverseKinematicsError(ValueError):
    """Thrown when inverse kinematics fails to find a solution."""


@dataclass(frozen=True)
class InverseKinematicsHyperparameters:
    """Hyperparameters for inverse kinematics."""

    convergence_atol: float = 1e-3
    max_iters: int = 100


def inverse_kinematics(
    robot: SingleArmPyBulletRobot,
    end_effector_pose: Pose,
    validate: bool = True,
    best_effort: bool = False,
    set_joints: bool = True,
    validation_atol: float = 1e-3,
) -> JointPositions:
    """Compute joint positions from a target end effector position, based on
    the robot's current joint positions. Uses IKFast if the robot has IKFast
    info specified.

    The robot's joint state is updated to the IK result. For convenience,
    the new joint positions are also returned.

    If set_joints is True, set the joints after running IK. This is
    recommended when using IKFast because IK is meant to be run
    sequentially from nearby states, since it is very sensitive to
    initialization.

    If validate is True, we guarantee that the returned joint positions
    would result in end_effector_pose if run through
    forward_kinematics.

    If best_effort is True and validate is False, return the best found solution
    after maximum effort is expended.

    WARNING: if validate is True, physics may be overridden, and so it
    should not be used within simulation.
    """
    assert not (validate and best_effort), "Cannot validate in best effort mode"

    if robot.default_inverse_kinematics_method == "custom":
        joint_positions = robot.custom_inverse_kinematics(
            end_effector_pose, validate, best_effort, validation_atol
        )
        if joint_positions is None:
            raise InverseKinematicsError()

    elif robot.default_inverse_kinematics_method == "ikfast":
        assert not best_effort, "Best effort not implemented for IKFast"

        ik_solutions = ikfast_closest_inverse_kinematics(
            robot,
            world_from_target=end_effector_pose,
        )
        if not ik_solutions:
            raise InverseKinematicsError(
                f"No IK solution found for target pose {end_effector_pose} "
                "using IKFast"
            )

        # Use first solution as it is closest to current joint state.
        joint_positions = list(ik_solutions[0])

        # IKFast doesn't handle fingers, so we add them afterwards.
        if isinstance(robot, FingeredSingleArmPyBulletRobot):
            joint_positions = add_fingers_to_joint_positions(robot, joint_positions)

        if validate:
            try:
                _validate_joints_state(
                    robot, joint_positions, end_effector_pose, validation_atol
                )
            except ValueError as e:
                raise InverseKinematicsError(e)

    elif robot.default_inverse_kinematics_method == "pybullet":
        joint_positions = pybullet_inverse_kinematics(
            robot,
            end_effector_pose,
            validate=validate,
            best_effort=best_effort,
        )

    else:
        raise NotImplementedError(
            f"Unrecognized IK method: {robot.default_inverse_kinematics_method}"
        )

    if set_joints:
        robot.set_joints(joint_positions)

    return joint_positions


def set_robot_joints_with_held_object(
    robot: SingleArmPyBulletRobot,
    physics_client_id: int,
    held_object: int | None,
    base_link_to_held_obj: Pose | None,
    joint_state: JointPositions,
) -> None:
    """Set a robot's joints and apply a transform to a held object."""

    robot.set_joints(joint_state)
    if held_object is not None:
        assert base_link_to_held_obj is not None
        world_to_base_link = get_link_state(
            robot.robot_id,
            robot.end_effector_id,
            physics_client_id=physics_client_id,
        ).com_pose
        world_to_held_obj = p.multiplyTransforms(
            world_to_base_link[0],
            world_to_base_link[1],
            base_link_to_held_obj.position,
            base_link_to_held_obj.orientation,
        )
        p.resetBasePositionAndOrientation(
            held_object,
            world_to_held_obj[0],
            world_to_held_obj[1],
            physicsClientId=physics_client_id,
        )


def check_collisions_with_held_object(
    robot: SingleArmPyBulletRobot,
    collision_bodies: Collection[int],
    physics_client_id: int,
    held_object: int | None,
    base_link_to_held_obj: Pose | None,
    joint_state: JointPositions,
    distance_threshold: float = 1e-6,
) -> bool:
    """Check if robot or a held object are in collision with certain bodies."""
    set_robot_joints_with_held_object(
        robot, physics_client_id, held_object, base_link_to_held_obj, joint_state
    )
    p.performCollisionDetection(physicsClientId=physics_client_id)
    if check_self_collisions(
        robot, perform_collision_detection=False, distance_threshold=distance_threshold
    ):
        return True
    for body in collision_bodies:
        if check_body_collisions(
            robot.robot_id,
            body,
            physics_client_id,
            perform_collision_detection=False,
            distance_threshold=distance_threshold,
        ):
            return True
        if held_object is not None and check_body_collisions(
            held_object,
            body,
            physics_client_id,
            perform_collision_detection=False,
            distance_threshold=distance_threshold,
        ):
            return True
    return False


def check_body_collisions(
    body1: int,
    body2: int,
    physics_client_id: int,
    link1: int | None = None,
    link2: int | None = None,
    distance_threshold: float = 1e-6,
    perform_collision_detection: bool = True,
) -> bool:
    """Check collisions between two bodies.

    NOTE: we previously used p.getContactPoints here instead, but ran
    into some very strange issues where the held object was clearly in
    collision, but p.getContactPoints was always empty.
    """
    closest_points = get_closest_points_with_optional_links(
        body1,
        body2,
        physics_client_id,
        link1=link1,
        link2=link2,
        distance_threshold=distance_threshold,
        perform_collision_detection=perform_collision_detection,
    )
    return len(closest_points) > 0


def check_self_collisions(
    robot: SingleArmPyBulletRobot,
    perform_collision_detection: bool = True,
    distance_threshold: float = 1e-6,
) -> bool:
    """Check if the robot has self-collisions in its current state."""
    if perform_collision_detection:
        p.performCollisionDetection(physicsClientId=robot.physics_client_id)
    for link1, link2 in robot.self_collision_link_ids:
        if check_body_collisions(
            robot.robot_id,
            robot.robot_id,
            robot.physics_client_id,
            link1,
            link2,
            perform_collision_detection=False,
            distance_threshold=distance_threshold,
        ):
            return True
    return False


def filter_collision_free_joint_generator(
    generator: Iterator[JointPositions],
    robot: SingleArmPyBulletRobot,
    collision_bodies: Collection[int],
    physics_client_id: int,
    held_object: int | None = None,
    base_link_to_held_obj: Pose | None = None,
) -> Iterator[JointPositions]:
    """Given a generator of joint positions, yield only those that pass
    collision checks.

    The typical use case is that we want to explore the null space of a
    target end effector position to find joint positions that have no
    collisions before then calling motion planning.
    """

    _collision_fn = partial(
        check_collisions_with_held_object,
        robot,
        collision_bodies,
        physics_client_id,
        held_object,
        base_link_to_held_obj,
    )

    for candidate in generator:
        if not _collision_fn(candidate):
            yield candidate


def sample_inverse_kinematics(
    robot: SingleArmPyBulletRobot,
    end_effector_pose: Pose,
    rng: np.random.Generator,
    max_time: float = 0.05,
    max_attempts: int = 1000000000,
) -> Iterator[JointPositions]:
    """Repeatedly sample inverse kinematics."""
    start_time = time.perf_counter()
    for _ in range(max_attempts):
        if time.perf_counter() - start_time > max_time:
            break
        init_joints = rng.uniform(robot.joint_lower_limits, robot.joint_upper_limits)
        robot.set_joints(init_joints.tolist())
        try:
            result = inverse_kinematics(robot, end_effector_pose)
        except InverseKinematicsError:
            continue
        if result is not None:
            yield result


def sample_collision_free_inverse_kinematics(
    robot: SingleArmPyBulletRobot,
    end_effector_pose: Pose,
    collision_bodies: set[int],
    rng: np.random.Generator,
    held_object: int | None = None,
    base_link_to_held_obj: Pose | None = None,
    max_time: float = 0.05,
    max_attempts: int = 1000000000,
    max_distance: float = np.inf,
    max_candidates: int = 100,
    norm: float = np.inf,
) -> Iterator[JointPositions]:
    """Sample in joints consistent with the end effector pose that also avoid
    collisions with the given objects."""

    if robot.ikfast_info():
        generator = ikfast_inverse_kinematics(
            robot,
            end_effector_pose,
            max_time=max_time,
            max_distance=max_distance,
            max_attempts=max_attempts,
            norm=norm,
            rng=rng,
        )

    else:
        assert np.isinf(max_distance), "Not yet implemented"
        generator = sample_inverse_kinematics(
            robot, end_effector_pose, rng, max_time, max_attempts
        )

    generator = islice(generator, max_candidates)

    if isinstance(robot, FingeredSingleArmPyBulletRobot):
        add_fingers = partial(add_fingers_to_joint_positions, robot)
        generator = map(add_fingers, generator)

    yield from filter_collision_free_joint_generator(
        generator,
        robot,
        collision_bodies,
        robot.physics_client_id,
        held_object,
        base_link_to_held_obj,
    )


def pybullet_inverse_kinematics(
    robot: SingleArmPyBulletRobot,
    target_pose: Pose,
    validate: bool = True,
    best_effort: bool = False,
    hyperparameters: InverseKinematicsHyperparameters | None = None,
) -> JointPositions:
    """Runs IK and returns joint positions for the given (free) joints.

    If validate is True, the PyBullet IK solver is called multiple
    times, resetting the robot state each time, until the target
    position is reached. If the target position is not reached after a
    maximum number of iters, an exception is raised.
    """
    if hyperparameters is None:
        hyperparameters = InverseKinematicsHyperparameters()

    # PyBullet IK optimizes over all free joints, but we only want to optimize
    # over the arm joints. So we figure out the correspondence and then limit
    # the lower and upper bounds for the non-optimized free joints.
    # Figure out which joint each dimension of the return of IK corresponds to.
    all_joints = get_joints(robot.robot_id, physics_client_id=robot.physics_client_id)
    joint_infos = get_joint_infos(
        robot.robot_id, all_joints, physics_client_id=robot.physics_client_id
    )
    free_joints = [
        joint_info.jointIndex for joint_info in joint_infos if joint_info.qIndex > -1
    ]
    assert set(robot.arm_joints).issubset(set(free_joints))
    current_joint_positions = get_joint_positions(
        robot.robot_id, free_joints, robot.physics_client_id
    )
    lower_limits = []
    upper_limits = []
    for idx, joint in enumerate(free_joints):
        if joint in robot.arm_joints:
            lower_limit = get_joint_lower_limits(
                robot.robot_id, [joint], robot.physics_client_id
            )[0]
            upper_limit = get_joint_upper_limits(
                robot.robot_id, [joint], robot.physics_client_id
            )[0]
            lower_limits.append(lower_limit)
            upper_limits.append(upper_limit)
        else:
            lower_limits.append(current_joint_positions[idx])
            upper_limits.append(current_joint_positions[idx])

    # Record the initial state of the joints (positions and velocities) so
    # that we can reset them after.
    current_joint_velocities = get_joint_velocities(
        robot.robot_id, free_joints, robot.physics_client_id
    )

    # Running IK once is often insufficient, so we run it multiple times until
    # convergence. If it does not converge, an error is raised.
    for _ in range(hyperparameters.max_iters):
        free_joint_vals = p.calculateInverseKinematics(
            robot.robot_id,
            robot.end_effector_id,
            target_pose.position,
            targetOrientation=target_pose.orientation,
            lowerLimits=lower_limits,
            upperLimits=upper_limits,
            physicsClientId=robot.physics_client_id,
        )
        assert len(free_joints) == len(free_joint_vals)
        if not validate and not best_effort:
            break

        # Check joint limits.
        limits_violated = False
        for lo, val, hi in zip(
            lower_limits, free_joint_vals, upper_limits, strict=True
        ):
            if not lo <= val <= hi:
                limits_violated = True
        if limits_violated:
            continue

        # Update the robot state and check if the desired position and
        # orientation are reached.
        for joint, joint_val in zip(free_joints, free_joint_vals):
            p.resetJointState(
                robot.robot_id,
                joint,
                targetValue=joint_val,
                physicsClientId=robot.physics_client_id,
            )

        # If there is a match, this succeeded, so break.
        ee_link_pose = get_link_pose(
            robot.robot_id, robot.end_effector_id, robot.physics_client_id
        )
        if ee_link_pose.allclose(target_pose, atol=hyperparameters.convergence_atol):
            break
    else:
        if not best_effort:
            raise InverseKinematicsError("Inverse kinematics failed to converge.")

    # Reset the joint state (positions and velocities) to their initial values
    # to avoid modifying the PyBullet internal state.
    if validate:
        for joint, pos, vel in zip(
            free_joints, current_joint_positions, current_joint_velocities, strict=True
        ):
            p.resetJointState(
                robot.robot_id,
                joint,
                targetValue=pos,
                targetVelocity=vel,
                physicsClientId=robot.physics_client_id,
            )

    # Order the found free_joint_vals based on the requested joints.
    joint_vals = []
    for joint in robot.arm_joints:
        free_joint_idx = free_joints.index(joint)
        joint_val = free_joint_vals[free_joint_idx]
        joint_vals.append(joint_val)

    return joint_vals


def end_effector_transform_to_joints(
    robot: SingleArmPyBulletRobot, transform: Pose
) -> JointPositions:
    """Given a transform for the robot's end effectors relative to the current
    joint state, return a next joint state."""
    current_end_effector_pose = robot.get_end_effector_pose()
    next_end_effector_pose = multiply_poses(current_end_effector_pose, transform)
    return inverse_kinematics(robot, next_end_effector_pose, set_joints=False)


def sample_joints_from_task_space_bounds(
    rng: np.random.Generator,
    robot: SingleArmPyBulletRobot,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    min_z: float,
    max_z: float,
    min_roll: float,
    max_roll: float,
    min_pitch: float,
    max_pitch: float,
    min_yaw: float,
    max_yaw: float,
) -> JointPositions:
    """Sample from end effector space bounds and return joints."""
    while True:
        x = rng.uniform(min_x, max_x)
        y = rng.uniform(min_y, max_y)
        z = rng.uniform(min_z, max_z)
        roll = rng.uniform(min_roll, max_roll)
        pitch = rng.uniform(min_pitch, max_pitch)
        yaw = rng.uniform(min_yaw, max_yaw)
        quat = tuple(p.getQuaternionFromEuler((roll, pitch, yaw)))
        pose = Pose((x, y, z), quat)
        try:
            return inverse_kinematics(robot, pose)
        except InverseKinematicsError:
            continue


def _validate_joints_state(
    robot: SingleArmPyBulletRobot,
    joint_positions: JointPositions,
    target_pose: Pose,
    validation_atol: float,
) -> None:
    """Validate that the given joint positions matches the target pose.

    This method should NOT be used during simulation mode as it resets
    the joint states.
    """
    # Store current joint positions so we can reset.
    initial_joint_states = robot.get_joint_positions()

    # Set joint states, forward kinematics to determine EE position.
    robot.set_joints(joint_positions)
    ee_pos = get_link_state(
        robot.robot_id,
        robot.end_effector_id,
        physics_client_id=robot.physics_client_id,
    ).worldLinkFramePosition
    target_pos = target_pose.position
    pos_is_close = np.allclose(ee_pos, target_pos, atol=validation_atol)

    # Reset joint positions before returning/raising error.
    robot.set_joints(initial_joint_states)

    if not pos_is_close:
        raise ValueError(
            f"Joint states do not match target pose {target_pos} " f"from {ee_pos}"
        )


def add_fingers_to_joint_positions(
    robot: FingeredSingleArmPyBulletRobot,
    joint_positions: JointPositions,
    finger_state: FingerState | None = None,
) -> JointPositions:
    """Extend arm joint positions to include the fingers.

    If finger_state is None, use the current robot finger state.
    """
    joint_idx_to_value = dict(enumerate(joint_positions))
    finger_idxs = robot.finger_joint_idxs
    if finger_state is None:
        finger_state = robot.get_finger_state()
    finger_joint_values = robot.finger_state_to_joints(finger_state)
    for idx, value in zip(finger_idxs, finger_joint_values, strict=True):
        joint_idx_to_value[idx] = value
    final_joint_positions = [
        joint_idx_to_value[i] for i in range(len(joint_idx_to_value))
    ]
    return final_joint_positions
