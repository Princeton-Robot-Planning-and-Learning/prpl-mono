"""Tests for the Dexmate Vega robot."""

import importlib.util
import warnings

import imageio.v2 as iio
import numpy as np
import pybullet as p
import pytest

from pybullet_helpers.camera import capture_image
from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    inverse_kinematics,
)
from pybullet_helpers.manipulation import get_kinematic_plan_to_pick_object
from pybullet_helpers.robots import _dexmate_vega_ik as _vega_ik
from pybullet_helpers.robots import dexmate_vega
from pybullet_helpers.robots.dexmate_vega import (
    DexmateVega1ULeftArmPyBulletRobot,
    DexmateVega1UPyBulletRobot,
    DexmateVega1URightArmPyBulletRobot,
)
from pybullet_helpers.states import KinematicState
from pybullet_helpers.utils import create_pybullet_block

EAIK_INSTALLED = importlib.util.find_spec("eaik") is not None


def test_dexmate_vega_1u_robot(physics_client_id):
    """Tests for DexmateVega1ULeftArmPyBulletRobot."""
    robot = DexmateVega1ULeftArmPyBulletRobot(physics_client_id)
    assert robot.get_name() == "dexmate-vega-1u-left-arm"
    # The 7 left-arm joints followed by the 2 parallel-jaw gripper joints.
    assert robot.arm_joint_names == [
        "L_arm_j1",
        "L_arm_j2",
        "L_arm_j3",
        "L_arm_j4",
        "L_arm_j5",
        "L_arm_j6",
        "L_arm_j7",
        "L_gripper_j1",
        "L_gripper_j2",
    ]
    assert robot.finger_joint_idxs == [7, 8]
    assert np.allclose(robot.action_space.low, robot.joint_lower_limits)
    assert np.allclose(robot.action_space.high, robot.joint_upper_limits)
    # Moving each joint to its midpoint produces an EE pose within reach
    # (forward_kinematics doesn't raise).
    for i in range(len(robot.arm_joints)):
        q = list(robot.home_joint_positions)
        q[i] = 0.5 * (robot.joint_lower_limits[i] + robot.joint_upper_limits[i])
        robot.forward_kinematics(q)


@pytest.mark.skipif(not EAIK_INSTALLED, reason="EAIK not installed")
def test_dexmate_vega_1u_ik_roundtrip(physics_client_id):
    """IK roundtrip: random q -> FK -> custom IK -> FK -> matches."""
    robot = DexmateVega1ULeftArmPyBulletRobot(physics_client_id)
    assert robot.default_inverse_kinematics_method == "custom"

    rng = np.random.default_rng(7)
    lo = np.array(robot.joint_lower_limits)
    hi = np.array(robot.joint_upper_limits)
    n_trials = 5
    n_success = 0
    for _ in range(n_trials):
        q_true = rng.uniform(lo, hi)
        target = robot.forward_kinematics(q_true.tolist())
        try:
            q_sol = inverse_kinematics(
                robot, target, validate=True, validation_atol=1e-3
            )
        except InverseKinematicsError:
            continue
        recovered = robot.forward_kinematics(q_sol)
        pos_err = np.linalg.norm(
            np.array(recovered.position) - np.array(target.position)
        )
        if pos_err < 1e-3:
            n_success += 1
    # Allow one failure to absorb the rare Nelder-Mead local-min case.
    assert (
        n_success >= n_trials - 1
    ), f"only {n_success}/{n_trials} IK roundtrips succeeded"


def test_dexmate_vega_arm_ik_params_extracted_from_urdf():
    """The left-arm params extracted from the URDF match the known-good values from the
    original single-arm implementation, guarding the URDF extraction.

    The right arm is a sagittal mirror, so its axes/limits differ from the left.
    """
    left = _vega_ik.get_arm_ik_params("L")
    expected_h = np.array(
        [[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dtype=float,
    ).T
    expected_p = np.array(
        [
            [0.0, 0.16946, 0.0],
            [0.04, 0.06, 0.0454],
            [0.1644, 0.0, -0.043],
            [0.113, 0.0433, 0.06],
            [0.1938, -0.0434, -0.04],
            [0.0762, 0.0319, 0.0],
            [0.065, -0.032, 0.0319],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    ).T
    assert np.allclose(left.H, expected_h)
    assert np.allclose(left.P, expected_p)
    assert np.allclose(
        left.lower, [-3.071, -0.453, -3.071, -3.071, -3.071, -1.396, -1.378]
    )
    assert np.allclose(left.upper, [3.071, 1.553, 3.071, 0.244, 3.071, 1.396, 1.117])

    # The right arm mirrors the left across the sagittal plane: j1 and j6 axes flip
    # sign, and the j2/j7 limits flip sign.
    right = _vega_ik.get_arm_ik_params("R")
    assert np.allclose(right.H[:, 0], [0, -1, 0])
    assert np.allclose(right.H[:, 5], [0, -1, 0])
    assert np.allclose(right.lower[1], -1.553) and np.allclose(right.upper[1], 0.453)


@pytest.mark.skipif(not EAIK_INSTALLED, reason="EAIK not installed")
@pytest.mark.parametrize("prefix", ["L", "R"])
def test_dexmate_vega_solve_arm_ik_roundtrip(prefix):
    """solve_arm_ik recovers joints for both arms: random q -> EAIK FK -> solve -> FK
    matches. Exercises the mirrored right-arm geometry, which is not used by the
    single-arm robot but will be by the bimanual robot."""
    # EAIK only constructs robots up to 6R, so build it with the two redundant joints
    # locked to evaluate forward kinematics of a full 7-vector.
    from eaik.pybindings import EAIK  # pylint: disable=import-outside-toplevel

    params = _vega_ik.get_arm_ik_params(prefix)

    def fk(q):
        locked = [
            (params.lock_a, float(q[params.lock_a])),
            (params.lock_b, float(q[params.lock_b])),
        ]
        robot = EAIK.Robot(params.H, params.P, params.R6T, locked, True)
        return robot.fwdkin(np.asarray(q, dtype=float))

    rng = np.random.default_rng(0)
    n_trials = 5
    n_success = 0
    for _ in range(n_trials):
        q_true = params.lower + (params.upper - params.lower) * rng.random(7)
        target = fk(q_true)
        q_sol = _vega_ik.solve_arm_ik(target, params)
        if q_sol is None:
            continue
        pose = fk(q_sol)
        if np.linalg.norm(pose[:3, 3] - target[:3, 3]) < 1e-3:
            n_success += 1
    # Allow one failure to absorb the rare Nelder-Mead local-min case.
    assert (
        n_success >= n_trials - 1
    ), f"only {n_success}/{n_trials} roundtrips succeeded"


def test_dexmate_vega_1u_gripper_open_close(physics_client_id):
    """Opening and closing the parallel-jaw gripper drives both jaw joints between the
    open and closed states."""
    robot = DexmateVega1ULeftArmPyBulletRobot(physics_client_id)
    robot.open_fingers()
    assert np.isclose(robot.get_finger_state(), robot.open_fingers_state)
    assert np.allclose(
        robot.get_joint_positions()[robot.finger_joint_idxs[0] :],
        robot.open_fingers_state,
    )
    robot.close_fingers()
    assert np.isclose(robot.get_finger_state(), robot.closed_fingers_state)
    assert np.allclose(
        robot.get_joint_positions()[robot.finger_joint_idxs[0] :],
        robot.closed_fingers_state,
    )


def _render_pick_plan(robot, plan, path):
    """Render a kinematic plan to a video."""
    frames = []
    for state in plan:
        state.set_pybullet(robot)
        frames.append(
            capture_image(
                robot.physics_client_id,
                camera_distance=1.6,
                camera_yaw=70,
                camera_pitch=-25,
                camera_target=(0.4, 0.2, 0.65),
                image_width=480,
                image_height=360,
            )
        )
    iio.mimsave(path, frames, fps=20)


@pytest.mark.skipif(not EAIK_INSTALLED, reason="EAIK not installed")
def test_dexmate_vega_1u_pick_cube(physics_client_id, make_videos):
    """The left arm plans a top-down grasp of a cube resting on a table.

    Run pytest with --make-videos to render the plan to dexmate_vega_1u_pick.mp4.
    """
    robot = DexmateVega1ULeftArmPyBulletRobot(physics_client_id)

    # Table and a small cube on top of it, within the left arm's reach.
    table_height = 0.5
    table_id = create_pybullet_block(
        (0.5, 0.5, 0.5, 1.0), (0.15, 0.15, table_height / 2), physics_client_id
    )
    p.resetBasePositionAndOrientation(
        table_id, (0.45, 0.3, table_height / 2), (0, 0, 0, 1), physics_client_id
    )
    cube_half = 0.025
    cube_id = create_pybullet_block(
        (0.9, 0.3, 0.3, 1.0), (cube_half,) * 3, physics_client_id
    )
    p.resetBasePositionAndOrientation(
        cube_id, (0.45, 0.3, table_height + cube_half), (0, 0, 0, 1), physics_client_id
    )

    initial_state = KinematicState.from_pybullet(robot, {cube_id, table_id})

    # Top-down grasps: the gripper approaches along +z in the L_ee frame, so a roll of
    # pi points it straight down. The standoff places L_ee far enough above the cube
    # that the cube sits between the fingertips rather than up in the gripper throat
    # (where it would be occluded by the gripper body); vary the yaw to find a feasible
    # arm configuration.
    grasp_standoff = 0.18

    def grasp_generator():
        for yaw in np.linspace(-np.pi, np.pi, 16, endpoint=False):
            top_down = multiply_poses(
                Pose.from_rpy((0, 0, 0), (0, 0, float(yaw))),
                Pose.from_rpy((0, 0, 0), (np.pi, 0, 0)),
            )
            yield Pose((0.0, 0.0, grasp_standoff), top_down.orientation)

    # For a clean video, spend more time in run_smooth_motion_planning_to_pose, which
    # reruns planning and keeps the smoothest (shortest geometrically-weighted) path; for
    # normal CI, use a small budget so the test stays fast.
    if make_videos:
        planning_kwargs = {
            "max_motion_planning_time": 30.0,
            "max_motion_planning_candidates": 30,
            "max_smoothing_iters_per_step": 100,
        }
    else:
        planning_kwargs = {"max_motion_planning_time": 2.0}

    plan = get_kinematic_plan_to_pick_object(
        initial_state,
        robot,
        cube_id,
        table_id,
        collision_ids={table_id, cube_id},
        grasp_generator=grasp_generator(),
        grasp_generator_iters=60,
        **planning_kwargs,
    )

    assert plan is not None
    # The plan ends with the cube grasped (attached to the end effector).
    assert cube_id in plan[-1].attachments

    if make_videos:
        _render_pick_plan(robot, plan, "dexmate_vega_1u_pick.mp4")


def test_dexmate_vega_1u_eaik_fallback_warning(physics_client_id, monkeypatch):
    """If EAIK is unavailable, the robot falls back to pybullet IK and emits a one-time
    RuntimeWarning explaining how to install EAIK."""
    monkeypatch.setattr(_vega_ik, "EAIK_AVAILABLE", False)
    monkeypatch.setattr(dexmate_vega, "_EAIK_FALLBACK_WARNED", False)
    robot = DexmateVega1ULeftArmPyBulletRobot(physics_client_id)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert robot.default_inverse_kinematics_method == "pybullet"
        assert robot.default_inverse_kinematics_method == "pybullet"  # second call
    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(runtime_warnings) == 1
    assert "EAIK" in str(runtime_warnings[0].message)


def test_dexmate_vega_1u_bimanual_construction(physics_client_id):
    """The bimanual robot exposes two arm views over one shared body, plus a torso and
    head, with a combined action space."""
    robot = DexmateVega1UPyBulletRobot(physics_client_id)
    assert robot.get_name() == "dexmate-vega-1u"

    # Both arms share the one loaded body.
    assert robot.left_arm.robot_id == robot.robot_id == robot.right_arm.robot_id
    assert isinstance(robot.left_arm, DexmateVega1ULeftArmPyBulletRobot)
    assert isinstance(robot.right_arm, DexmateVega1URightArmPyBulletRobot)
    assert p.getNumBodies(physics_client_id) == 1

    assert robot.torso_joint_names == ["Lift", "torso_flip"]
    assert robot.head_joint_names == ["head_j1", "head_j2", "head_j3"]
    assert len(robot.get_torso_joints()) == 2
    assert len(robot.get_head_joints()) == 3

    # Whole-robot joint vector: torso (2) + left arm+fingers (9) + right (9) + head (3).
    positions = robot.get_joint_positions()
    assert len(positions) == 2 + 9 + 9 + 3
    assert robot.action_space.shape == (2 + 9 + 9 + 3,)
    assert np.allclose(robot.action_space.low, robot.action_space.low)  # finite bounds
    assert np.all(np.isfinite(robot.action_space.low))
    assert np.all(np.isfinite(robot.action_space.high))


def test_dexmate_vega_1u_bimanual_torso_moves_both_arms(physics_client_id):
    """Setting the shared torso joints moves both arms' end effectors."""
    robot = DexmateVega1UPyBulletRobot(physics_client_id)
    left_before = np.array(robot.left_arm.get_end_effector_pose().position)
    right_before = np.array(robot.right_arm.get_end_effector_pose().position)

    # Raise the prismatic lift; both arms should rise with the torso.
    robot.set_torso_joints([0.1, 0.0])
    assert np.allclose(robot.get_torso_joints(), [0.1, 0.0], atol=1e-6)
    left_after = np.array(robot.left_arm.get_end_effector_pose().position)
    right_after = np.array(robot.right_arm.get_end_effector_pose().position)
    assert not np.allclose(left_before, left_after)
    assert not np.allclose(right_before, right_after)

    # Head joints set/get round-trips.
    robot.set_head_joints([0.1, -0.2, 0.3])
    assert np.allclose(robot.get_head_joints(), [0.1, -0.2, 0.3], atol=1e-6)


@pytest.mark.skipif(not EAIK_INSTALLED, reason="EAIK not installed")
def test_dexmate_vega_1u_bimanual_both_arm_ik(physics_client_id):
    """Each arm of the bimanual robot solves IK via its own view."""
    robot = DexmateVega1UPyBulletRobot(physics_client_id)
    rng = np.random.default_rng(3)
    for arm in (robot.left_arm, robot.right_arm):
        lo = np.array(arm.joint_lower_limits)
        hi = np.array(arm.joint_upper_limits)
        q_true = rng.uniform(lo, hi)
        target = arm.forward_kinematics(q_true.tolist())
        try:
            q_sol = inverse_kinematics(arm, target, validate=True, validation_atol=1e-3)
        except InverseKinematicsError:
            pytest.fail(f"IK failed for {arm.get_name()}")
        recovered = arm.forward_kinematics(q_sol)
        assert np.allclose(recovered.position, target.position, atol=1e-3)


def test_dexmate_vega_1u_bimanual_grippers(physics_client_id):
    """Both grippers open and close independently."""
    robot = DexmateVega1UPyBulletRobot(physics_client_id)
    for arm in (robot.left_arm, robot.right_arm):
        arm.open_fingers()
        assert np.isclose(arm.get_finger_state(), arm.open_fingers_state)
        arm.close_fingers()
        assert np.isclose(arm.get_finger_state(), arm.closed_fingers_state)


@pytest.mark.skipif(not EAIK_INSTALLED, reason="EAIK not installed")
def test_dexmate_vega_1u_bimanual_pick_cubes(physics_client_id, make_videos):
    """Each arm of the bimanual robot picks its own cube: the left arm grasps a cube on
    its (+y) side and the right arm grasps a mirrored cube on its (-y) side.

    Run pytest with --make-videos to render both pick plans to
    dexmate_vega_1u_bimanual_pick.mp4.
    """
    robot = DexmateVega1UPyBulletRobot(physics_client_id)
    table_height = 0.5

    def grasp_generator():
        for yaw in np.linspace(-np.pi, np.pi, 16, endpoint=False):
            top_down = multiply_poses(
                Pose.from_rpy((0, 0, 0), (0, 0, float(yaw))),
                Pose.from_rpy((0, 0, 0), (np.pi, 0, 0)),
            )
            yield Pose((0.0, 0.0, 0.18), top_down.orientation)

    frames = []
    for arm, y, color in (
        (robot.left_arm, 0.3, (0.9, 0.3, 0.3, 1.0)),
        (robot.right_arm, -0.3, (0.3, 0.3, 0.9, 1.0)),
    ):
        table_id = create_pybullet_block(
            (0.5, 0.5, 0.5, 1.0), (0.15, 0.15, table_height / 2), physics_client_id
        )
        p.resetBasePositionAndOrientation(
            table_id, (0.45, y, table_height / 2), (0, 0, 0, 1), physics_client_id
        )
        cube_half = 0.025
        cube_id = create_pybullet_block(color, (cube_half,) * 3, physics_client_id)
        p.resetBasePositionAndOrientation(
            cube_id,
            (0.45, y, table_height + cube_half),
            (0, 0, 0, 1),
            physics_client_id,
        )

        initial_state = KinematicState.from_pybullet(arm, {cube_id, table_id})
        plan = get_kinematic_plan_to_pick_object(
            initial_state,
            arm,
            cube_id,
            table_id,
            collision_ids={table_id, cube_id},
            grasp_generator=grasp_generator(),
            grasp_generator_iters=40,
            max_motion_planning_time=2.0,
        )
        assert plan is not None, f"no pick plan for {arm.get_name()}"
        assert cube_id in plan[-1].attachments

        if make_videos:
            for state in plan:
                state.set_pybullet(arm)
                frames.append(
                    capture_image(
                        physics_client_id,
                        camera_distance=1.8,
                        camera_yaw=90,
                        camera_pitch=-25,
                        camera_target=(0.4, 0.0, 0.7),
                        image_width=480,
                        image_height=360,
                    )
                )

    if make_videos:
        iio.mimsave("dexmate_vega_1u_bimanual_pick.mp4", frames, fps=20)
