"""Tests for PyBullet motion planning."""

from functools import partial

import numpy as np
import pybullet as p

from pybullet_helpers.geometry import Pose, set_pose
from pybullet_helpers.inverse_kinematics import (
    inverse_kinematics,
    sample_joints_from_task_space_bounds,
)
from pybullet_helpers.joint import get_joint_infos
from pybullet_helpers.math_utils import geometric_sequence
from pybullet_helpers.motion_planning import (
    MotionPlanningHyperparameters,
    get_joint_positions_distance,
    run_base_motion_planning,
    run_motion_planning,
    select_shortest_motion_plan,
    smoothly_follow_end_effector_path,
)
from pybullet_helpers.robots.fetch import FetchPyBulletRobot
from pybullet_helpers.robots.kinova import KinovaGen3RobotiqGripperPyBulletRobot
from pybullet_helpers.utils import create_pybullet_block, get_assets_path


def test_run_motion_planning(physics_client_id):
    """Tests for run_motion_planning()."""
    base_pose = Pose(position=(0.75, 0.7441, 0.0))
    seed = 123
    robot = FetchPyBulletRobot(physics_client_id, base_pose=base_pose)
    joint_initial = robot.get_joint_positions()
    # Should succeed with a path of length 1.
    joint_target = list(joint_initial)
    path = run_motion_planning(
        robot,
        joint_initial,
        joint_target,
        collision_bodies=set(),
        seed=seed,
        physics_client_id=physics_client_id,
    )
    assert len(path) == 1
    assert np.allclose(path[0], joint_initial)
    assert np.allclose(path[-1], joint_target)
    # Should succeed, no collisions.
    ee_pose = robot.get_end_effector_pose()
    ee_target_position = np.add(ee_pose.position, (0.0, 0.0, -0.05))
    ee_target = Pose(ee_target_position, ee_pose.orientation)
    joint_target = inverse_kinematics(robot, ee_target, validate=True)
    path = run_motion_planning(
        robot,
        joint_initial,
        joint_target,
        collision_bodies=set(),
        seed=seed,
        physics_client_id=physics_client_id,
    )
    assert np.allclose(path[0], joint_initial)
    assert np.allclose(path[-1], joint_target)
    # Should fail because the target collides with the table.
    table_pose = (1.35, 0.75, 0.0)
    table_orientation = [0.0, 0.0, 0.0, 1.0]
    table_urdf_path = get_assets_path() / "urdf" / "table.urdf"
    table_id = p.loadURDF(
        str(table_urdf_path), useFixedBase=True, physicsClientId=physics_client_id
    )
    p.resetBasePositionAndOrientation(
        table_id, table_pose, table_orientation, physicsClientId=physics_client_id
    )
    ee_pose = robot.get_end_effector_pose()
    ee_target_position = np.add(ee_pose.position, (0.0, 0.0, -0.6))
    ee_target = Pose(ee_target_position, ee_pose.orientation)
    joint_target = inverse_kinematics(robot, ee_target, validate=True)
    path = run_motion_planning(
        robot,
        joint_initial,
        joint_target,
        collision_bodies={table_id},
        seed=seed,
        physics_client_id=physics_client_id,
    )
    assert path is None
    # Should fail because the initial state collides with the table.
    path = run_motion_planning(
        robot,
        joint_target,
        joint_initial,
        collision_bodies={table_id},
        seed=seed,
        physics_client_id=physics_client_id,
    )
    assert path is None
    # Should succeed, but will need to move the arm up to avoid the obstacle.
    block_pose = (1.35, 0.6, 0.5)
    block_orientation = [0.0, 0.0, 0.0, 1.0]
    block_id = create_pybullet_block(
        color=(1.0, 0.0, 0.0, 1.0),
        half_extents=(0.2, 0.01, 0.3),
        physics_client_id=physics_client_id,
    )
    p.resetBasePositionAndOrientation(
        block_id, block_pose, block_orientation, physicsClientId=physics_client_id
    )
    ee_target_position = (1.35, 0.4, 0.6)
    ee_target = Pose(ee_target_position, ee_pose.orientation)
    joint_target = inverse_kinematics(robot, ee_target, validate=True)
    path = run_motion_planning(
        robot,
        joint_initial,
        joint_target,
        collision_bodies={table_id, block_id},
        seed=seed,
        physics_client_id=physics_client_id,
    )
    assert path is not None
    p.removeBody(block_id, physicsClientId=physics_client_id)
    # Should fail because the hyperparameters are too limited.
    hyperparameters = MotionPlanningHyperparameters(
        birrt_num_attempts=1, birrt_num_iters=1
    )
    block_id = create_pybullet_block(
        color=(1.0, 0.0, 0.0, 1.0),
        half_extents=(0.2, 0.01, 0.3),
        physics_client_id=physics_client_id,
    )
    p.resetBasePositionAndOrientation(
        block_id, block_pose, block_orientation, physicsClientId=physics_client_id
    )
    path = run_motion_planning(
        robot,
        joint_initial,
        joint_target,
        collision_bodies={table_id, block_id},
        seed=seed,
        physics_client_id=physics_client_id,
        hyperparameters=hyperparameters,
    )
    assert path is None
    p.removeBody(block_id, physicsClientId=physics_client_id)


def test_motion_planning_additional_constraint(physics_client_id):
    """Tests for run_motion_planning with an additional state constraint."""
    initial_joints = [
        2.5,
        -1.5,
        -2.1,
        1.9,
        2.9,
        1.5,
        -2.6,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    target_joints = list(initial_joints)
    target_joints[0] = -0.5
    robot = KinovaGen3RobotiqGripperPyBulletRobot(physics_client_id)
    robot.set_joints(target_joints)
    ee_target = robot.get_end_effector_pose()
    robot.set_joints(initial_joints)
    ee_initial = robot.get_end_effector_pose()
    seed = 123
    initial_roll = p.getEulerFromQuaternion(ee_initial.orientation)[0]
    target_roll = p.getEulerFromQuaternion(ee_target.orientation)[0]
    assert np.isclose(initial_roll, target_roll)
    roll_tolerance = 0.5

    # Add blocks to prevent direct movement.
    block1_pose = (0.0, 0.5, 0.1)
    block1_orientation = (0.0, 0.0, 0.0, 1.0)
    block1_id = create_pybullet_block(
        color=(1.0, 0.0, 0.0, 1.0),
        half_extents=(0.1, 0.1, 0.1),
        physics_client_id=physics_client_id,
    )
    p.resetBasePositionAndOrientation(
        block1_id, block1_pose, block1_orientation, physicsClientId=physics_client_id
    )
    block2_pose = (0.0, -0.5, 0.1)
    block2_orientation = (0.0, 0.0, 0.0, 1.0)
    block2_id = create_pybullet_block(
        color=(1.0, 0.0, 0.0, 1.0),
        half_extents=(0.1, 0.1, 0.1),
        physics_client_id=physics_client_id,
    )
    p.resetBasePositionAndOrientation(
        block2_id, block2_pose, block2_orientation, physicsClientId=physics_client_id
    )
    collision_ids = {block1_id, block2_id}

    def _check_hand_orientation(joint_positions):
        robot.set_joints(joint_positions)
        ee_pose = robot.get_end_effector_pose()
        # Prevent rotating the hand by too much.
        roll = p.getEulerFromQuaternion(ee_pose.orientation)[0]
        return abs(roll - initial_roll) < roll_tolerance

    # Running motion planning WITHOUT the constraint creates a path that
    # violates it.
    robot.set_joints(initial_joints)
    path = run_motion_planning(
        robot,
        initial_joints,
        target_joints,
        collision_bodies=collision_ids,
        seed=seed,
        physics_client_id=physics_client_id,
    )
    assert any(not _check_hand_orientation(s) for s in path)

    # Running motion planning WITH the constraint creates a path that works.
    robot.set_joints(initial_joints)
    path = run_motion_planning(
        robot,
        initial_joints,
        target_joints,
        collision_bodies=collision_ids,
        seed=seed,
        physics_client_id=physics_client_id,
        additional_state_constraint_fn=_check_hand_orientation,
    )
    assert all(_check_hand_orientation(s) for s in path)


def test_task_space_motion_planning(physics_client_id):
    """Tests for run_motion_planning with a custom sampling function."""
    initial_joints = [
        2.5,
        -1.5,
        -2.1,
        1.9,
        2.9,
        1.5,
        -2.6,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    target_joints = list(initial_joints)
    target_joints[0] = -0.5
    robot = KinovaGen3RobotiqGripperPyBulletRobot(physics_client_id)
    robot.set_joints(target_joints)
    ee_target = robot.get_end_effector_pose()
    robot.set_joints(initial_joints)
    ee_initial = robot.get_end_effector_pose()
    seed = 123
    initial_roll = p.getEulerFromQuaternion(ee_initial.orientation)[0]
    target_roll = p.getEulerFromQuaternion(ee_target.orientation)[0]
    assert np.isclose(initial_roll, target_roll)

    # Add blocks to prevent direct movement.
    block1_pose = (0.0, 0.5, 0.1)
    block1_orientation = (0.0, 0.0, 0.0, 1.0)
    block1_id = create_pybullet_block(
        color=(1.0, 0.0, 0.0, 1.0),
        half_extents=(0.1, 0.1, 0.1),
        physics_client_id=physics_client_id,
    )
    p.resetBasePositionAndOrientation(
        block1_id, block1_pose, block1_orientation, physicsClientId=physics_client_id
    )
    block2_pose = (0.0, -0.5, 0.1)
    block2_orientation = (0.0, 0.0, 0.0, 1.0)
    block2_id = create_pybullet_block(
        color=(1.0, 0.0, 0.0, 1.0),
        half_extents=(0.1, 0.1, 0.1),
        physics_client_id=physics_client_id,
    )
    p.resetBasePositionAndOrientation(
        block2_id, block2_pose, block2_orientation, physicsClientId=physics_client_id
    )
    collision_ids = {block1_id, block2_id}

    # Create task space sampler that constrains roll.
    rng = np.random.default_rng(123)

    def _sample_fn(current_joints):
        del current_joints  # not used
        return sample_joints_from_task_space_bounds(
            rng,
            robot,
            -2.0,
            2.0,
            -2.0,
            2.0,
            -1.0,
            1.0,
            initial_roll - 1e-6,
            initial_roll + 1e-6,
            -np.pi,
            np.pi,
            -np.pi,
            np.pi,
        )

    # Running motion planning WITH the constraint creates a path that works.
    robot.set_joints(initial_joints)
    path = run_motion_planning(
        robot,
        initial_joints,
        target_joints,
        collision_bodies=collision_ids,
        seed=seed,
        physics_client_id=physics_client_id,
        sampling_fn=_sample_fn,
    )
    assert path is not None


def test_select_shortest_motion_plan(physics_client_id):
    """Tests for select_shortest_motion_plan()."""

    robot = FetchPyBulletRobot(physics_client_id)
    joint_initial = robot.get_joint_positions()
    joint_space = robot.action_space
    joint_space.seed(123)
    joint_perturbed = joint_space.sample()

    longer_plan = [joint_initial, joint_perturbed]
    shorter_plan = [joint_initial, joint_initial]

    joint_infos = get_joint_infos(
        robot.robot_id, robot.arm_joints, robot.physics_client_id
    )
    weights = [1.0] * len(joint_infos)
    dist_fn = partial(
        get_joint_positions_distance,
        robot,
        joint_infos,
        metric="weighted_joints",
        weights=weights,
    )

    ret_plan = select_shortest_motion_plan([shorter_plan, longer_plan], dist_fn)
    assert ret_plan is shorter_plan


def test_smoothly_follow_end_effector_path(physics_client_id):
    """Tests for smoothly_follow_end_effector_path()."""
    robot = KinovaGen3RobotiqGripperPyBulletRobot(physics_client_id)
    initial_joints = robot.get_joint_positions()

    # Constructed using the joint visualizer.
    waypoints = [
        Pose(
            position=(0.5741068720817566, 0.06966808438301086, 0.181205615401268),
            orientation=(
                -0.539107084274292,
                0.4131602644920349,
                -0.45173147320747375,
                0.5784468650817871,
            ),
        ),
        Pose(
            position=(0.5822595357894897, 0.07682612538337708, 0.26875561475753784),
            orientation=(
                -0.47179025411605835,
                0.399135559797287,
                -0.4618518352508545,
                0.6362371444702148,
            ),
        ),
        Pose(
            position=(0.4648642838001251, 0.35892781615257263, 0.2687576711177826),
            orientation=(
                -0.5593748092651367,
                0.26266589760780334,
                -0.2804473042488098,
                0.7344765067100525,
            ),
        ),
        Pose(
            position=(0.4627046287059784, 0.3524962365627289, 0.36749306321144104),
            orientation=(
                -0.47692224383354187,
                0.26456379890441895,
                -0.27181705832481384,
                0.7928850650787354,
            ),
        ),
        Pose(
            position=(0.22300678491592407, 0.5372304320335388, 0.3674944043159485),
            orientation=(
                -0.5293239951133728,
                0.1313922107219696,
                -0.05620665103197098,
                0.8362972140312195,
            ),
        ),
        Pose(
            position=(0.04950527101755142, 0.5752283930778503, 0.38660216331481934),
            orientation=(
                -0.6207988262176514,
                0.0051186466589570045,
                0.012963184155523777,
                0.7838460206985474,
            ),
        ),
        Pose(
            position=(-0.18992702662944794, 0.5452213287353516, 0.3866019546985626),
            orientation=(
                -0.608161985874176,
                -0.12472657114267349,
                0.17647969722747803,
                0.7638306021690369,
            ),
        ),
        Pose(
            position=(-0.34986400604248047, 0.4592755436897278, 0.3866013288497925),
            orientation=(
                -0.6007586121559143,
                -0.1565486639738083,
                0.36909976601600647,
                0.6916263699531555,
            ),
        ),
        Pose(
            position=(-0.5148458480834961, 0.2753393352031708, 0.3379387855529785),
            orientation=(
                -0.5763673186302185,
                -0.31119245290756226,
                0.47536608576774597,
                0.5873559713363647,
            ),
        ),
    ]

    # Add blocks to prevent direct movement.
    block1_pose = (0.0, 0.5, 0.1)
    block1_orientation = (0.0, 0.0, 0.0, 1.0)
    block1_id = create_pybullet_block(
        color=(1.0, 0.0, 0.0, 1.0),
        half_extents=(0.1, 0.1, 0.1),
        physics_client_id=physics_client_id,
    )
    p.resetBasePositionAndOrientation(
        block1_id, block1_pose, block1_orientation, physicsClientId=physics_client_id
    )
    block2_pose = (0.0, -0.5, 0.1)
    block2_orientation = (0.0, 0.0, 0.0, 1.0)
    block2_id = create_pybullet_block(
        color=(1.0, 0.0, 0.0, 1.0),
        half_extents=(0.1, 0.1, 0.1),
        physics_client_id=physics_client_id,
    )
    p.resetBasePositionAndOrientation(
        block2_id, block2_pose, block2_orientation, physicsClientId=physics_client_id
    )
    collision_ids = {block1_id, block2_id}

    weights = geometric_sequence(0.9, len(robot.arm_joints))
    joint_infos = get_joint_infos(
        robot.robot_id, robot.arm_joints, robot.physics_client_id
    )

    def joint_distance_fn(pt1, pt2):
        return get_joint_positions_distance(
            robot,
            joint_infos,
            pt1,
            pt2,
            metric="weighted_joints",
            weights=weights,
        )

    joint_waypoints = smoothly_follow_end_effector_path(
        robot,
        waypoints,
        initial_joints,
        collision_ids,
        joint_distance_fn,
        max_time=0.5,
    )
    assert len(joint_waypoints) == len(waypoints) + 1
    assert np.allclose(joint_waypoints[0], initial_joints)
    recovered_waypoints = [robot.forward_kinematics(w) for w in joint_waypoints[1:]]
    for i in range(len(waypoints)):
        w1 = waypoints[i]
        w2 = recovered_waypoints[i]
        assert w1.allclose(w2, atol=1e-3)

    # Uncomment to visualize.
    # from pybullet_helpers.gui import visualize_pose
    # from pybullet_helpers.joint import interpolate_joints
    # robot.set_joints(initial_joints)
    # for i in range(len(joint_waypoints) - 1):
    #     pt1 = joint_waypoints[i]
    #     pt2 = joint_waypoints[i + 1]
    #     ee2 = waypoints[i + 1]
    #     visualize_pose(ee2, physics_client_id)
    #     import time; time.sleep(0.5)
    #     for s in interpolate_joints(joint_infos, pt1, pt2, num_interp_per_unit=100):
    #         robot.set_joints(s)
    #         import time; time.sleep(0.01)


def test_base_motion_planning_to_goal():
    """Tests for run_base_motion_planning_to_goal()."""

    # Uncomment for debugging.
    # from pybullet_helpers.gui import create_gui_connection
    # physics_client_id = create_gui_connection(camera_pitch=-90)
    physics_client_id = p.connect(p.DIRECT)

    robot = KinovaGen3RobotiqGripperPyBulletRobot(physics_client_id, fixed_base=False)
    platform = create_pybullet_block(
        (0.3, 1.0, 0.3, 1.0), (0.1, 0.1, 0.1), physics_client_id
    )

    obstacle = create_pybullet_block(
        (1.0, 0.0, 0.0, 1.0), (0.2, 0.2, 0.1), physics_client_id
    )
    obstacle_pose = Pose((1.0, 0.0, 0.0))
    set_pose(obstacle, obstacle_pose, physics_client_id)

    collision_bodies = {obstacle}
    seed = 123

    initial_pose = robot.get_base_pose()
    target_pose = Pose.from_rpy((2.0, 0.0, 0.0), (0.0, 0.0, np.pi))

    position_lower_bounds = (-5.0, -5.0)
    position_upper_bounds = (5.0, 5.0)

    def goal_check(pt):
        return (
            np.linalg.norm(np.subtract(pt.position, target_pose.position)) < 0.5
            and np.linalg.norm(np.subtract(pt.orientation, target_pose.orientation))
            < 1.0
        )

    plan = run_base_motion_planning(
        robot,
        initial_pose,
        goal_check,
        position_lower_bounds,
        position_upper_bounds,
        collision_bodies,
        seed,
        physics_client_id,
        platform=platform,
    )
    assert plan is not None

    # Uncomment to visualize.
    # import time
    # for base_pose in plan:
    #     robot.set_base(base_pose)
    #     time.sleep(0.5)
