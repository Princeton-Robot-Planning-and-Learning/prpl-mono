"""Tests for ground3d.py."""

import numpy as np
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from prpl_utils.utils import wrap_angle
from pybullet_helpers.geometry import Pose, SE2Pose
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    remap_joint_position_plan_to_constant_distance,
    run_single_arm_mobile_base_motion_planning,
    smoothly_follow_end_effector_path,
)
from relational_structs.spaces import ObjectCentricBoxSpace

from prbench.envs.geom3d.ground3d import (
    Ground3DEnv,
    Ground3DObjectCentricState,
    ObjectCentricGround3DEnv,
)


def test_base_motion3d_env():
    """Tests for basic methods in base motion3D env."""

    env = Ground3DEnv(use_gui=False)  # set use_gui=True to debug
    obs, _ = env.reset(seed=123)
    assert isinstance(obs, np.ndarray)

    for _ in range(10):
        act = env.action_space.sample()
        assert isinstance(act, np.ndarray)
        obs, _, _, _, _ = env.step(act)

    # Uncomment to debug.
    # import pybullet as p

    # while True:
    #     p.getMouseEvents(env._object_centric_env.physics_client_id)


def test_motion_planning_in_ground3d_env():
    """Proof of concept that motion planning works in this environment."""

    # Create the real environment.
    env = Ground3DEnv(render_mode="rgb_array", use_gui=False)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    config = env._object_centric_env.config  # pylint: disable=protected-access
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    vec_obs, _ = env.reset(seed=123)
    # NOTE: we should soon make this smoother.
    oc_obs = env.observation_space.devectorize(vec_obs)
    obs = Ground3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Create a simulator for planning.
    sim = ObjectCentricGround3DEnv(config=config)

    target_object_pose = obs.get_object_pose("cube0")
    base_plan = run_single_arm_mobile_base_motion_planning(
        sim.robot,
        sim.robot.base.get_pose(),
        target_object_pose.to_se2(),
        collision_bodies=set(),
        seed=123,
    )
    assert base_plan is not None

    env.action_space.seed(123)
    for target_base_pose in base_plan[1:]:
        current_base_pose = obs.base_pose
        delta = target_base_pose - current_base_pose
        delta_lst = [delta.x, delta.y, delta.rot]
        action_lst = delta_lst + [0.0] * 7 + [0.0]
        action = np.array(action_lst, dtype=np.float32)
        vec_obs, _, done, _, _ = env.step(action)
        # NOTE: we should soon make this smoother.
        oc_obs = env.observation_space.devectorize(vec_obs)
        obs = Ground3DObjectCentricState(oc_obs.data, oc_obs.type_features)
        if done:
            break
    # else:
    #     assert False, "Plan did not reach goal"
    env.close()


def test_pick_place_after_moving():
    """Test moving in front of a block, picking it up, and placing it."""
    # Create the real environment.
    env = Ground3DEnv(num_cubes=2, use_gui=False, render_mode="rgb_array")
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    config = env._object_centric_env.config  # pylint: disable=protected-access
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    vec_obs, _ = env.reset(seed=123)
    oc_obs = env.observation_space.devectorize(vec_obs)
    obs = Ground3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Create a simulator for planning.
    sim = ObjectCentricGround3DEnv(num_cubes=2, config=config)
    sim.set_state(obs)

    if MAKE_VIDEOS:
        max_candidate_plans = 20
    else:
        max_candidate_plans = 1

    # Step 1: Move the base in front of cube1
    target_object_pose_temp = obs.get_object_pose("cube1").to_se2()
    target_object_pose = SE2Pose(
        target_object_pose_temp.x - 0.3,
        target_object_pose_temp.y,
        target_object_pose_temp.rot,
    )
    base_plan = run_single_arm_mobile_base_motion_planning(
        sim.robot,
        sim.robot.base.get_pose(),
        target_object_pose,
        collision_bodies=set(),
        seed=123,
    )
    assert base_plan is not None

    for target_base_pose in base_plan[1:]:
        current_base_pose = obs.base_pose
        delta = target_base_pose - current_base_pose
        delta_lst = [delta.x, delta.y, delta.rot]
        action_lst = delta_lst + [0.0] * 7 + [0.0]
        action = np.array(action_lst, dtype=np.float32)
        vec_obs, _, _, _, _ = env.step(action)
        oc_obs = env.observation_space.devectorize(vec_obs)
        obs = Ground3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Step 2: Move arm to pre-grasp pose and then to grasp pose
    sim.set_state(obs)
    x, y, z = obs.get_object_pose("cube1").position
    dz = 0.05
    pre_grasp_pose = Pose.from_rpy((x, y, z + dz), (np.pi, 0, np.pi / 2))
    grasp_pose = Pose.from_rpy((x, y, z + 0.005), (np.pi, 0, np.pi / 2))

    joint_distance_fn = create_joint_distance_fn(sim.robot.arm)
    joint_plan = smoothly_follow_end_effector_path(
        sim.robot.arm,
        [sim.robot.arm.get_end_effector_pose(), pre_grasp_pose, grasp_pose],
        sim.robot.arm.get_joint_positions(),
        collision_ids=set(),
        joint_distance_fn=joint_distance_fn,
        max_smoothing_iters_per_step=max_candidate_plans,
    )
    assert joint_plan is not None

    joint_plan = remap_joint_position_plan_to_constant_distance(
        joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
    )

    for target_joints in joint_plan[1:]:
        delta = np.subtract(target_joints[:7], obs.joint_positions)
        delta_lst = [wrap_angle(a) for a in delta]
        action_lst = [0.0] * 3 + delta_lst + [0.0]
        action = np.array(action_lst, dtype=np.float32)
        vec_obs, _, _, _, _ = env.step(action)
        oc_obs = env.observation_space.devectorize(vec_obs)
        obs = Ground3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Step 3: Close the gripper to grasp cube1 (takes multiple steps)
    for _ in range(5):
        action = np.array([0.0] * 3 + [0.0] * 7 + [-1.0], dtype=np.float32)
        vec_obs, _, _, _, _ = env.step(action)
        oc_obs = env.observation_space.devectorize(vec_obs)
        obs = Ground3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # The cube should now be grasped
    assert obs.grasped_object == "cube1"

    # Step 4: Move up to lift the cube
    sim.set_state(obs)
    current_end_effector_pose = sim.robot.arm.get_end_effector_pose()
    lifted_pose = Pose(
        (
            current_end_effector_pose.position[0],
            current_end_effector_pose.position[1],
            current_end_effector_pose.position[2] + 0.1,
        ),
        current_end_effector_pose.orientation,
    )
    joint_plan = smoothly_follow_end_effector_path(
        sim.robot.arm,
        [current_end_effector_pose, lifted_pose],
        sim.robot.arm.get_joint_positions(),
        collision_ids=set(),
        joint_distance_fn=joint_distance_fn,
        max_smoothing_iters_per_step=max_candidate_plans,
    )

    joint_plan = remap_joint_position_plan_to_constant_distance(
        joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
    )

    for target_joints in joint_plan[1:]:
        delta = np.subtract(target_joints[:7], obs.joint_positions)
        delta_lst = [wrap_angle(a) for a in delta]
        action_lst = [0.0] * 3 + delta_lst + [0.0]
        action = np.array(action_lst, dtype=np.float32)
        vec_obs, _, _, _, _ = env.step(action)
        oc_obs = env.observation_space.devectorize(vec_obs)
        obs = Ground3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Verify cube is still grasped after lifting
    assert obs.grasped_object == "cube1"

    # Step 5: Place it back down
    sim.set_state(obs)
    current_end_effector_pose = sim.robot.arm.get_end_effector_pose()
    placement_pose = Pose(
        (
            current_end_effector_pose.position[0] + 0.1,
            current_end_effector_pose.position[1],
            obs.get_object_half_extents("cube1")[2] + 0.01,
        ),
        current_end_effector_pose.orientation,
    )

    joint_plan = smoothly_follow_end_effector_path(
        sim.robot.arm,
        [current_end_effector_pose, placement_pose],
        sim.robot.arm.get_joint_positions(),
        collision_ids=set(),
        joint_distance_fn=joint_distance_fn,
        max_smoothing_iters_per_step=max_candidate_plans,
    )

    if joint_plan is not None:
        joint_plan = remap_joint_position_plan_to_constant_distance(
            joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
        )

        for target_joints in joint_plan[1:]:
            delta = np.subtract(target_joints[:7], obs.joint_positions)
            delta_lst = [wrap_angle(a) for a in delta]
            action_lst = [0.0] * 3 + delta_lst + [0.0]
            action = np.array(action_lst, dtype=np.float32)
            vec_obs, _, _, _, _ = env.step(action)
            oc_obs = env.observation_space.devectorize(vec_obs)
            obs = Ground3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Step 6: Open the gripper to place the cube
    for _ in range(5):
        action = np.array([0.0] * 3 + [0.0] * 7 + [1.0], dtype=np.float32)
        vec_obs, _, _, _, _ = env.step(action)
        oc_obs = env.observation_space.devectorize(vec_obs)
        obs = Ground3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    assert obs.grasped_object is None, "Object not released"

    env.close()
