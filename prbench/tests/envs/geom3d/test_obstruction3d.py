"""Tests for obstruction3d.py."""

import numpy as np
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from prpl_utils.utils import wrap_angle
from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    remap_joint_position_plan_to_constant_distance,
    run_smooth_motion_planning_to_pose,
    smoothly_follow_end_effector_path,
)
from relational_structs.spaces import ObjectCentricBoxSpace

from prbench.envs.geom3d.obstruction3d import (
    ObjectCentricObstruction3DEnv,
    Obstruction3DEnv,
    Obstruction3DObjectCentricState,
)


def test_obstruction3d_env():
    """Tests for basic methods in obstruction3d env."""

    env = Obstruction3DEnv(use_gui=False)  # set use_gui=True to debug
    obs, _ = env.reset(seed=123)
    assert isinstance(obs, np.ndarray)

    for _ in range(10):
        act = env.action_space.sample()
        assert isinstance(obs, np.ndarray)
        obs, _, _, _, _ = env.step(act)

    # Uncomment to debug.
    # import pybullet as p
    # while True:
    #     p.getMouseEvents(env.physics_client_id)


def test_pick_place_no_obstructions():
    """Test that picking and placing succeeds when there are no obstructions."""
    # Create the real environment.
    env = Obstruction3DEnv(num_obstructions=0, use_gui=False, render_mode="rgb_array")
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    config = env._object_centric_env.config  # pylint: disable=protected-access
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    vec_obs, _ = env.reset(seed=123)
    # NOTE: we should soon make this smoother.
    oc_obs = env.observation_space.devectorize(vec_obs)
    obs = Obstruction3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Create a simulator for planning.
    sim = ObjectCentricObstruction3DEnv(num_obstructions=0, config=config)
    sim.set_state(obs)

    # Run motion planning.
    if MAKE_VIDEOS:  # make a smooth motion plan for videos
        max_candidate_plans = 20
    else:
        max_candidate_plans = 1

    # First, move to pre-grasp pose (top-down).
    x, y, z = obs.target_block_pose.position
    dz = 0.025
    pre_grasp_pose = Pose.from_rpy((x, y, z + dz), (np.pi, 0, np.pi / 2))
    joint_plan = run_smooth_motion_planning_to_pose(
        pre_grasp_pose,
        sim.robot,
        collision_ids=sim._get_collision_object_ids(),  # pylint: disable=protected-access
        end_effector_frame_to_plan_frame=Pose.identity(),
        seed=123,
        max_candidate_plans=max_candidate_plans,
    )
    assert joint_plan is not None

    # Make sure we stay below the required max_action_mag by a fair amount.
    joint_plan = remap_joint_position_plan_to_constant_distance(
        joint_plan, sim.robot, max_distance=config.max_action_mag / 2
    )

    for target_joints in joint_plan[1:]:
        delta = np.subtract(target_joints[:7], obs.joint_positions)
        delta_lst = [wrap_angle(a) for a in delta]
        action_lst = delta_lst + [0.0]
        action = np.array(action_lst, dtype=np.float32)
        vec_obs, _, _, _, _ = env.step(action)
        # NOTE: we should soon make this smoother.
        oc_obs = env.observation_space.devectorize(vec_obs)
        obs = Obstruction3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Close the gripper to grasp.
    action = np.array([0.0] * 7 + [-1.0], dtype=np.float32)
    vec_obs, _, _, _, _ = env.step(action)
    # NOTE: we should soon make this smoother.
    oc_obs = env.observation_space.devectorize(vec_obs)
    obs = Obstruction3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # The target block should now be grasped.
    assert obs.grasped_object == "target_block"

    # Move up slightly to break contact with the table.
    sim.set_state(obs)
    current_end_effector_pose = sim.robot.get_end_effector_pose()
    post_grasp_pose = Pose(
        (
            current_end_effector_pose.position[0],
            current_end_effector_pose.position[1],
            current_end_effector_pose.position[2] + 1e-2,
        ),
        current_end_effector_pose.orientation,
    )
    joint_distance_fn = create_joint_distance_fn(sim.robot)
    joint_plan = smoothly_follow_end_effector_path(
        sim.robot,
        [current_end_effector_pose, post_grasp_pose],
        sim.robot.get_joint_positions(),
        collision_ids=set(),
        joint_distance_fn=joint_distance_fn,
        max_smoothing_iters_per_step=max_candidate_plans,
    )

    joint_plan = remap_joint_position_plan_to_constant_distance(
        joint_plan, sim.robot, max_distance=config.max_action_mag / 2
    )

    for target_joints in joint_plan[1:]:
        delta = np.subtract(target_joints[:7], obs.joint_positions)
        delta_lst = [wrap_angle(a) for a in delta]
        action_lst = delta_lst + [0.0]
        action = np.array(action_lst, dtype=np.float32)
        vec_obs, _, _, _, _ = env.step(action)
        # NOTE: we should soon make this smoother.
        oc_obs = env.observation_space.devectorize(vec_obs)
        obs = Obstruction3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Determine placement pose and pre-placement pose. Place directly in the center of
    # the target region for this test.
    placement_padding = 1e-4  # leave some room to prevent collisions with surface
    block_placement_pose = Pose(
        (
            obs.target_region_pose.position[0],
            obs.target_region_pose.position[1],
            obs.target_region_pose.position[2]
            + obs.target_region_half_extents[2]
            + obs.target_block_half_extents[2]
            + placement_padding,
        ),
        obs.target_region_pose.orientation,
    )
    end_effector_placement_pose = multiply_poses(
        block_placement_pose,
        obs.grasped_object_transform,
    )
    end_effector_pre_placement_pose = Pose(
        (
            end_effector_placement_pose.position[0],
            end_effector_placement_pose.position[1],
            end_effector_placement_pose.position[2] + 1e-2,
        ),
        end_effector_placement_pose.orientation,
    )

    # We don't really have to motion plan here because there are no other objects, but
    # in general we would motion plan.
    sim.set_state(obs)
    current_end_effector_pose = sim.robot.get_end_effector_pose()
    joint_plan = smoothly_follow_end_effector_path(
        sim.robot,
        [
            current_end_effector_pose,
            end_effector_pre_placement_pose,
            end_effector_placement_pose,
        ],
        sim.robot.get_joint_positions(),
        collision_ids=set(),
        joint_distance_fn=joint_distance_fn,
        max_smoothing_iters_per_step=max_candidate_plans,
    )
    joint_plan = remap_joint_position_plan_to_constant_distance(
        joint_plan, sim.robot, max_distance=config.max_action_mag / 2
    )

    for target_joints in joint_plan[1:]:
        delta = np.subtract(target_joints[:7], obs.joint_positions)
        delta_lst = [wrap_angle(a) for a in delta]
        action_lst = delta_lst + [0.0]
        action = np.array(action_lst, dtype=np.float32)
        vec_obs, _, _, _, _ = env.step(action)
        # NOTE: we should soon make this smoother.
        oc_obs = env.observation_space.devectorize(vec_obs)
        obs = Obstruction3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Open the gripper to finish the placement. Should trigger "done" (goal reached).
    action = np.array([0.0] * 7 + [1.0], dtype=np.float32)
    vec_obs, _, done, _, _ = env.step(action)
    # NOTE: we should soon make this smoother.
    oc_obs = env.observation_space.devectorize(vec_obs)
    obs = Obstruction3DObjectCentricState(oc_obs.data, oc_obs.type_features)
    assert obs.grasped_object is None, "Object not released"
    assert done, "Goal not reached"

    # Uncomment to debug.
    # import pybullet as p
    # from pybullet_helpers.gui import visualize_pose
    # visualize_pose(end_effector_placement_pose, env.physics_client_id)
    # while True:
    #     p.getMouseEvents(env.physics_client_id)

    env.close()
