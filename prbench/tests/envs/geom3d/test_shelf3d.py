"""Tests for ground3d.py."""

import numpy as np
import pytest
from gymnasium.wrappers import RecordVideo
from prpl_utils.utils import wrap_angle
from pybullet_helpers.geometry import Pose, SE2Pose
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    remap_joint_position_plan_to_constant_distance,
    run_single_arm_mobile_base_motion_planning,
    smoothly_follow_end_effector_path,
    run_smooth_motion_planning_to_pose,
    run_motion_planning,
)
from relational_structs.spaces import ObjectCentricBoxSpace
from prbench.envs.geom3d.utils import extend_joints_to_include_fingers

from prbench.envs.geom3d.shelf3d import ObjectCentricShelf3DEnv, Shelf3DEnv, \
    Shelf3DObjectCentricState
from tests.conftest import MAKE_VIDEOS


@pytest.fixture(scope="module")
def env():
    """Create a shared environment for all tests in this module."""
    environment = Shelf3DEnv(num_cubes=2, use_gui=True, render_mode="rgb_array")
    yield environment
    environment.close()


def test_shelf3d_env(env):  # pylint: disable=redefined-outer-name
    """Tests for basic methods in shelf env."""
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



def test_pick_place(env):  # pylint: disable=redefined-outer-name
    """Test picking and placing a cube into the shelf."""
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    config = env._object_centric_env.config  # pylint: disable=protected-access

    test_env = env
    if MAKE_VIDEOS:
        test_env = RecordVideo(env, "unit_test_videos")

    vec_obs, _ = test_env.reset(seed=123)
    oc_obs = test_env.observation_space.devectorize(vec_obs)
    obs = Shelf3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Create a simulator for planning.
    sim = ObjectCentricShelf3DEnv(num_cubes=2, config=config, use_gui=False)
    sim.set_state(obs)

    # Extract body IDs for collision checking
    shelf_id = sim._shelf_id
    base_id = sim.robot.base.robot_id

    if MAKE_VIDEOS:
        max_candidate_plans = 20
    else:
        max_candidate_plans = 1

    # Step 1: Move the base in front of cube1
    target_object_pose_temp = obs.get_object_pose("cube1").to_se2()
    target_object_pose = SE2Pose(
        target_object_pose_temp.x - 0.5,
        target_object_pose_temp.y,
        target_object_pose_temp.rot,
    )
    base_plan = run_single_arm_mobile_base_motion_planning(
        sim.robot,
        sim.robot.base.get_pose(),
        target_object_pose,
        collision_bodies={shelf_id},
        seed=123,
    )
    assert base_plan is not None

    for target_base_pose in base_plan[1:]:
        current_base_pose = obs.base_pose
        delta = target_base_pose - current_base_pose
        delta_lst = [delta.x, delta.y, delta.rot]
        action_lst = delta_lst + [0.0] * 7 + [0.0]
        action = np.array(action_lst, dtype=np.float32)
        vec_obs, _, _, _, _ = test_env.step(action)
        oc_obs = test_env.observation_space.devectorize(vec_obs)
        obs = Shelf3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Step 2: Move arm to pre-grasp pose and then to grasp pose
    sim.set_state(obs)
    x, y, z = obs.get_object_pose("cube1").position
    dz = 0.05
    pre_grasp_pose = Pose.from_rpy((x, y, z + dz), (np.pi, 0, np.pi / 2))
    grasp_pose = Pose.from_rpy((x, y, z + 0.005), (np.pi, 0, np.pi / 2))

    joint_distance_fn = create_joint_distance_fn(sim.robot.arm)
    joint_plan = run_smooth_motion_planning_to_pose(
        pre_grasp_pose,
        sim.robot.arm,
        collision_ids={base_id, shelf_id},
        end_effector_frame_to_plan_frame=Pose.identity(),
        seed=123,
        max_candidate_plans=max_candidate_plans,
    )
    joint_plan = remap_joint_position_plan_to_constant_distance(
        joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
    )

    for target_joints in joint_plan[1:]:
        delta = np.subtract(target_joints[:7], obs.joint_positions)
        delta_lst = [wrap_angle(a) for a in delta]
        action_lst = [0.0] * 3 + delta_lst + [0.0]
        action = np.array(action_lst, dtype=np.float32)
        vec_obs, _, _, _, _ = test_env.step(action)
        oc_obs = test_env.observation_space.devectorize(vec_obs)
        obs = Shelf3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Step 2.5: Move down to grasp cube1
    sim.set_state(obs)
    joint_plan = smoothly_follow_end_effector_path(
        sim.robot.arm,
        [sim.robot.arm.get_end_effector_pose(), grasp_pose],
        sim.robot.arm.get_joint_positions(),
        collision_ids={shelf_id, base_id},
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
        vec_obs, _, _, _, _ = test_env.step(action)
        oc_obs = test_env.observation_space.devectorize(vec_obs)
        obs = Shelf3DObjectCentricState(oc_obs.data, oc_obs.type_features)
    

    # Step 3: Close the gripper to grasp cube1 (takes multiple steps)
    for _ in range(5):
        action = np.array([0.0] * 3 + [0.0] * 7 + [-1.0], dtype=np.float32)
        vec_obs, _, _, _, _ = test_env.step(action)
        oc_obs = test_env.observation_space.devectorize(vec_obs)
        obs = Shelf3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # The cube should now be grasped
    assert obs.grasped_object == "cube1"

    # Step 4: Retract the arm
    sim.set_state(obs)
    joint_plan = run_motion_planning(
        sim.robot.arm,
        sim.robot.arm.get_joint_positions(),
        extend_joints_to_include_fingers(sim.config.initial_joints),
        collision_bodies={shelf_id, base_id},
        seed=123,
        physics_client_id=sim.physics_client_id,
        held_object=sim._grasped_object_id,
        base_link_to_held_obj=sim._grasped_object_transform,
    )
    joint_plan = remap_joint_position_plan_to_constant_distance(
        joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
    )
    for target_joints in joint_plan[1:]:
        delta = np.subtract(target_joints[:7], obs.joint_positions)
        delta_lst = [wrap_angle(a) for a in delta]
        action_lst = [0.0] * 3 + delta_lst + [0.0]
        action = np.array(action_lst, dtype=np.float32)
        vec_obs, _, _, _, _ = test_env.step(action)
        oc_obs = test_env.observation_space.devectorize(vec_obs)
        obs = Shelf3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Verify cube is still grasped after lifting
    assert obs.grasped_object == "cube1"

    # Step 5: Move the base in front of the shelf
    sim.set_state(obs)
    shelf_pose = obs.get_object_pose("shelf")
    # Position the base in front of the shelf (shelf is at y=1.5)
    target_shelf_base_pose = SE2Pose(
        shelf_pose.position[0],
        shelf_pose.position[1] - 0.8,
        np.pi / 2,
    )
    base_plan = run_single_arm_mobile_base_motion_planning(
        sim.robot,
        sim.robot.base.get_pose(),
        target_shelf_base_pose,
        collision_bodies={shelf_id},
        seed=456,
    )
    assert base_plan is not None

    for target_base_pose in base_plan[1:]:
        current_base_pose = obs.base_pose
        delta = target_base_pose - current_base_pose
        delta_lst = [delta.x, delta.y, delta.rot]
        action_lst = delta_lst + [0.0] * 7 + [0.0]
        action = np.array(action_lst, dtype=np.float32)
        vec_obs, _, _, _, _ = test_env.step(action)
        oc_obs = test_env.observation_space.devectorize(vec_obs)
        obs = Shelf3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Step 6: Move arm to place the cube on the first shelf layer
    sim.set_state(obs)
    # Calculate placement pose: first shelf layer at z = shelf_spacing (0.254)
    # Place slightly above to account for the cube and shelf surface
    place_x = shelf_pose.position[0]
    place_y = shelf_pose.position[1] - 0.05
    place_z = config.shelf_spacing + config.shelf_height / 2 + config.block_half_extents[0] + 0.015
    pre_place_x = place_x
    pre_place_y = place_y - 0.1
    pre_place_z = place_z
    pre_place_pose = Pose.from_rpy((pre_place_x, pre_place_y, pre_place_z), (-np.pi / 2, np.pi, 0))
    place_pose = Pose.from_rpy((place_x, place_y, place_z), (-np.pi / 2, np.pi, 0))

    joint_plan = run_smooth_motion_planning_to_pose(
        pre_place_pose,
        sim.robot.arm,
        collision_ids={base_id, shelf_id},
        end_effector_frame_to_plan_frame=Pose.identity(),
        seed=123,
        max_candidate_plans=max_candidate_plans,
        held_object=sim._grasped_object_id,
        base_link_to_held_obj=sim._grasped_object_transform,
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
        vec_obs, _, _, _, _ = test_env.step(action)
        oc_obs = test_env.observation_space.devectorize(vec_obs)
        obs = Shelf3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Step 6.5: move to place pose
    sim.set_state(obs)
    joint_plan = smoothly_follow_end_effector_path(
        sim.robot.arm,
        [sim.robot.arm.get_end_effector_pose(), place_pose],
        sim.robot.arm.get_joint_positions(),
        collision_ids={shelf_id, base_id},
        joint_distance_fn=joint_distance_fn,
        max_smoothing_iters_per_step=max_candidate_plans,
        held_object=sim._grasped_object_id,
        base_link_to_held_obj=sim._grasped_object_transform,
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
        vec_obs, _, _, _, _ = test_env.step(action)
        oc_obs = test_env.observation_space.devectorize(vec_obs)
        obs = Shelf3DObjectCentricState(oc_obs.data, oc_obs.type_features)


    # Verify cube is still grasped before releasing
    assert obs.grasped_object == "cube1"

    # Step 7: Open the gripper to release cube1 (takes multiple steps)
    for _ in range(5):
        action = np.array([0.0] * 3 + [0.0] * 7 + [1.0], dtype=np.float32)
        vec_obs, _, _, _, _ = test_env.step(action)
        oc_obs = test_env.observation_space.devectorize(vec_obs)
        obs = Shelf3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # The cube should no longer be grasped
    assert obs.grasped_object is None

    # Verify the cube is approximately on the shelf
    cube_pose = obs.get_object_pose("cube1")
    assert abs(cube_pose.position[2] - place_z) < 0.1, "Cube should be at shelf height"

    # Uncomment to debug.
    # import pybullet as p
    # while True:
    #     p.getMouseEvents(env._object_centric_env.physics_client_id)

    test_env.close()
