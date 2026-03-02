"""Tests for Obstruction3D parameterized skills."""

from typing import Any

import kinder
import numpy as np
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from kinder.envs.kinematic3d.obstruction3d import ObjectCentricObstruction3DEnv
from kinder.envs.kinematic3d.save_utils import DEFAULT_DEMOS_DIR, save_demo
from relational_structs.spaces import ObjectCentricBoxSpace

from kinder_models.kinematic3d.obstruction3d.parameterized_skills import (
    create_lifted_controllers,
)

# Flag to enable trajectory saving
SAVE_TRAJECTORIES = MAKE_VIDEOS

kinder.register_all_environments()


def test_pick_controller():
    """Test pick controller in Obstruction3D environment."""

    env = kinder.make(
        "kinder/Obstruction3D-o1-v0",
        render_mode="rgb_array",
        use_gui=False,
        realistic_bg=True,
    )
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos", name_prefix="Obstruction3D")

    obs, _ = env.reset(seed=123)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    sim = ObjectCentricObstruction3DEnv(
        num_obstructions=1, use_gui=False, allow_state_access=True
    )
    controllers = create_lifted_controllers(
        env.action_space,
        sim,
    )
    lifted_controller = controllers["pick"]
    robot = state.get_object_from_name("robot")
    target_block = state.get_object_from_name("target_block")
    object_parameters = (robot, target_block)
    controller = lifted_controller.ground(object_parameters)

    rng = np.random.default_rng(123)
    params = controller.sample_parameters(state, rng)

    controller.reset(state, params)
    for _ in range(500):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    env.close()


def test_pick_place_controller():
    """Test pick and place controller in Obstruction3D environment."""

    seed = 123
    env = kinder.make(
        "kinder/Obstruction3D-o0-v0",
        render_mode="rgb_array",
        use_gui=False,
        realistic_bg=True,
    )
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos", name_prefix="Obstruction3D")

    obs, _ = env.reset(seed=seed)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    # Initialize trajectory collection
    traj_observations: list[Any] = [obs.copy()]
    traj_actions: list[Any] = []
    traj_rewards: list[float] = []
    ep_terminated = False
    ep_truncated = False

    sim = ObjectCentricObstruction3DEnv(num_obstructions=0, allow_state_access=True)
    controllers = create_lifted_controllers(
        env.action_space,
        sim,
    )
    lifted_controller = controllers["pick"]
    robot = state.get_object_from_name("robot")
    target_block = state.get_object_from_name("target_block")
    object_parameters = (robot, target_block)
    controller = lifted_controller.ground(object_parameters)

    rng = np.random.default_rng(123)
    params = controller.sample_parameters(state, rng)

    controller.reset(state, params)
    for _ in range(500):
        action = controller.step()
        obs, reward, terminated, truncated, _ = env.step(action)
        # Collect trajectory data
        traj_observations.append(obs.copy())
        traj_actions.append(action.copy())
        traj_rewards.append(float(reward))
        ep_terminated = ep_terminated or terminated
        ep_truncated = ep_truncated or truncated
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    lifted_controller = controllers["place"]
    robot = state.get_object_from_name("robot")
    target_region = state.get_object_from_name("target_region")
    object_parameters = (robot, target_region)
    controller = lifted_controller.ground(object_parameters)

    rng = np.random.default_rng(123)
    params = controller.sample_parameters(state, rng)

    controller.reset(state, params)
    for _ in range(500):
        action = controller.step()
        obs, reward, terminated, truncated, _ = env.step(action)
        # Collect trajectory data
        traj_observations.append(obs.copy())
        traj_actions.append(action.copy())
        traj_rewards.append(float(reward))
        ep_terminated = ep_terminated or terminated
        ep_truncated = ep_truncated or truncated
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # Save trajectory to pickle file
    if SAVE_TRAJECTORIES and len(traj_actions) > 0:
        demo_path = save_demo(
            demo_dir=DEFAULT_DEMOS_DIR,
            env_id="kinder/Obstruction3D-o0-v0",
            seed=seed,
            observations=traj_observations,
            actions=traj_actions,
            rewards=traj_rewards,
            terminated=ep_terminated,
            truncated=ep_truncated,
        )
        print(f"Trajectory saved to {demo_path}")
        print(f"  Observations: {len(traj_observations)}, Actions: {len(traj_actions)}")

    env.close()
