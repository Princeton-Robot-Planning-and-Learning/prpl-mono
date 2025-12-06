"""Tests for DynObstruction2D parameterized skills."""

import numpy as np
import prbench
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from relational_structs.spaces import ObjectCentricBoxSpace

from prbench_models.dynamic2d.dynobstruction2d.parameterized_skills import (
    create_lifted_controllers,
)

prbench.register_all_environments()


def test_pick_target_controller():
    """Test pick-target controller in DynObstruction2D environment."""

    # Create the environment.
    num_obstructions = 1
    env = prbench.make(
        f"prbench/DynObstruction2D-o{num_obstructions}-v0", render_mode="rgb_array"
    )
    if MAKE_VIDEOS:
        env = RecordVideo(
            env,
            "unit_test_videos",
            name_prefix=f"DynObstruction2D-o{num_obstructions}-pick-tgt",
        )

    # Reset the environment and get the initial state.
    obs, _ = env.reset(seed=123)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    # Create the controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["pick_tgt"]
    robot = state.get_object_from_name("robot")
    target_block = state.get_object_from_name("target_block")
    object_parameters = (robot, target_block)
    controller = lifted_controller.ground(object_parameters)

    # Sample parameters: grasp_ratio, side, arm_length
    rng = np.random.default_rng(123)
    params = controller.sample_parameters(state, rng)

    # Reset and execute the controller until it terminates.
    controller.reset(state, (0, 0.6, params[2]))
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


def test_pick_obstruction_controller():
    """Test pick-obstruction controller in DynObstruction2D environment."""

    # Create the environment.
    num_obstructions = 1
    env = prbench.make(
        f"prbench/DynObstruction2D-o{num_obstructions}-v0", render_mode="rgb_array"
    )
    if MAKE_VIDEOS:
        env = RecordVideo(
            env,
            "unit_test_videos",
            name_prefix=f"DynObstruction2D-o{num_obstructions}-pick-obstruction",
        )

    # Reset the environment and get the initial state.
    obs, _ = env.reset(seed=124)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    # Create the controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["pick_obstruction"]
    robot = state.get_object_from_name("robot")
    obstructions = [obj for obj in state if obj.name.startswith("obstruction")]
    assert len(obstructions) > 0, "No obstructions found"
    obstruction = obstructions[0]
    object_parameters = (robot, obstruction)
    controller = lifted_controller.ground(object_parameters)

    # Sample parameters: grasp_ratio, side, arm_length
    rng = np.random.default_rng(124)
    params = controller.sample_parameters(state, rng)

    # Reset and execute the controller.
    # Controller should be unable to pick up the block.
    controller.reset(state, params)
    is_successful = False
    for _ in range(500):
        try:
            action = controller.step()
            obs, _, _, _, _ = env.step(action)
            next_state = env.observation_space.devectorize(obs)
            controller.observe(next_state)
            state = next_state
            if controller.terminated():
                break
        except TrajectorySamplingFailure:
            break

    assert not is_successful

    env.close()


def test_place_target_controller():
    """Test place-target controller in DynObstruction2D environment."""

    # Create the environment.
    num_obstructions = 0
    env = prbench.make(
        f"prbench/DynObstruction2D-o{num_obstructions}-v0", render_mode="rgb_array"
    )
    if MAKE_VIDEOS:
        env = RecordVideo(
            env,
            "unit_test_videos",
            name_prefix=f"DynObstruction2D-o{num_obstructions}-place-tgt",
        )

    # Reset the environment and get the initial state.
    obs, _ = env.reset(seed=123)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    # First pick up the target block
    controllers = create_lifted_controllers(env.action_space)
    pick_controller = controllers["pick_tgt"]
    robot = state.get_object_from_name("robot")
    target_block = state.get_object_from_name("target_block")
    target_surface = state.get_object_from_name("target_surface")
    object_parameters_pick = (robot, target_block)
    controller = pick_controller.ground(object_parameters_pick)

    rng = np.random.default_rng(123)
    params = controller.sample_parameters(state, rng)

    # Ensure that the robot grasps the block from the top
    controller.reset(state, (0, 0.6, params[2]))
    for _ in range(500):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Pick controller did not terminate"

    # Now move to target surface
    move_controller = controllers["move_to_tgt"]
    object_parameters_place = (robot, target_block, target_surface)
    controller = move_controller.ground(object_parameters_place)
    params = controller.sample_parameters(state, rng)

    # Ensure that the robot has no rotation
    controller.reset(state, (0.25))
    for _ in range(500):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state

        if controller.terminated():
            break
    else:
        assert False, "Place controller did not terminate"

    # Finally, place target
    place_controller = controllers["place_tgt"]
    object_parameters_place = (robot, target_block)
    controller = place_controller.ground(object_parameters_place)
    params = controller.sample_parameters(state, rng)

    # Ensure that the robot has no rotation
    controller.reset(state, (0.1, 0.6, 0.25))
    is_successful = False
    for _ in range(500):
        action = controller.step()
        obs, _, terminated, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state

        if controller.terminated():
            is_successful = terminated
            break
    else:
        assert False, "Place controller did not terminate"

    assert is_successful, "Task was not successfully completed"
    env.close()


def test_place_target_obstruction_controller():
    """Test place-target controller in DynObstruction2D environment with obstruction."""

    # Create the environment.
    num_obstructions = 1
    env = prbench.make(
        f"prbench/DynObstruction2D-o{num_obstructions}-v0", render_mode="rgb_array"
    )
    if MAKE_VIDEOS:
        env = RecordVideo(
            env,
            "unit_test_videos",
            name_prefix=f"DynObstruction2D-o{num_obstructions}-push-tgt",
        )

    # Reset the environment and get the initial state.
    obs, _ = env.reset(seed=123)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    # First pick up the target block
    controllers = create_lifted_controllers(env.action_space)
    pick_controller = controllers["pick_tgt"]
    robot = state.get_object_from_name("robot")
    target_block = state.get_object_from_name("target_block")
    object_parameters_pick = (robot, target_block)
    controller = pick_controller.ground(object_parameters_pick)

    rng = np.random.default_rng(123)
    params = controller.sample_parameters(state, rng)

    # Ensure that the robot grasps the block from the top
    controller.reset(state, (0, 0.6, params[2]))
    for _ in range(500):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Pick controller did not terminate"

    # Now move to desired location, pushing all obstructions along the way.
    move_controller = controllers["push_tgt"]
    object_parameters_push = (robot, target_block)
    controller = move_controller.ground(object_parameters_push)
    params = controller.sample_parameters(state, rng)

    # Ensure that the robot goes to the target location
    controller.reset(state, (0.1, 0.6, 0.25))
    for _ in range(500):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state

        if controller.terminated():
            break
    else:
        assert False, "Place controller did not terminate"

    # Finally, place target
    place_controller = controllers["place_tgt"]
    object_parameters_place = (robot, target_block)
    controller = place_controller.ground(object_parameters_place)
    params = controller.sample_parameters(state, rng)

    # Ensure that the robot has no rotation
    controller.reset(state, (0.1, 0.6, 0.25))
    is_successful = False
    for _ in range(500):
        action = controller.step()
        obs, _, terminated, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state

        if controller.terminated():
            is_successful = terminated
            break
    else:
        assert False, "Place controller did not terminate"

    assert is_successful, "Task was not successfully completed"
    env.close()
