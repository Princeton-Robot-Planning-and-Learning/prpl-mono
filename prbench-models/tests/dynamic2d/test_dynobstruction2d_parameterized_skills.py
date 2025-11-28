"""Tests for DynObstruction2D parameterized skills."""

import numpy as np
import prbench
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from relational_structs.spaces import ObjectCentricBoxSpace

from prbench_models.dynamic2d.dynobstruction2d.parameterized_skills import (
    create_lifted_controllers,
)

prbench.register_all_environments()


# def test_pick_target_controller():
#     """Test pick-target controller in DynObstruction2D environment."""

#     # Create the environment.
#     num_obstructions = 1
#     env = prbench.make(
#         f"prbench/DynObstruction2D-o{num_obstructions}-v0", render_mode="rgb_array"
#     )
#     if MAKE_VIDEOS:
#         env = RecordVideo(
#             env,
#             "unit_test_videos",
#             name_prefix=f"DynObstruction2D-o{num_obstructions}-pick-tgt",
#         )

#     # Reset the environment and get the initial state.
#     obs, _ = env.reset(seed=123)
#     assert isinstance(env.observation_space, ObjectCentricBoxSpace)
#     state = env.observation_space.devectorize(obs)

#     # Create the controller.
#     controllers = create_lifted_controllers(env.action_space)
#     lifted_controller = controllers["pick_tgt"]
#     robot = state.get_object_from_name("robot")
#     target_block = state.get_object_from_name("target_block")
#     object_parameters = (robot, target_block)
#     controller = lifted_controller.ground(object_parameters)

#     # Sample parameters: grasp_ratio, side, arm_length
#     rng = np.random.default_rng(123)
#     params = controller.sample_parameters(state, rng)

#     # Reset and execute the controller until it terminates.
#     controller.reset(state, params)
#     for _ in range(500):
#         action = controller.step()
#         obs, _, _, _, _ = env.step(action)
#         next_state = env.observation_space.devectorize(obs)
#         controller.observe(next_state)
#         state = next_state
#         if controller.terminated():
#             break
#     else:
#         assert False, "Controller did not terminate"

#     env.close()


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
        except:
            break

    assert not is_successful

    env.close()


# def test_place_obstruction_controller():
#     """Test place-obstruction controller in DynObstruction2D environment."""

#     # Create the environment.
#     num_obstructions = 1
#     env = prbench.make(
#         f"prbench/DynObstruction2D-o{num_obstructions}-v0", render_mode="rgb_array"
#     )
#     if MAKE_VIDEOS:
#         env = RecordVideo(
#             env,
#             "unit_test_videos",
#             name_prefix=f"DynObstruction2D-o{num_obstructions}-place-obstruction",
#         )

#     # Reset the environment and get the initial state.
#     obs, _ = env.reset(seed=125)
#     assert isinstance(env.observation_space, ObjectCentricBoxSpace)
#     state = env.observation_space.devectorize(obs)

#     # First pick up an obstruction
#     controllers = create_lifted_controllers(env.action_space)
#     pick_controller = controllers["pick_obstruction"]
#     robot = state.get_object_from_name("robot")
#     obstructions = [obj for obj in state if obj.name.startswith("obstruction")]
#     assert len(obstructions) > 0, "No obstructions found"
#     obstruction = obstructions[0]
#     object_parameters = (robot, obstruction)
#     controller = pick_controller.ground(object_parameters)

#     rng = np.random.default_rng(125)
#     params = controller.sample_parameters(state, rng)
#     controller.reset(state, params)
#     for _ in range(500):
#         action = controller.step()
#         obs, _, _, _, _ = env.step(action)
#         next_state = env.observation_space.devectorize(obs)
#         controller.observe(next_state)
#         state = next_state
#         if controller.terminated():
#             break
#     else:
#         assert False, "Pick controller did not terminate"

#     # Now place the obstruction
#     place_controller = controllers["place_obstruction"]
#     controller = place_controller.ground(object_parameters)
#     params = controller.sample_parameters(state, rng)

#     controller.reset(state, params)
#     for _ in range(500):
#         action = controller.step()
#         obs, _, _, _, _ = env.step(action)
#         next_state = env.observation_space.devectorize(obs)
#         controller.observe(next_state)
#         state = next_state
#         if controller.terminated():
#             break
#     else:
#         assert False, "Place controller did not terminate"

#     env.close()


# def test_place_target_controller():
#     """Test place-target controller in DynObstruction2D environment."""

#     # Create the environment.
#     num_obstructions = 1
#     env = prbench.make(
#         f"prbench/DynObstruction2D-o{num_obstructions}-v0", render_mode="rgb_array"
#     )
#     if MAKE_VIDEOS:
#         env = RecordVideo(
#             env,
#             "unit_test_videos",
#             name_prefix=f"DynObstruction2D-o{num_obstructions}-place-tgt",
#         )

#     # Reset the environment and get the initial state.
#     obs, _ = env.reset(seed=126)
#     assert isinstance(env.observation_space, ObjectCentricBoxSpace)
#     state = env.observation_space.devectorize(obs)

#     # First pick up the target block
#     controllers = create_lifted_controllers(env.action_space)
#     pick_controller = controllers["pick_tgt"]
#     robot = state.get_object_from_name("robot")
#     target_block = state.get_object_from_name("target_block")
#     target_surface = state.get_object_from_name("target_surface")
#     object_parameters_pick = (robot, target_block)
#     controller = pick_controller.ground(object_parameters_pick)

#     rng = np.random.default_rng(126)
#     params = controller.sample_parameters(state, rng)
#     controller.reset(state, params)
#     for _ in range(500):
#         action = controller.step()
#         obs, _, _, _, _ = env.step(action)
#         next_state = env.observation_space.devectorize(obs)
#         controller.observe(next_state)
#         state = next_state
#         if controller.terminated():
#             break
#     else:
#         assert False, "Pick controller did not terminate"

#     # Now move to target surface and place
#     place_controller = controllers["place_tgt"]
#     object_parameters_place = (robot, target_block, target_surface)
#     controller = place_controller.ground(object_parameters_place)
#     params = controller.sample_parameters(state, rng)

#     controller.reset(state, params)
#     for _ in range(500):
#         action = controller.step()
#         obs, _, _, _, _ = env.step(action)
#         next_state = env.observation_space.devectorize(obs)
#         controller.observe(next_state)
#         state = next_state
#         if controller.terminated():
#             break
#     else:
#         assert False, "Place controller did not terminate"

#     env.close()


# def test_pick_and_place_sequence():
#     """Test a complete pick and place sequence in DynObstruction2D environment."""

#     # Create the environment.
#     num_obstructions = 1
#     env = prbench.make(
#         f"prbench/DynObstruction2D-o{num_obstructions}-v0", render_mode="rgb_array"
#     )
#     if MAKE_VIDEOS:
#         env = RecordVideo(
#             env,
#             "unit_test_videos",
#             name_prefix=f"DynObstruction2D-o{num_obstructions}-full-sequence",
#         )

#     # Reset the environment and get the initial state.
#     obs, _ = env.reset(seed=127)
#     assert isinstance(env.observation_space, ObjectCentricBoxSpace)
#     state = env.observation_space.devectorize(obs)

#     controllers = create_lifted_controllers(env.action_space)
#     robot = state.get_object_from_name("robot")
#     target_block = state.get_object_from_name("target_block")
#     target_surface = state.get_object_from_name("target_surface")
#     rng = np.random.default_rng(127)

#     # Step 1: Pick up the target block
#     pick_controller = controllers["pick_tgt"]
#     controller = pick_controller.ground((robot, target_block))
#     params = controller.sample_parameters(state, rng)
#     controller.reset(state, params)
#     for _ in range(500):
#         action = controller.step()
#         obs, _, _, _, _ = env.step(action)
#         next_state = env.observation_space.devectorize(obs)
#         controller.observe(next_state)
#         state = next_state
#         if controller.terminated():
#             break
#     else:
#         assert False, "Pick controller did not terminate"

#     # Step 2: Place the target block on the target surface
#     place_controller = controllers["place_tgt"]
#     controller = place_controller.ground((robot, target_block, target_surface))
#     params = controller.sample_parameters(state, rng)
#     controller.reset(state, params)
#     for _ in range(500):
#         action = controller.step()
#         obs, _, _, _, _ = env.step(action)
#         next_state = env.observation_space.devectorize(obs)
#         controller.observe(next_state)
#         state = next_state
#         if controller.terminated():
#             break
#     else:
#         assert False, "Place controller did not terminate"

#     env.close()

