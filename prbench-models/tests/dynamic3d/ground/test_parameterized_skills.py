"""Tests for ground parameterized skills."""

import numpy as np
import prbench
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from relational_structs.spaces import ObjectCentricBoxSpace

from prbench_models.dynamic3d.ground.parameterized_skills import (
    create_lifted_controllers,
)

prbench.register_all_environments()


def test_move_to_target_controller_one_cube():
    """Test move-to-target controller in ground environment with 1 cube."""

    # Create the environment.
    num_cubes = 1
    env = prbench.make(
        f"prbench/TidyBot3D-ground-o{num_cubes}-v0", render_mode="rgb_array"
    )
    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"TidyBot3D-ground-o{num_cubes}"
        )

    # Reset the environment and get the initial state.
    obs, _ = env.reset(seed=123)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    # Create the controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["move_to_target"]
    robot = state.get_object_from_name("robot")
    cube = state.get_object_from_name("cube1")
    object_parameters = (robot, cube)
    controller = lifted_controller.ground(object_parameters)
    params = np.array([])  # TODO

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(200):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    # else:
    # TODO add this!
    # assert False, "Controller did not terminate"

    env.close()
