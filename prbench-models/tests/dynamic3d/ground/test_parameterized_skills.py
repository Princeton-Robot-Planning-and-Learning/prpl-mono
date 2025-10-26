"""Tests for ground parameterized skills."""

from prbench_models.dynamic3d.ground.parameterized_skills import create_lifted_controllers

import prbench
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo


prbench.register_all_environments()


def test_move_to_target_controller_one_cube():
    """Test move-to-target controller in ground environment with 1 cube."""

    num_cubes = 1

    env = prbench.make(f"prbench/TidyBot3D-ground-o{num_cubes}-v0", render_mode="rgb_array")

    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"TidyBot3D-ground-o{num_cubes}"
        )

    controllers = create_lifted_controllers(env.action_space)
    move_to_target_controller = controllers["move_to_target"]

    import ipdb; ipdb.set_trace()
