"""Tests for Motion3D parameterized skills."""

import kinder
import numpy as np
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from kinder.envs.kinematic3d.motion3d import ObjectCentricMotion3DEnv
from relational_structs.spaces import ObjectCentricBoxSpace

from kinder_models.kinematic3d.motion3d.parameterized_skills import (
    create_lifted_controllers,
)

kinder.register_all_environments()


def test_move_to_target_controller():
    """Test move-to-target controller in Motion3D environment."""

    env = kinder.make(
        "kinder/Motion3D-v0", render_mode="rgb_array", use_gui=False, realistic_bg=True
    )
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos", name_prefix="Motion3D")

    obs, _ = env.reset(seed=123)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    sim = ObjectCentricMotion3DEnv(allow_state_access=True)
    controllers = create_lifted_controllers(
        env.action_space,
        sim,
    )
    lifted_controller = controllers["move_to_target"]
    robot = state.get_object_from_name("robot")
    target = state.get_object_from_name("target")
    object_parameters = (robot, target)
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
