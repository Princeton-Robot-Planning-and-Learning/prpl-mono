"""Tests for Transport3D parameterized skills."""

import kinder
import numpy as np
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from kinder.envs.kinematic3d.transport3d import ObjectCentricTransport3DEnv
from relational_structs.spaces import ObjectCentricBoxSpace

from kinder_models.kinematic3d.transport3d.parameterized_skills import (
    create_lifted_controllers,
)

kinder.register_all_environments()


def test_pick_controller():
    """Test pick controller in Transport3D environment."""

    env = kinder.make(
        "kinder/Transport3D-o1-v0",
        render_mode="rgb_array",
        use_gui=False,
        realistic_bg=False,
    )
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos", name_prefix="Transport3D")

    obs, _ = env.reset(seed=123)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    sim = ObjectCentricTransport3DEnv(
        num_cubes=1, use_gui=False, allow_state_access=True
    )
    controllers = create_lifted_controllers(
        env.action_space,
        sim,
    )
    lifted_controller = controllers["pick"]
    robot = state.get_object_from_name("robot")
    target = state.get_object_from_name("box0")
    object_parameters = (robot, target)
    controller = lifted_controller.ground(object_parameters)

    params = np.array([0.5, 0.0])

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


def test_pick_and_place_controller():
    """Test pick and place controller in Transport3D environment."""

    env = kinder.make(
        "kinder/Transport3D-o1-v0",
        render_mode="rgb_array",
        use_gui=False,
        realistic_bg=False,
    )
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos", name_prefix="Transport3D")

    obs, _ = env.reset(seed=123)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    sim = ObjectCentricTransport3DEnv(
        num_cubes=1, use_gui=False, allow_state_access=True
    )
    controllers = create_lifted_controllers(
        env.action_space,
        sim,
    )
    lifted_controller = controllers["pick"]
    robot = state.get_object_from_name("robot")
    target = state.get_object_from_name("box0")
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

    lifted_controller = controllers["place"]
    robot = state.get_object_from_name("robot")
    target = state.get_object_from_name("box0")
    target_table = state.get_object_from_name("table")
    object_parameters = (robot, target, target_table)
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


def test_pick_and_place_inside_box_controller():
    """Test pick and place controller inside box in Transport3D environment."""

    num_cubes = 2
    env = kinder.make(
        f"kinder/Transport3D-o{num_cubes}-v0",
        render_mode="rgb_array",
        use_gui=False,
        realistic_bg=False,
    )
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos", name_prefix="Transport3D")

    obs, _ = env.reset(seed=124)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    sim = ObjectCentricTransport3DEnv(
        num_cubes=num_cubes, use_gui=False, allow_state_access=True
    )
    controllers = create_lifted_controllers(
        env.action_space,
        sim,
    )
    lifted_controller = controllers["pick"]
    robot = state.get_object_from_name("robot")
    target = state.get_object_from_name("cube0")
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

    lifted_controller = controllers["place"]
    robot = state.get_object_from_name("robot")
    target = state.get_object_from_name("cube0")
    target_box = state.get_object_from_name("box0")
    object_parameters = (robot, target, target_box)
    controller = lifted_controller.ground(object_parameters)

    params = np.array([0.0, -0.06])

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

    lifted_controller = controllers["pick"]
    robot = state.get_object_from_name("robot")
    target = state.get_object_from_name("cube1")
    object_parameters = (robot, target)
    controller = lifted_controller.ground(object_parameters)

    params = np.array([0.5, 0.0])

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

    lifted_controller = controllers["place"]
    robot = state.get_object_from_name("robot")
    target = state.get_object_from_name("cube1")
    target_box = state.get_object_from_name("box0")
    object_parameters = (robot, target, target_box)
    controller = lifted_controller.ground(object_parameters)

    params = np.array([0.0, 0.06])

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

    lifted_controller = controllers["pick"]
    robot = state.get_object_from_name("robot")
    target = state.get_object_from_name("box0")
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

    lifted_controller = controllers["place"]
    robot = state.get_object_from_name("robot")
    target = state.get_object_from_name("box0")
    target_table = state.get_object_from_name("table")
    object_parameters = (robot, target, target_table)
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


def test_pick_cube_and_place_on_table_controller():
    """Test pick cube and place on table controller in Transport3D environment."""

    num_cubes = 2
    env = kinder.make(
        f"kinder/Transport3D-o{num_cubes}-v0",
        render_mode="rgb_array",
        use_gui=False,
        realistic_bg=False,
    )
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos", name_prefix="Transport3D")

    obs, _ = env.reset(seed=123)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    sim = ObjectCentricTransport3DEnv(
        num_cubes=num_cubes, use_gui=False, allow_state_access=True
    )
    controllers = create_lifted_controllers(
        env.action_space,
        sim,
    )
    lifted_controller = controllers["pick"]
    robot = state.get_object_from_name("robot")
    target = state.get_object_from_name("cube0")
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

    lifted_controller = controllers["place"]
    robot = state.get_object_from_name("robot")
    target = state.get_object_from_name("cube0")
    target_table = state.get_object_from_name("table")
    object_parameters = (robot, target, target_table)
    controller = lifted_controller.ground(object_parameters)

    rng = np.random.default_rng(123)
    params = controller.sample_parameters(state, rng)

    controller.reset(state, params)
    for _ in range(300):
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
