"""Tests for dyn_scooppour.py."""

from gymnasium.spaces import Box

import prbench


def test_dyn_scooppour_observation_random_actions():
    """Tests that observations are vectors with fixed dimensionality.

    Also tests env creation and random actions.
    """
    prbench.register_all_environments()
    env = prbench.make("prbench/DynScoopPour-o30-v0")
    assert isinstance(env.observation_space, Box)
    for _ in range(3):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)
        for _ in range(10):
            action = env.action_space.sample()
            obs, _, _, _, _ = env.step(action)
            assert env.observation_space.contains(obs)
    env.close()


def test_dyn_scooppour_small_objects_not_graspable():
    """Test that small objects cannot be grasped directly."""
    prbench.register_all_environments()
    env = prbench.make("prbench/DynScoopPour-o30-v0")

    obs, _ = env.reset()
    assert env.observation_space.contains(obs)

    # Get the object-centric environment to check collision types
    obj_env = env.unwrapped._object_centric_env

    # Import the collision type
    from prbench.envs.dynamic2d.utils import NON_GRASPABLE_COLLISION_TYPE

    # Check that small objects have the correct collision type
    state = obj_env._current_state
    from prbench.envs.dynamic2d.object_types import SmallCircleType, SmallSquareType

    for obj in state:
        if obj.is_instance(SmallCircleType) or obj.is_instance(SmallSquareType):
            # Get the pymunk body
            pymunk_body = obj_env._state_obj_to_pymunk_body[obj]
            # Check all shapes have non-graspable collision type
            for shape in pymunk_body.shapes:
                assert shape.collision_type == NON_GRASPABLE_COLLISION_TYPE

    env.close()


def test_dyn_scooppour_hook_graspable():
    """Test that the hook can be grasped."""
    prbench.register_all_environments()
    env = prbench.make("prbench/DynScoopPour-o30-v0")

    obs, _ = env.reset()
    assert env.observation_space.contains(obs)

    # Get the object-centric environment
    obj_env = env.unwrapped._object_centric_env

    # Import the collision type
    from prbench.envs.dynamic2d.utils import DYNAMIC_COLLISION_TYPE

    # Check that hook has the graspable collision type
    state = obj_env._current_state
    from prbench.envs.dynamic2d.object_types import LObjectType

    hook_objects = [obj for obj in state if obj.is_instance(LObjectType)]
    assert len(hook_objects) == 1
    hook = hook_objects[0]

    # If not held, should have DYNAMIC_COLLISION_TYPE
    if not state.get(hook, "held"):
        pymunk_body = obj_env._state_obj_to_pymunk_body[hook]
        for shape in pymunk_body.shapes:
            assert shape.collision_type == DYNAMIC_COLLISION_TYPE

    env.close()


def test_dyn_scooppour_object_counts():
    """Test that the correct number of objects are created."""
    prbench.register_all_environments()

    # Test with default 30 objects (15 circles + 15 squares)
    env = prbench.make("prbench/DynScoopPour-o30-v0")
    obs, _ = env.reset()

    obj_env = env.unwrapped._object_centric_env
    state = obj_env._current_state

    from prbench.envs.dynamic2d.object_types import SmallCircleType, SmallSquareType

    circles = [obj for obj in state if obj.is_instance(SmallCircleType)]
    squares = [obj for obj in state if obj.is_instance(SmallSquareType)]

    assert len(circles) == 15
    assert len(squares) == 15

    env.close()


def test_dyn_scooppour_middle_wall_height():
    """Test that the middle wall is half the height of the world."""
    prbench.register_all_environments()
    env = prbench.make("prbench/DynScoopPour-o30-v0")
    obs, _ = env.reset()

    obj_env = env.unwrapped._object_centric_env
    config = obj_env.config

    # Check that middle wall height is half of world max y
    expected_height = config.world_max_y / 2
    assert abs(config.middle_wall_height - expected_height) < 1e-6

    # Check that the middle wall base is at quarter height
    expected_y = config.world_max_y / 4
    assert abs(config.middle_wall_y - expected_y) < 1e-6

    env.close()


def test_dyn_scooppour_initial_positions():
    """Test that small objects are initially on the left side."""
    prbench.register_all_environments()
    env = prbench.make("prbench/DynScoopPour-o30-v0")
    obs, _ = env.reset()

    obj_env = env.unwrapped._object_centric_env
    state = obj_env._current_state
    config = obj_env.config

    from prbench.envs.dynamic2d.object_types import SmallCircleType, SmallSquareType

    # Check that all small objects start on the left side
    middle_wall_x = config.middle_wall_x
    for obj in state:
        if obj.is_instance(SmallCircleType) or obj.is_instance(SmallSquareType):
            obj_x = state.get(obj, "x")
            # Should be on left side (x < middle_wall_x)
            assert obj_x < middle_wall_x, f"Object {obj.name} at x={obj_x} should be < {middle_wall_x}"

    env.close()
