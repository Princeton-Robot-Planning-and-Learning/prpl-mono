"""Tests for dyn_scooppour.py."""

from gymnasium.spaces import Box

import prbench
from prbench.envs.dynamic2d.object_types import (
    LObjectType,
    SmallCircleType,
    SmallSquareType,
)


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


def test_dyn_scooppour_object_counts():
    """Test that the correct number of objects are created."""
    prbench.register_all_environments()

    # Test with default 30 objects (15 circles + 15 squares)
    env = prbench.make("prbench/DynScoopPour-o30-v0")
    obs, _ = env.reset()

    # Use public API to access object-centric state
    state = env.observation_space.devectorize(obs)

    circles = [obj for obj in state if obj.is_instance(SmallCircleType)]
    squares = [obj for obj in state if obj.is_instance(SmallSquareType)]
    hooks = [obj for obj in state if obj.is_instance(LObjectType)]

    assert len(circles) == 15
    assert len(squares) == 15
    assert len(hooks) == 1

    env.close()


def test_dyn_scooppour_initial_positions():
    """Test that small objects are initially on the left side."""
    prbench.register_all_environments()
    env = prbench.make("prbench/DynScoopPour-o30-v0")
    obs, _ = env.reset()

    # Use public API to access object-centric state
    state = env.observation_space.devectorize(obs)

    # The middle wall should be approximately in the center of the world
    # We'll infer this from the observation space bounds
    world_max_x = env.observation_space.high[0]  # Assuming first feature is x
    middle_x = world_max_x / 2

    # Check that all small objects start on the left side
    for obj in state:
        if obj.is_instance(SmallCircleType) or obj.is_instance(SmallSquareType):
            obj_x = state.get(obj, "x")
            # Should be on left side (x < middle)
            assert (
                obj_x < middle_x
            ), f"Object {obj.name} at x={obj_x} should be < {middle_x}"

    env.close()
