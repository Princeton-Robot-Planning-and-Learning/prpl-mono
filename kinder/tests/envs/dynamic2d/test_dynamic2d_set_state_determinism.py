"""Tests for set_state determinism in dynamic2d environments."""

import numpy as np

import kinder


def test_pusht2d_set_state_determinism():
    """Tests that set_state followed by the same actions reproduces the same
    trajectory."""
    kinder.register_all_environments()
    env = kinder.make("kinder/DynPushT2D-t1-v0", allow_state_access=True)
    inner = env.unwrapped
    obs, _ = env.reset(seed=42)

    # Generate a fixed sequence of actions
    num_steps = 30
    actions = []
    for _ in range(num_steps):
        actions.append(env.action_space.sample())
    # Push toward the T-block to ensure collision
    for i in range(5):
        actions[i] = np.array([0.05, 0.05], dtype=np.float64)

    # Run the full trajectory, recording states
    states = [obs.copy()]
    for action in actions:
        obs, _, _, _, _ = env.step(action)
        states.append(obs.copy())

    # Pick a midpoint where physics interactions have happened
    midpoint = 10
    inner.set_state(states[midpoint])

    # Replay remaining actions and compare
    for i, action in enumerate(actions[midpoint:]):
        obs, _, _, _, _ = env.step(action)
        expected = states[midpoint + 1 + i]
        np.testing.assert_allclose(
            obs,
            expected,
            atol=1e-6,
            err_msg=f"State mismatch at step {midpoint + 1 + i} after set_state",
        )


def test_obstruction2d_set_state_determinism():
    """Tests set_state determinism for the obstruction environment."""
    kinder.register_all_environments()
    env = kinder.make("kinder/DynObstruction2D-o0-v0", allow_state_access=True)
    inner = env.unwrapped
    obs, _ = env.reset(seed=42)

    # Generate actions
    num_steps = 20
    actions = []
    for _ in range(num_steps):
        actions.append(env.action_space.sample())

    # Run full trajectory
    states = [obs.copy()]
    for action in actions:
        obs, _, _, _, _ = env.step(action)
        states.append(obs.copy())

    # Reset to midpoint and replay
    midpoint = 5
    inner.set_state(states[midpoint])

    for i, action in enumerate(actions[midpoint:]):
        obs, _, _, _, _ = env.step(action)
        expected = states[midpoint + 1 + i]
        np.testing.assert_allclose(
            obs,
            expected,
            atol=1e-6,
            err_msg=f"State mismatch at step {midpoint + 1 + i} after set_state",
        )
