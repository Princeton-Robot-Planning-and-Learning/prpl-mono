"""Tests for set_state determinism in dynamic2d environments."""

import numpy as np
import pytest

import kinder

# All dynamic2d environments to test for rollout consistency.
DYNAMIC2D_ENVS = [
    "kinder/DynPushT2D-t1-v0",
    "kinder/DynObstruction2D-o0-v0",
    "kinder/DynObstruction2D-o2-v0",
    "kinder/DynPushPullHook2D-o0-v0",
]


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


@pytest.mark.parametrize("env_id", DYNAMIC2D_ENVS)
def test_get_next_state_matches_step(env_id):
    """Tests that get_next_state(s, a) produces the same result as set_state(s) followed
    by step(a).

    This is the core invariant that MPC rollouts depend on.
    """
    kinder.register_all_environments()
    env = kinder.make(env_id, allow_state_access=True)
    inner = env.unwrapped
    env.reset(seed=42)

    num_steps = 10
    actions = [env.action_space.sample() for _ in range(num_steps)]

    for i, action in enumerate(actions):
        state_before = inner.get_state()

        # Method 1: get_next_state
        next_via_rollout = inner.get_next_state(state_before, action)

        # Method 2: set_state + step
        inner.set_state(state_before)
        obs_via_step, _, _, _, _ = env.step(action)

        np.testing.assert_allclose(
            next_via_rollout,
            obs_via_step,
            atol=1e-6,
            err_msg=f"[{env_id}] get_next_state != set_state+step at step {i}",
        )


@pytest.mark.parametrize("env_id", DYNAMIC2D_ENVS)
def test_mpc_rollout_does_not_corrupt_state(env_id):
    """Tests the MPC pattern: save state, do multiple simulated rollouts via
    get_next_state, restore state, then step. The result should match stepping
    directly without any rollouts.

    This is the pattern used in the MPC notebook: for each real step, many
    candidate action sequences are evaluated via get_next_state, then the real
    state is restored and the best action is executed.
    """
    kinder.register_all_environments()
    env = kinder.make(env_id, allow_state_access=True)
    inner = env.unwrapped
    rng = np.random.RandomState(99)

    obs, _ = env.reset(seed=42)
    initial_state = inner.get_state()

    # Run a "ground truth" trajectory with no rollouts, starting from
    # set_state so both paths have the same collision cache state.
    num_real_steps = 8
    actions = [env.action_space.sample() for _ in range(num_real_steps)]
    inner.set_state(initial_state)
    ground_truth_states = [inner.get_state()]
    for action in actions:
        obs, _, _, _, _ = env.step(action)
        ground_truth_states.append(obs.copy())

    # Replay with MPC-style rollouts interleaved.
    inner.set_state(initial_state)
    num_candidates = 5
    horizon = 3

    for step_idx in range(num_real_steps):
        current_state = inner.get_state()

        # Simulate multiple candidate rollouts (like MPC random shooting)
        for _ in range(num_candidates):
            state = current_state
            for _ in range(horizon):
                random_action = rng.uniform(
                    low=env.action_space.low,
                    high=env.action_space.high,
                ).astype(env.action_space.dtype)
                state = inner.get_next_state(state, random_action)

        # Restore and execute the real action
        inner.set_state(current_state)
        obs, _, _, _, _ = env.step(actions[step_idx])

        np.testing.assert_allclose(
            obs,
            ground_truth_states[step_idx + 1],
            atol=1e-6,
            err_msg=(f"[{env_id}] MPC rollouts corrupted state at step {step_idx + 1}"),
        )


@pytest.mark.parametrize("env_id", DYNAMIC2D_ENVS)
def test_cross_environment_determinism(env_id):
    """Tests that planning in one environment instance and executing in another produces
    identical results.

    This is critical for MPC where a 'planning env' simulates rollouts and an 'execution
    env' runs the chosen actions.
    """
    kinder.register_all_environments()
    plan_env = kinder.make(env_id, allow_state_access=True)
    exec_env = kinder.make(env_id, allow_state_access=True)
    plan_inner = plan_env.unwrapped
    exec_inner = exec_env.unwrapped

    # Both envs start from the same seed
    plan_env.reset(seed=42)
    exec_env.reset(seed=42)

    num_steps = 8
    actions = [plan_env.action_space.sample() for _ in range(num_steps)]

    for i, action in enumerate(actions):
        state = exec_inner.get_state()

        # Plan in one env
        planned_next = plan_inner.get_next_state(state, action)

        # Execute in the other env
        exec_inner.set_state(state)
        obs_exec, _, _, _, _ = exec_env.step(action)

        np.testing.assert_allclose(
            planned_next,
            obs_exec,
            atol=1e-6,
            err_msg=f"[{env_id}] Cross-env mismatch at step {i}",
        )
