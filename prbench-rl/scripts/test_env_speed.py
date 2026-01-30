"""Compare step speed between TidyBot3D environments with parallelization."""

import time
import numpy as np
import gymnasium as gym
import prbench

prbench.register_all_environments()

ENVS = [
    "prbench/TidyBot3D-base_motion-o1-v0",
    "prbench/TidyBot3D-tool_use-lab2_kitchen-o5-sweep_the_blocks_into_the_top_drawer_of_the_kitchen_island-v0",
]

NUM_STEPS = 100
NUM_ENVS_LIST = [1, 2, 4, 8]


def make_env(env_id):
    """Factory function for creating environments."""
    def thunk():
        return prbench.make(env_id, render_mode="rgb_array")
    return thunk


def test_single_env(env_id):
    """Test single environment speed."""
    print(f"\n  [Single Env]")

    start = time.time()
    env = prbench.make(env_id, render_mode="rgb_array")
    print(f"  Create: {time.time() - start:.3f}s")
    print(f"  Obs space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    start = time.time()
    obs, info = env.reset()
    print(f"  Reset: {time.time() - start:.3f}s")

    start = time.time()
    for i in range(NUM_STEPS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    elapsed = time.time() - start
    print(f"  {NUM_STEPS} steps: {elapsed:.2f}s ({NUM_STEPS/elapsed:.1f} steps/sec)")

    env.close()


def test_sync_vector_env(env_id, num_envs):
    """Test SyncVectorEnv speed."""
    print(f"\n  [SyncVectorEnv x{num_envs}]")

    start = time.time()
    envs = gym.vector.SyncVectorEnv([make_env(env_id) for _ in range(num_envs)])
    print(f"  Create: {time.time() - start:.3f}s")

    start = time.time()
    obs, info = envs.reset()
    print(f"  Reset: {time.time() - start:.3f}s")

    start = time.time()
    for i in range(NUM_STEPS):
        actions = np.array([envs.single_action_space.sample() for _ in range(num_envs)])
        obs, reward, term, trunc, info = envs.step(actions)
    elapsed = time.time() - start
    total_steps = NUM_STEPS * num_envs
    print(f"  {NUM_STEPS} batches ({total_steps} total steps): {elapsed:.2f}s ({total_steps/elapsed:.1f} steps/sec)")

    envs.close()


def test_async_vector_env(env_id, num_envs):
    """Test AsyncVectorEnv speed."""
    print(f"\n  [AsyncVectorEnv x{num_envs}]")

    try:
        start = time.time()
        envs = gym.vector.AsyncVectorEnv([make_env(env_id) for _ in range(num_envs)])
        print(f"  Create: {time.time() - start:.3f}s")

        start = time.time()
        obs, info = envs.reset()
        print(f"  Reset: {time.time() - start:.3f}s")

        start = time.time()
        for i in range(NUM_STEPS):
            actions = np.array([envs.single_action_space.sample() for _ in range(num_envs)])
            obs, reward, term, trunc, info = envs.step(actions)
        elapsed = time.time() - start
        total_steps = NUM_STEPS * num_envs
        print(f"  {NUM_STEPS} batches ({total_steps} total steps): {elapsed:.2f}s ({total_steps/elapsed:.1f} steps/sec)")

        envs.close()
    except Exception as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    for env_id in ENVS:
        print(f"\n{'='*70}")
        print(f"Testing: {env_id}")
        print("=" * 70)

        # Single env baseline
        test_single_env(env_id)

        # SyncVectorEnv with different num_envs
        for num_envs in NUM_ENVS_LIST:
            test_sync_vector_env(env_id, num_envs)

        # AsyncVectorEnv with different num_envs
        for num_envs in NUM_ENVS_LIST:
            test_async_vector_env(env_id, num_envs)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
