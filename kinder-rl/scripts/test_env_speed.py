"""Compare step speed between TidyBot3D environments with parallelization."""

import time
from typing import Callable

import gymnasium as gym
import kinder
import numpy as np

kinder.register_all_environments()

ENVS = [
    "kinder/TidyBot3D-base_motion-o1-v0",
    (
        "kinder/TidyBot3D-tool_use-lab2_kitchen-o5-"
        "sweep_the_blocks_into_the_top_drawer_of_the_kitchen_island-v0"
    ),
]

NUM_STEPS = 100
NUM_ENVS_LIST = [1, 2, 4, 8]


def make_env(env_name: str) -> Callable[[], gym.Env]:
    """Factory function for creating environments."""

    def thunk() -> gym.Env:
        return kinder.make(env_name, render_mode="rgb_array")

    return thunk


def test_single_env(env_name: str) -> None:
    """Test single environment speed."""
    print("\n  [Single Env]")

    start = time.time()
    env = kinder.make(env_name, render_mode="rgb_array")
    print(f"  Create: {time.time() - start:.3f}s")
    print(f"  Obs space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    start = time.time()
    env.reset()
    print(f"  Reset: {time.time() - start:.3f}s")

    start = time.time()
    for _ in range(NUM_STEPS):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()
    elapsed = time.time() - start
    print(f"  {NUM_STEPS} steps: {elapsed:.2f}s ({NUM_STEPS/elapsed:.1f} steps/sec)")

    env.close()  # type: ignore[no-untyped-call]


def test_sync_vector_env(env_name: str, n_envs: int) -> None:
    """Test SyncVectorEnv speed."""
    print(f"\n  [SyncVectorEnv x{n_envs}]")

    start = time.time()
    envs = gym.vector.SyncVectorEnv([make_env(env_name) for _ in range(n_envs)])
    print(f"  Create: {time.time() - start:.3f}s")

    start = time.time()
    envs.reset()
    print(f"  Reset: {time.time() - start:.3f}s")

    start = time.time()
    for _ in range(NUM_STEPS):
        actions = np.array([envs.single_action_space.sample() for _ in range(n_envs)])
        envs.step(actions)
    elapsed = time.time() - start
    total_steps = NUM_STEPS * n_envs
    steps_per_sec = total_steps / elapsed
    print(
        f"  {NUM_STEPS} batches ({total_steps} total): "
        f"{elapsed:.2f}s ({steps_per_sec:.1f} steps/sec)"
    )

    envs.close()  # type: ignore[no-untyped-call]


def test_async_vector_env(env_name: str, n_envs: int) -> None:
    """Test AsyncVectorEnv speed."""
    print(f"\n  [AsyncVectorEnv x{n_envs}]")

    try:
        start = time.time()
        envs = gym.vector.AsyncVectorEnv([make_env(env_name) for _ in range(n_envs)])
        print(f"  Create: {time.time() - start:.3f}s")

        start = time.time()
        envs.reset()
        print(f"  Reset: {time.time() - start:.3f}s")

        start = time.time()
        for _ in range(NUM_STEPS):
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(n_envs)]
            )
            envs.step(actions)
        elapsed = time.time() - start
        total_steps = NUM_STEPS * n_envs
        steps_per_sec = total_steps / elapsed
        print(
            f"  {NUM_STEPS} batches ({total_steps} total): "
            f"{elapsed:.2f}s ({steps_per_sec:.1f} steps/sec)"
        )

        envs.close()  # type: ignore[no-untyped-call]
    except Exception as e:  # pylint: disable=broad-exception-caught
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
