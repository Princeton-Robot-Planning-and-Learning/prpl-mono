"""Collect demonstrations using domain-specific policies.

This script mirrors kinder/scripts/collect_demos.py but uses domain-specific
policies instead of human teleoperation. The output format is identical,
so generated demos can be used with generate_demo_video.py and generate_docs.py.

Demos are saved to ../kinder/demos by default so they can be used directly
with the kinder documentation generation scripts.

Examples:
    python experiments/collect_demos_ds.py env=base_motion3d seed=0

    python experiments/collect_demos_ds.py env=base_motion3d \
        seed=0 num_demos=5

    python experiments/collect_demos_ds.py -m env=base_motion3d \
        seed='range(0,10)'
"""

import logging
import sys
import time
from pathlib import Path

import dill as pkl  # type: ignore[import-untyped]
import hydra
import kinder
import numpy as np
from gymnasium.core import Env
from omegaconf import DictConfig
from prpl_utils.utils import sample_seed_from_rng

from kinder_ds_policies.policies import create_domain_specific_policy
from kinder_ds_policies.policies.base import PolicyFailure, StatefulPolicy


def sanitize_env_id(env_id: str) -> str:
    """Remove unnecessary stuff from the env ID.

    Mirrors the function in kinder/scripts/generate_env_docs.py.
    """
    assert env_id.startswith("kinder/")
    env_id = env_id[len("kinder/") :]
    env_id = env_id.replace("/", "_")
    assert env_id[-3:-1] == "-v"
    return env_id[:-3]


def save_demo(
    demo_dir: Path,
    env_id: str,
    seed: int,
    observations: list,
    actions: list,
    rewards: list[float],
    terminated: bool,
    truncated: bool,
) -> Path:
    """Save a demo to disk in the same format as collect_demos.py."""
    timestamp = int(time.time())
    demo_subdir = demo_dir / sanitize_env_id(env_id) / str(seed)
    demo_subdir.mkdir(parents=True, exist_ok=True)
    demo_path = demo_subdir / f"{timestamp}.p"
    demo_data = {
        "env_id": env_id,
        "timestamp": timestamp,
        "seed": seed,
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminated": terminated,
        "truncated": truncated,
    }
    with open(demo_path, "wb") as f:
        pkl.dump(demo_data, f)
    return demo_path


def collect_single_demo(
    policy: StatefulPolicy,
    env: Env,
    seed: int,
    max_steps: int,
) -> tuple[list, list, list[float], bool, bool]:
    """Run the domain-specific policy and collect a single demonstration.

    Returns:
        observations: List of observations [obs_0, ..., obs_T]
        actions: List of actions [action_0, ..., action_{T-1}]
        rewards: List of rewards [reward_0, ..., reward_{T-1}]
        terminated: Whether the episode terminated successfully
        truncated: Whether the episode was truncated
    """
    observations: list = []
    actions: list = []
    rewards: list[float] = []
    terminated = False
    truncated = False

    policy.reset()
    obs, _ = env.reset(seed=seed)
    observations.append(obs)

    for _ in range(max_steps):
        try:
            action = policy(obs)
        except PolicyFailure:
            break

        obs, rew, done, trunc, _ = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(float(rew))

        if done:
            terminated = True
            break
        if trunc:
            truncated = True
            break

    return observations, actions, rewards, terminated, truncated


@hydra.main(version_base=None, config_name="config", config_path="conf/")
def _main(cfg: DictConfig) -> None:
    logging.info(f"Collecting demos: seed={cfg.seed}, env={cfg.env.env_name}")

    # Get demo directory from config or use default (kinder/demos for easy
    # integration with generate_demo_video.py and generate_env_docs.py).
    demo_dir = Path(cfg.get("demo_dir", "../kinder/demos"))
    demo_dir.mkdir(parents=True, exist_ok=True)

    # Create the environment.
    kinder.register_all_environments()
    env = kinder.make(**cfg.env.make_kwargs, render_mode="rgb_array", use_gui=False)
    env_id = cfg.env.make_kwargs["id"]

    # Create the domain-specific policy.
    policy = create_domain_specific_policy(
        cfg.env.env_name,
        observation_space=env.observation_space,
        action_space=env.action_space,
        **cfg.env.policy_kwargs,
    )

    # Collect demos.
    num_demos = cfg.get("num_demos", 1)
    rng = np.random.default_rng(cfg.seed)

    successful_demos = 0
    failed_demos = 0

    for demo_idx in range(num_demos):
        seed = sample_seed_from_rng(rng)
        logging.info(f"Collecting demo {demo_idx + 1}/{num_demos} with seed={seed}")

        observations, actions, rewards, terminated, truncated = collect_single_demo(
            policy,
            env,
            seed,
            max_steps=cfg.max_eval_steps,
        )

        if terminated:
            demo_path = save_demo(
                demo_dir,
                env_id,
                seed,
                observations,
                actions,
                rewards,
                terminated,
                truncated,
            )
            logging.info(f"Demo saved to {demo_path}")
            successful_demos += 1
        else:
            logging.warning(
                f"Demo {demo_idx + 1} did not terminate successfully "
                f"(truncated={truncated}). Not saving."
            )
            failed_demos += 1

    logging.info(f"Done. Successful: {successful_demos}, Failed: {failed_demos}")

    if successful_demos == 0:
        logging.error("No successful demos collected!")
        sys.exit(1)

    env.close()  # type: ignore[no-untyped-call]


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
