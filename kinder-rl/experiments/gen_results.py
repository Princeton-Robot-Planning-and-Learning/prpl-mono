"""Script to generate comprehensive results across all environments.

This script:
1. For each environment, reward type (sparse/dense), and agent (ppo/sac):
   - Calculates averaged success rate across seeds
   - Calculates averaged return for successful episodes
2. Measures inference time by loading checkpoints and running 100 steps

Usage:
    python kinder-rl/experiments/gen_results.py --outputs_dir kinder-rl/outputs 
    --runs_dir kinder-rl/runs
"""

import argparse
import time
from pathlib import Path

import gymnasium as gym
import kinder
import numpy as np
import pandas as pd
import torch
import yaml  # type: ignore[import-untyped]
from omegaconf import DictConfig

from kinder_rl.ppo_agent import PPOAgent
from kinder_rl.sac_agent import SACAgent


def create_simple_env(env_id: str, max_episode_steps: int):
    """Create a simple environment without training wrappers for inference timing."""
    if "kinder" in env_id:
        env = kinder.make(env_id)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    else:
        env = gym.make(env_id)
    env = gym.wrappers.FlattenObservation(env)
    return env


def measure_inference_time(
    agent_type: str,
    env_id: str,
    max_episode_steps: int,
    checkpoint_path: Path,
    hidden_size: int,
    num_steps: int = 100,
    seed: int = 0,
) -> float:
    """Measure average inference time per step.

    Args:
        agent_type: 'ppo' or 'sac'
        env_id: Environment ID
        max_episode_steps: Max steps per episode
        checkpoint_path: Path to the checkpoint file
        hidden_size: Hidden layer size used during training
        num_steps: Number of steps to run for timing
        seed: Random seed

    Returns:
        Average inference time per step in seconds
    """
    # Create environment
    env = create_simple_env(env_id, max_episode_steps)

    # Create agent with minimal config (no logging)
    cfg = DictConfig({"tf_log": False, "cuda": False, "hidden_size": hidden_size})

    agent: PPOAgent | SACAgent
    if agent_type == "ppo":
        agent = PPOAgent(
            seed=seed,
            env_id=env_id,
            max_episode_steps=max_episode_steps,
            cfg=cfg,
        )
    else:  # sac
        agent = SACAgent(
            seed=seed,
            env_id=env_id,
            max_episode_steps=max_episode_steps,
            cfg=cfg,
        )

    # Load checkpoint
    agent.load(str(checkpoint_path))

    # Set to eval mode
    if agent_type == "ppo":
        agent.agent.eval()  # type: ignore[union-attr]
    else:
        agent.actor.eval()  # type: ignore[union-attr]

    # Run timing
    obs, _ = env.reset(seed=seed)
    inference_times = []

    for _ in range(num_steps):
        obs_tensor = torch.Tensor(obs).unsqueeze(0).to(agent.device)

        start_time = time.perf_counter()
        with torch.no_grad():
            if agent_type == "ppo":
                action = agent.agent.get_action(  # type: ignore[union-attr]
                    obs_tensor, deterministic=True
                )
            else:
                action, _, _ = agent.actor.get_action(  # type: ignore[union-attr]
                    obs_tensor, deterministic=True
                )
        end_time = time.perf_counter()

        inference_times.append(end_time - start_time)

        action_np = action.cpu().numpy()[0]
        action_np = np.clip(action_np, env.action_space.low, env.action_space.high)
        obs, _, terminated, truncated, _ = env.step(action_np)

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()

    return float(np.mean(inference_times))


def process_experiment(
    env_name: str,
    reward_type: str,
    agent_type: str,
    outputs_dir: Path,
    runs_dir: Path,
) -> dict | None:
    """Process results for a single experiment configuration.

    Args:
        env_name: Environment name
        reward_type: 'sparse' or 'dense'
        agent_type: 'ppo' or 'sac'
        outputs_dir: Base outputs directory
        runs_dir: Base runs directory

    Returns:
        Dictionary with statistics
    """
    env_outputs_dir = outputs_dir / env_name
    env_runs_dir = runs_dir / env_name

    if not env_outputs_dir.exists():
        return None

    # Find all seed directories
    seed_dirs = sorted(
        [
            d
            for d in env_outputs_dir.iterdir()
            if d.is_dir() and d.name.startswith("seed")
        ]
    )

    if not seed_dirs:
        return None

    # Collect per-seed statistics
    seed_success_rates = []
    seed_successful_returns = []
    seed_inference_times = []
    total_episodes = 0

    for seed_dir in seed_dirs:
        results_dir = seed_dir / reward_type / agent_type
        eval_file = results_dir / "eval_results.csv"
        config_file = results_dir / "config.yaml"

        if not eval_file.exists() or not config_file.exists():
            continue

        # Load config to get max_episode_steps and env_id
        with open(config_file, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        max_episode_steps = config["max_episode_steps"]
        env_id = config["env_id"]
        seed = config["seed"]
        hidden_size = config["agent"]["args"]["hidden_size"]

        # Load eval results
        df = pd.read_csv(eval_file)
        total_episodes += len(df)

        # Calculate success rate (step_length < max_episode_steps means success)
        successful = df["step_length"] < max_episode_steps
        success_rate = successful.mean()
        seed_success_rates.append(success_rate)

        # Calculate average return for successful episodes
        if successful.any():
            successful_returns = df.loc[successful, "episodic_return"]
            seed_successful_returns.append(successful_returns.mean())
        else:
            seed_successful_returns.append(np.nan)

        # Measure inference time using checkpoint from runs directory
        # Checkpoint path: runs/{env}/{seed}/{reward_type}/{agent}_{env}/final_ckpt.pt
        checkpoint_dir = (
            env_runs_dir / seed_dir.name / reward_type / f"{agent_type}_{env_name}"
        )
        checkpoint_path = checkpoint_dir / "final_ckpt.pt"

        if checkpoint_path.exists():
            try:
                avg_step_time = measure_inference_time(
                    agent_type=agent_type,
                    env_id=env_id,
                    max_episode_steps=max_episode_steps,
                    checkpoint_path=checkpoint_path,
                    hidden_size=hidden_size,
                    num_steps=100,
                    seed=seed,
                )

                # Calculate average episode inference time for this seed
                # (multiply step time by step_length for each episode)
                episode_inference_times = df["step_length"] * avg_step_time
                seed_inference_times.append(episode_inference_times.mean())
            except Exception as e:
                print(
                    f"Warning: Failed to measure inference for {checkpoint_path}: {e}"
                )
                seed_inference_times.append(np.nan)
        else:
            seed_inference_times.append(np.nan)

    if not seed_success_rates:
        return None

    # Calculate overall statistics
    stats = {
        "env_name": env_name,
        "reward_type": reward_type,
        "agent_type": agent_type,
        "num_seeds": len(seed_success_rates),
        "total_episodes": total_episodes,
        "success_rate_mean": np.mean(seed_success_rates),
        "success_rate_std": (
            np.std(seed_success_rates, ddof=1) if len(seed_success_rates) > 1 else 0.0
        ),
    }

    # Calculate successful return statistics
    valid_returns = [r for r in seed_successful_returns if not np.isnan(r)]
    if valid_returns:
        stats["successful_return_mean"] = np.mean(valid_returns)
        stats["successful_return_std"] = (
            np.std(valid_returns, ddof=1) if len(valid_returns) > 1 else 0.0
        )
    else:
        stats["successful_return_mean"] = np.nan
        stats["successful_return_std"] = np.nan

    # Calculate inference time statistics
    valid_times = [t for t in seed_inference_times if not np.isnan(t)]
    if valid_times:
        stats["inference_time_mean"] = np.mean(valid_times)
        stats["inference_time_std"] = (
            np.std(valid_times, ddof=1) if len(valid_times) > 1 else 0.0
        )
    else:
        stats["inference_time_mean"] = np.nan
        stats["inference_time_std"] = np.nan

    return stats


def main(outputs_dir: Path, runs_dir: Path, output_file: Path | None = None):
    """Generate comprehensive results across all experiments.

    Args:
        outputs_dir: Directory containing experiment outputs
        runs_dir: Directory containing training runs (with checkpoints)
        output_file: Output CSV file path
    """
    kinder.register_all_environments()
    print(f"Outputs directory: {outputs_dir}")
    print(f"Runs directory: {runs_dir}")
    print("=" * 80)

    # Find all environment directories
    env_dirs = sorted([d for d in outputs_dir.iterdir() if d.is_dir()])
    print(f"Found {len(env_dirs)} environments: {[d.name for d in env_dirs]}")

    all_results = []

    for env_dir in env_dirs:
        env_name = env_dir.name
        print(f"\nProcessing environment: {env_name}")
        print("-" * 40)

        for reward_type in ["sparse", "dense"]:
            for agent_type in ["ppo", "sac"]:
                stats = process_experiment(
                    env_name=env_name,
                    reward_type=reward_type,
                    agent_type=agent_type,
                    outputs_dir=outputs_dir,
                    runs_dir=runs_dir,
                )

                if stats is not None:
                    all_results.append(stats)
                    sr_mean = stats["success_rate_mean"]
                    sr_std = stats["success_rate_std"]
                    ret_mean = stats["successful_return_mean"]
                    time_ms = stats["inference_time_mean"] * 1000
                    print(
                        f"  {reward_type}/{agent_type}: "
                        f"success={sr_mean:.2%} +/- {sr_std:.2%}, "
                        f"return={ret_mean:.2f}, time={time_ms:.2f}ms"
                    )

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))

    # Save to file
    if output_file is not None:
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate comprehensive results across all experiments",
    )

    parser.add_argument(
        "--outputs_dir",
        type=Path,
        default=Path("kinder-rl/outputs"),
        help="Directory containing experiment outputs",
    )

    parser.add_argument(
        "--runs_dir",
        type=Path,
        default=Path("kinder-rl/runs"),
        help="Directory containing training runs with checkpoints",
    )

    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path("kinder-rl/results.csv"),
        help="Output CSV file path",
    )

    args = parser.parse_args()
    main(args.outputs_dir, args.runs_dir, args.output_file)
