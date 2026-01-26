"""Main entry point for running RL experiments.

Examples:
    python experiments/run_experiment.py agent=ppo_basemotion3d \\
        env_id="prbench/BaseMotion3D-v0" seed=0
"""

import logging
import os

import hydra
import numpy as np
import pandas as pd
import prbench
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from prbench_rl import create_rl_agents


def _print_results_summary(metrics: dict, cfg: DictConfig) -> None:
    """Print a summary of training and evaluation results."""
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)
    print(f"Agent: {cfg.agent.name}")
    print(f"Environment: {cfg.env_id}")
    print(f"Seed: {cfg.seed}")
    print("-" * 60)

    # Training results
    if "train" in metrics and metrics["train"].get("episodic_return"):
        train_returns = metrics["train"]["episodic_return"]
        print("TRAINING:")
        print(f"  Total episodes: {len(train_returns)}")
        print(f"  Mean return: {np.mean(train_returns):.2f}")
        print(f"  Std return: {np.std(train_returns):.2f}")
        print(f"  Min return: {np.min(train_returns):.2f}")
        print(f"  Max return: {np.max(train_returns):.2f}")
        if len(train_returns) >= 10:
            print(f"  Last 10 episodes mean: {np.mean(train_returns[-10:]):.2f}")
        print("-" * 60)

    # Evaluation results
    if "eval" in metrics and metrics["eval"].get("episodic_return"):
        eval_returns = metrics["eval"]["episodic_return"]
        eval_lengths = metrics["eval"]["step_length"]
        print("EVALUATION:")
        print(f"  Episodes: {len(eval_returns)}")
        print(f"  Mean return: {np.mean(eval_returns):.2f}")
        print(f"  Std return: {np.std(eval_returns):.2f}")
        print(f"  Min return: {np.min(eval_returns):.2f}")
        print(f"  Max return: {np.max(eval_returns):.2f}")

        # Calculate success rate (episodes with positive return or > -max_steps)
        # Assuming -1 reward per step, success means finishing early
        threshold = cfg.max_episode_steps
        successes = sum(1 for l in eval_lengths if l < threshold)
        print(f"  Success rate (step < {threshold:.0f}): {successes}/{len(eval_returns)} ({100*successes/len(eval_returns):.1f}%)")

    print("=" * 60 + "\n")


@hydra.main(version_base=None, config_name="config", config_path="conf/")
def _main(cfg: DictConfig) -> None:
    logging.info(
        f"Running agent={cfg.agent.name}, env={cfg.env_id},"
        f" max_episode_steps={cfg.max_episode_steps},"
        f" seed={cfg.seed}, eval_episodes={cfg.eval_episodes}"
    )

    # Create the environment
    prbench.register_all_environments()

    # Create the agent
    agent = create_rl_agents(cfg.agent, cfg.env_id, cfg.max_episode_steps, cfg.seed)

    # Training pipeline (includes evaluation at the end)
    logging.info("Starting training and evaluation...")
    metrics = agent.train(eval_episodes=cfg.eval_episodes)

    # Save trained agent
    current_dir = HydraConfig.get().runtime.output_dir
    agent_path = os.path.join(current_dir, "agent.pkl")
    agent.save(agent_path)
    logging.info(f"Saved trained agent to {agent_path}")

    # Save training metrics
    if "train" in metrics:
        train_results_path = os.path.join(current_dir, "train_results.csv")
        pd.DataFrame(metrics["train"]).to_csv(train_results_path, index=False)
        logging.info(f"Saved training results to {train_results_path}")

    # Save evaluation metrics
    if "eval" in metrics:
        eval_results_path = os.path.join(current_dir, "eval_results.csv")
        pd.DataFrame(metrics["eval"]).to_csv(eval_results_path, index=False)
        logging.info(f"Saved evaluation results to {eval_results_path}")

    # Save config
    config_path = os.path.join(current_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        OmegaConf.save(cfg, f)
    logging.info(f"Saved config to {config_path}")

    # Print results summary
    _print_results_summary(metrics, cfg)


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
