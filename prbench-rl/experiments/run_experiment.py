"""Main entry point for running RL experiments.

Examples:
    python experiments/run_experiment.py agent=random env=obstruction2d-o0 seed=0
"""

import logging
import os
from typing import Any

import hydra
import pandas as pd
import prbench
from gymnasium.core import Env
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from prbench_rl import BaseRLAgent, create_rl_agents


@hydra.main(version_base=None, config_name="config", config_path="conf/")
def _main(cfg: DictConfig) -> None:
    logging.info(
        f"Running agent={cfg.agent.name}, env={cfg.env.env_name}, seed={cfg.seed}"
    )

    # Create the environment
    prbench.register_all_environments()
    env = prbench.make(**cfg.env.make_kwargs)

    # Create the agent
    agent = create_rl_agents(cfg.agent, env, cfg.seed)

    if cfg.mode == "train":
        # Training pipeline
        logging.info("Starting training...")
        train_metrics = agent.train_with_env(env)

        # Save trained agent
        current_dir = HydraConfig.get().runtime.output_dir
        agent_path = os.path.join(current_dir, "agent.pkl")
        agent.save(agent_path)
        logging.info(f"Saved trained agent to {agent_path}")

        # Save training metrics
        results_path = os.path.join(current_dir, "train_results.csv")
        pd.DataFrame(train_metrics).to_csv(results_path, index=False)
        logging.info(f"Saved training results to {results_path}")

    elif cfg.mode == "eval":
        # Evaluation pipeline
        logging.info("Starting evaluation...")
        if cfg.get("load_agent_path"):
            agent.load(cfg.load_agent_path)
            logging.info(f"Loaded agent from {cfg.load_agent_path}")

        agent.eval()
        eval_metrics = _run_evaluation(
            agent,
            env,
            cfg.eval_episodes,
            cfg.max_eval_steps,
            cfg.seed,
        )

        # Save evaluation results
        current_dir = HydraConfig.get().runtime.output_dir
        results_path = os.path.join(current_dir, "eval_results.csv")
        pd.DataFrame(eval_metrics).to_csv(results_path, index=False)
        logging.info(f"Saved evaluation results to {results_path}")

    # Save config
    current_dir = HydraConfig.get().runtime.output_dir
    config_path = os.path.join(current_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        OmegaConf.save(cfg, f)
    logging.info(f"Saved config to {config_path}")


def _run_evaluation(
    agent: BaseRLAgent,
    env: Env,
    num_episodes: int,
    max_steps: int,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Run evaluation episodes."""
    eval_metrics = []

    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode if seed is not None else None)
        agent.reset(obs, info)

        episode_reward = 0.0
        episode_steps = 0
        success = False

        for _ in range(max_steps):
            action = agent.step()
            obs, reward, done, truncated, info = env.step(action)
            agent.update(obs, float(reward), done or truncated, info)

            episode_reward += float(reward)
            episode_steps += 1

            if done:
                success = True
                break
            if truncated:
                break

        eval_metrics.append(
            {
                "episode": episode,
                "episode_reward": episode_reward,
                "episode_steps": episode_steps,
                "success": success,
            }
        )

    return eval_metrics


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
