"""Main entry point for running RL experiments.

Examples:
    python experiments/run_experiment.py agent=random env=obstruction2d-o0 seed=0
"""

import logging
import os

import hydra
import pandas as pd
import prbench
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from prbench_rl import create_rl_agents


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


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
