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
        f" seed={cfg.seed}"
        f"   in mode={cfg.mode}"
    )

    # Create the environment
    prbench.register_all_environments()

    # Create the agent
    agent = create_rl_agents(cfg.agent, cfg.env_id, cfg.max_episode_steps, cfg.seed)

    if cfg.mode == "train":
        # Training pipeline
        logging.info("Starting training...")
        train_metrics = agent.train()

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
        if cfg.get("eval_episodes") is not None:
            eval_episodes = cfg.eval_episodes
        else:
            eval_episodes = 10  # Default value
        logging.info(f"Evaluating for {eval_episodes} episodes.")
        if cfg.get("load_agent_path"):
            agent.load(cfg.load_agent_path)
            logging.info(f"Loaded agent from {cfg.load_agent_path}")

        agent.eval()
        eval_metrics = agent.evaluate(eval_episodes)

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


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
