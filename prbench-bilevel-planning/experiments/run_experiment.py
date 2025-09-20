"""Main entry point for running experiments.

Examples:
    python experiments/run_experiment.py env=obstruction2d-o0 seed=0

    python experiments/run_experiment.py -m env=obstruction2d-o0 seed='range(0,10)'

    python experiments/run_experiment.py -m env=obstruction2d-o0 seed=0 \
        samples_per_step=1,5,10

    python experiments/run_experiment.py -m env=stickbutton2d-b3 seed=0 \
        max_abstract_plans=1,5,10,20
"""

import logging
import os

import hydra
import numpy as np
import pandas as pd
import prbench
from gymnasium.core import Env
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from prpl_utils.utils import sample_seed_from_rng, timer

from prbench_bilevel_planning.agent import AgentFailure, BilevelPlanningAgent
from prbench_bilevel_planning.env_models import create_bilevel_planning_models


@hydra.main(version_base=None, config_name="config", config_path="conf/")
def _main(cfg: DictConfig) -> None:

    logging.info(f"Running seed={cfg.seed}, env={cfg.env.env_name}")

    # Create the environment.
    prbench.register_all_environments()
    env = prbench.make(**cfg.env.make_kwargs)

    # Create the env models.
    env_models = create_bilevel_planning_models(
        cfg.env.env_name,
        env.observation_space,
        env.action_space,
        **cfg.env.env_model_kwargs,
    )

    # Create the agent.
    agent: BilevelPlanningAgent = BilevelPlanningAgent(
        env_models,
        cfg.seed,
        max_abstract_plans=cfg.max_abstract_plans,
        samples_per_step=cfg.samples_per_step,
        max_skill_horizon=cfg.max_skill_horizon,
        heuristic_name=cfg.heuristic_name,
        planning_timeout=cfg.planning_timeout,
    )

    # Evaluate.
    rng = np.random.default_rng(cfg.seed)
    metrics: list[dict[str, float]] = []
    for eval_episode in range(cfg.num_eval_episodes):
        logging.info(f"Starting evaluation episode {eval_episode}")
        episode_metrics = _run_single_episode_evaluation(
            agent,
            env,
            rng,
            max_eval_steps=cfg.max_eval_steps,
        )
        episode_metrics["eval_episode"] = eval_episode
        metrics.append(episode_metrics)

    # Aggregate and save results.
    df = pd.DataFrame(metrics)

    # Save results and config.
    current_dir = HydraConfig.get().runtime.output_dir

    # Save the metrics dataframe.
    results_path = os.path.join(current_dir, "results.csv")
    df.to_csv(results_path, index=False)
    logging.info(f"Saved results to {results_path}")

    # Save the full hydra config.
    config_path = os.path.join(current_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        OmegaConf.save(cfg, f)
    logging.info(f"Saved config to {config_path}")


def _run_single_episode_evaluation(
    agent: BilevelPlanningAgent,
    env: Env,
    rng: np.random.Generator,
    max_eval_steps: int,
) -> dict[str, float]:
    steps = 0
    success = False
    seed = sample_seed_from_rng(rng)
    obs, info = env.reset(seed=seed)
    planning_time = 0.0  # measure the time taken by the approach only
    planning_failed = False
    with timer() as result:
        try:
            agent.reset(obs, info)
        except AgentFailure:
            logging.info("Agent failed to find any plan.")
            planning_failed = True
    planning_time += result["time"]
    if planning_failed:
        return {"success": False, "steps": steps, "planning_time": planning_time}
    for _ in range(max_eval_steps):
        with timer() as result:
            action = agent.step()
        planning_time += result["time"]
        obs, rew, done, truncated, info = env.step(action)
        reward = float(rew)
        assert not truncated
        with timer() as result:
            agent.update(obs, reward, done, info)
        planning_time += result["time"]
        if done:
            success = True
            break
        steps += 1
    logging.info(f"Success result: {success}")
    return {"success": success, "steps": steps, "planning_time": planning_time}


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
