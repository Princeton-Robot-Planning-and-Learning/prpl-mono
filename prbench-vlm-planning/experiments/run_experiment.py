"""Main entry point for running VLM planning experiments.

Examples:
    python experiments/run_experiment.py env=Motion2D-p1 seed=0 vlm_model=gpt-5
    python experiments/run_experiment.py env=Motion2D-p1 seed=0 \\
        vlm_model=gpt-5-nano temperature=1

    python experiments/run_experiment.py -m env=Motion2D-p1 seed='range(0,10)'
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

from prbench_vlm_planning.agent import VLMPlanningAgent, VLMPlanningAgentFailure
from prbench_vlm_planning.env_controllers import get_controllers_for_environment


@hydra.main(version_base=None, config_name="config", config_path="conf/")
def _main(cfg: DictConfig) -> None:

    logging.info(
        f"Running seed={cfg.seed}, env={cfg.env.env_name}, vlm_model={cfg.vlm_model}"
    )

    # Create the environment.
    prbench.register_all_environments()
    env = prbench.make(cfg.env.make_kwargs.env_id)

    # Load environment-specific controllers if available.
    env_controllers = None
    if cfg.get("use_env_models", True):
        env_controllers = get_controllers_for_environment(cfg.env.env_name)
        if env_controllers:
            logging.info(
                f"Successfully loaded {cfg.env.env_name} controllers: "
                f"{env_controllers}"
            )
        else:
            raise ValueError(
                f"No specific controllers found for " f"{cfg.env.env_name}"
            )

    # Create the agent.
    agent: VLMPlanningAgent = VLMPlanningAgent(
        observation_space=env.observation_space,
        vlm_model_name=cfg.vlm_model,
        temperature=cfg.temperature,
        max_planning_horizon=cfg.max_planning_horizon,
        seed=cfg.seed,
        env_controllers=env_controllers,
        use_image=cfg.get("use_image", True),
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
    agent: VLMPlanningAgent,
    env: Env,
    rng: np.random.Generator,
    max_eval_steps: int,
) -> dict[str, float]:
    steps = 0
    success = False
    seed = sample_seed_from_rng(rng)
    obs, info = env.reset(seed=seed)
    assert (
        env.metadata["description"] is not None
    ), "Environment must have a description."
    info.update({"description": env.metadata["description"]})
    planning_time = 0.0  # measure the time taken by the approach only
    planning_failed = False

    # Initial planning
    with timer() as result:
        try:
            agent.reset(obs, info)
        except VLMPlanningAgentFailure as e:
            logging.info(f"Agent failed to find any plan: {e}")
            planning_failed = True
    planning_time += result["time"]

    if planning_failed:
        return {"success": False, "steps": steps, "planning_time": planning_time}

    # Execute the plan
    for _ in range(max_eval_steps):
        with timer() as result:
            try:
                action = agent.step()
            except VLMPlanningAgentFailure as e:
                logging.info(f"Agent failed during execution: {e}")
                break
        planning_time += result["time"]

        # Execute action in environment
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
