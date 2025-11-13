"""Main entry point for running VLM planning experiments.

Examples:
VLM planning:
    python experiments/run_experiment.py env=Motion2D-p0-v0 seed=0 vlm_model=gpt-5
LLM planning:
    python experiments/run_experiment.py env=Motion2D-p0-v0 seed=0 vlm_model=gpt-5 \
        rgb_observation=false
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

    logging.info(f"Running seed={cfg.seed}, env={cfg.env}, vlm_model={cfg.vlm_model}")
    # Create the environment.
    prbench.register_all_environments()
    if cfg.get("rgb_observation", False):
        env = prbench.make(f"prbench/{cfg.env}", render_mode="rgb_array")
    else:
        env = prbench.make(f"prbench/{cfg.env}")
    assert env.spec is not None, "Environment spec must not be None"
    assert hasattr(
        env.spec, "entry_point"
    ), "We use the entry point to identify env class"
    entry_point = env.spec.entry_point
    assert isinstance(entry_point, str), "Entry point must be a string"
    module_path = entry_point.split(":")[0]  # "prbench_envs.geom2d.motion2d"
    parts = module_path.split(".")
    env_class_name = parts[-2]  # "geom2d"
    env_name = parts[-1]  # "motion2d"

    # Load environment-specific controllers if available.
    env_controllers = get_controllers_for_environment(
        env_class_name, env_name, action_space=env.action_space
    )
    assert env_controllers is not None, "Environment controllers must be available"

    # Create the agent.
    agent: VLMPlanningAgent = VLMPlanningAgent(
        observation_space=env.observation_space,
        env_controllers=env_controllers,
        vlm_model_name=cfg.vlm_model,
        temperature=cfg.temperature,
        max_planning_horizon=cfg.max_planning_horizon,
        seed=cfg.seed,
        rgb_observation=cfg.get("rgb_observation", False),
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

    # Wrap observation with rendered image if using RGB observations
    if agent.rgb_observation:
        rendered_img: np.ndarray = env.render()  # type: ignore[assignment]
        obs = {"state": obs, "img": rendered_img}

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

        # Wrap observation with rendered image if using RGB observations
        if agent.rgb_observation:
            rendered_img = env.render()  # type: ignore[assignment]
            obs = {"state": obs, "img": rendered_img}

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
