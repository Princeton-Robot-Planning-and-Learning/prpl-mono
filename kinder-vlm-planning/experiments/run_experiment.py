"""Main entry point for running VLM planning experiments.

Examples:
- Running on a single environment with a single seed:
    python experiments/run_experiment.py env=Motion2D-p0-v0 seed=0 vlm_model=gpt-5 \
        temperature=1
    python experiments/run_experiment.py -m env=StickButton2D-b1-v0 seed=0 \
        vlm_model=gpt-5 temperature=1

- Running on multiple environments and multiple seeds:
    python experiments/run_experiment.py -m seed='range(0,3)' \
        env=Motion2D-p0-v0,StickButton2D-b1-v0 \
        vlm_model=gpt-5 rgb_observation=true,false temperature=1
    python experiments/run_experiment.py -m seed='range(0,3)' \
        env=BaseMotion3D-v0,Transport3D-o2-v0,Shelf3D-o1-v0 \
        vlm_model=gpt-5 rgb_observation=true,false temperature=1 hydra/launcher=joblib \
        hydra.launcher.n_jobs=-1
"""

import logging
import os

import hydra
import kinder
import numpy as np
import pandas as pd
from gymnasium.core import Env
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from prpl_utils.utils import sample_seed_from_rng, timer

from kinder_vlm_planning.agent import VLMPlanningAgent, VLMPlanningAgentFailure
from kinder_vlm_planning.env_controllers import get_controllers_for_environment


@hydra.main(version_base=None, config_name="config", config_path="conf/")
def _main(cfg: DictConfig) -> None:

    logging.info(f"Running seed={cfg.seed}, env={cfg.env}, vlm_model={cfg.vlm_model}")
    # Create the environment.
    kinder.register_all_environments()
    if cfg.get("rgb_observation", False):
        env = kinder.make(f"kinder/{cfg.env}", render_mode="rgb_array")
    else:
        env = kinder.make(f"kinder/{cfg.env}")
    assert env.spec is not None, "Environment spec must not be None"
    assert hasattr(
        env.spec, "entry_point"
    ), "We use the entry point to identify env class"
    entry_point = env.spec.entry_point
    assert isinstance(entry_point, str), "Entry point must be a string"
    module_path = entry_point.split(":")[0]  # "kinder_envs.kinematic2d.motion2d"
    parts = module_path.split(".")
    env_class_name = parts[-2]  # "kinematic2d"
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
        max_planning_horizon=cfg.max_eval_steps,
        seed=cfg.seed,
        rgb_observation=cfg.get("rgb_observation", False),
    )

    # Evaluate.
    rng = np.random.default_rng(cfg.seed)
    metrics: list[dict[str, float | bool | str]] = []
    for eval_episode in range(cfg.num_eval_episodes):
        logging.info(f"Starting evaluation episode {eval_episode}")
        try:
            episode_metrics = _run_single_episode_evaluation(
                agent,
                env,
                rng,
                max_eval_steps=cfg.max_eval_steps,
            )
            episode_metrics["eval_episode"] = eval_episode
            metrics.append(episode_metrics)
        except Exception as e:
            logging.error(
                f"Episode {eval_episode} failed with error: {e}", exc_info=True
            )
            # Record failure and continue to next episode
            episode_metrics = {
                "success": False,
                "steps": 0,
                "planning_time": 0.0,
                "execution_time": 0.0,
                "reward": 0.0,
                "eval_episode": eval_episode,
                "error": str(e),
            }
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
) -> dict[str, float | bool | str]:
    steps = 0
    success = False
    total_reward = 0.0
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
    planning_time = 0.0  # time spent generating plans (VLM queries)
    execution_time = 0.0  # time spent executing the policy (getting actions)
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
        return {
            "success": False,
            "steps": steps,
            "planning_time": planning_time,
            "execution_time": execution_time,
            "reward": total_reward,
        }

    # Execute the plan
    for _ in range(max_eval_steps):
        with timer() as result:
            try:
                action = agent.step()
            except VLMPlanningAgentFailure as e:
                logging.info(f"Agent failed during execution: {e}")
                break
        execution_time += result["time"]

        # Execute action in environment
        obs, rew, done, truncated, info = env.step(action)
        reward = float(rew)
        total_reward += reward
        assert not truncated

        # Log the resulting state and done status
        logging.info(f"[ENV STEP DEBUG] Done: {done}")
        # Devectorize to get readable state
        state_obs_for_log = obs if not agent.rgb_observation else obs
        try:
            obs_space = env.observation_space
            state = obs_space.devectorize(state_obs_for_log)  # type: ignore
            logging.info(f"[ENV STEP DEBUG] Resulting state:\n{state.pretty_str()}")
        except Exception as e:
            logging.info(f"[ENV STEP DEBUG] Could not devectorize state: {e}")

        # Wrap observation with rendered image if using RGB observations
        if agent.rgb_observation:
            rendered_img = env.render()  # type: ignore[assignment]
            obs = {"state": obs, "img": rendered_img}

        if done:
            success = True
            break
        steps += 1

        with timer() as result:
            try:
                agent.update(obs, reward, done, info)
            except VLMPlanningAgentFailure as e:
                logging.info(f"Agent failed during update: {e}")
                break
        execution_time += result["time"]

    logging.info(f"Success result: {success}")
    return {
        "success": success,
        "steps": steps,
        "planning_time": planning_time,
        "execution_time": execution_time,
        "reward": total_reward,
    }


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
