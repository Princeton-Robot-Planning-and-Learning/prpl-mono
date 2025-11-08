"""Tests for tidybot3d_base_motion.py."""

import prbench
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo

from prbench_bilevel_planning.agent import BilevelPlanningAgent
from prbench_bilevel_planning.env_models import create_bilevel_planning_models

prbench.register_all_environments()


def test_tidybot3d_base_motion_bilevel_planning():
    """Tests for bilevel planning in the TidyBot3D base motion environment."""

    env = prbench.make("prbench/TidyBot3D-base_motion-o1-v0", render_mode="rgb_array")

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos", name_prefix="TidyBot3D-base_motion")

    env_models = create_bilevel_planning_models(
        "tidybot3d_base_motion",
        env.observation_space,
        env.action_space,
    )
    seed = 123
    agent = BilevelPlanningAgent(
        env_models,
        seed=seed,
        max_abstract_plans=1,
        samples_per_step=1,
        planning_timeout=30.0,
    )
    obs, info = env.reset(seed=seed)
    total_reward = 0
    agent.reset(obs, info)
    for _ in range(100):
        action = agent.step()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        agent.update(obs, reward, terminated or truncated, info)
        if terminated or truncated:
            break

    else:
        assert False, "Did not terminate successfully"

    env.close()
