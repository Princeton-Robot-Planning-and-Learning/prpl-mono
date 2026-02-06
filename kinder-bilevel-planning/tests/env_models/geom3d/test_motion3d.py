"""Tests for motion3d.py."""

import kinder
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo

from kinder_bilevel_planning.agent import BilevelPlanningAgent
from kinder_bilevel_planning.env_models import create_bilevel_planning_models

kinder.register_all_environments()


def test_motion3d_bilevel_planning():
    """Tests for bilevel planning in the Motion3D environment."""

    env = kinder.make("kinder/Motion3D-v0", render_mode="rgb_array")

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos", name_prefix="Motion3D-bilevel")

    env_models = create_bilevel_planning_models(
        "motion3d",
        env.observation_space,
        env.action_space,
    )
    agent = BilevelPlanningAgent(
        env_models,
        seed=123,
        max_abstract_plans=1,
        samples_per_step=1,
        planning_timeout=60.0,
        max_skill_horizon=1000,
    )
    obs, info = env.reset(seed=123)
    total_reward = 0
    agent.reset(obs, info)

    for _ in range(1000):
        action = agent.step()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        agent.update(obs, reward, terminated or truncated, info)
        if terminated or truncated:
            break

    else:
        assert False, "Did not terminate successfully"

    env.close()
