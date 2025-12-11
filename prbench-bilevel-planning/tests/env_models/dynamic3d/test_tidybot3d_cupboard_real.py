"""Tests for tidybot3d_cupboard.py."""

import prbench
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo

from prbench_bilevel_planning.agent import BilevelPlanningAgent
from prbench_bilevel_planning.env_models import create_bilevel_planning_models

prbench.register_all_environments()


def test_tidybot3d_cupboard_bilevel_planning():
    """Tests for bilevel planning in the TidyBot3D cupboard real environment."""

    num_objects = 4
    env = prbench.make(
        f"prbench/TidyBot3D-cupboard_real-o{num_objects}-v0", render_mode="rgb_array"
    )

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos", name_prefix="TidyBot3D-cupboard")

    seed = 123
    obs, info = env.reset(seed=seed)
    total_reward = 0
    state = env.observation_space.devectorize(obs)

    env_models = create_bilevel_planning_models(
        "tidybot3d_cupboard_real",
        env.observation_space,
        env.action_space,
        num_objects=num_objects,
        initial_state=state,
    )
    agent = BilevelPlanningAgent(
        env_models,
        seed=seed,
        max_abstract_plans=1,
        samples_per_step=1,
        planning_timeout=120.0,
        max_skill_horizon=400,
    )

    agent.reset(obs, info)
    for _ in range(4000):
        action = agent.step()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        agent.update(obs, reward, terminated or truncated, info)
        if (
            terminated
            or truncated
            or len(agent._current_plan) == 0  # pylint: disable=protected-access
        ):
            break

    else:
        assert False, "Did not terminate successfully"

    env.close()
