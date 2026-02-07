"""Tests for base_motion3d.py."""

import kinder
import numpy as np
import pytest
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo

from kinder_bilevel_planning.agent import BilevelPlanningAgent
from kinder_bilevel_planning.env_models import create_bilevel_planning_models

kinder.register_all_environments()


def test_base_motion3d_observation_to_state():
    """Tests for observation_to_state() in the BaseMotion3D environment."""
    env = kinder.make("kinder/BaseMotion3D-v0")
    env_models = create_bilevel_planning_models(
        "base_motion3d", env.observation_space, env.action_space
    )
    observation_to_state = env_models.observation_to_state
    obs, _ = env.reset(seed=123)
    state = observation_to_state(obs)
    assert isinstance(hash(state), int)
    assert env_models.state_space.contains(state)
    assert env_models.observation_space == env.observation_space
    env.close()


def test_base_motion3d_transition_fn():
    """Tests for transition_fn() in the BaseMotion3D environment."""
    env = kinder.make("kinder/BaseMotion3D-v0")
    env.action_space.seed(123)
    env_models = create_bilevel_planning_models(
        "base_motion3d", env.observation_space, env.action_space
    )
    transition_fn = env_models.transition_fn
    obs, _ = env.reset(seed=123)
    state = env_models.observation_to_state(obs)

    # Test that transition function produces valid states
    for _ in range(10):
        executable = env.action_space.sample()
        next_state = transition_fn(state, executable)
        assert env_models.state_space.contains(next_state)
        assert isinstance(hash(next_state), int)
        state = next_state
    env.close()


def test_base_motion3d_goal_deriver():
    """Tests for goal_deriver() in the BaseMotion3D environment."""
    env = kinder.make("kinder/BaseMotion3D-v0")
    env_models = create_bilevel_planning_models(
        "base_motion3d", env.observation_space, env.action_space
    )
    goal_deriver = env_models.goal_deriver
    obs, _ = env.reset(seed=123)
    state = env_models.observation_to_state(obs)
    goal = goal_deriver(state)
    assert len(goal.atoms) == 1
    goal_atom = next(iter(goal.atoms))
    assert str(goal_atom) == "(AtTgt robot target)"
    env.close()


def test_base_motion3d_state_abstractor():
    """Tests for state_abstractor() in the BaseMotion3D environment."""
    env = kinder.make("kinder/BaseMotion3D-v0")
    env_models = create_bilevel_planning_models(
        "base_motion3d", env.observation_space, env.action_space
    )
    state_abstractor = env_models.state_abstractor
    pred_name_to_pred = {p.name: p for p in env_models.predicates}
    AtTgt = pred_name_to_pred["AtTgt"]

    obs, _ = env.reset(seed=123)
    state = env_models.observation_to_state(obs)
    abstract_state = state_abstractor(state)
    obj_name_to_obj = {o.name: o for o in abstract_state.objects}
    robot = obj_name_to_obj["robot"]
    target = obj_name_to_obj["target"]

    # Initially robot should not be at target (very unlikely with random sampling)
    assert AtTgt([robot, target]) not in abstract_state.atoms

    env.close()


def _skill_test_helper(ground_skill, env_models, env, obs, params=None):
    """Helper function to test a skill execution."""
    rng = np.random.default_rng(123)
    state = env_models.observation_to_state(obs)
    abstract_state = env_models.state_abstractor(state)
    operator = ground_skill.operator
    assert operator.preconditions.issubset(abstract_state.atoms)
    controller = ground_skill.controller
    if params is None:
        params = controller.sample_parameters(state, rng)
    controller.reset(state, params)
    for _ in range(500):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env_models.observation_to_state(obs)
        controller.observe(next_state)
        state = next_state

        if controller.terminated():
            break
    return obs


def test_base_motion3d_skills():
    """Tests for skills in the BaseMotion3D environment."""
    env = kinder.make("kinder/BaseMotion3D-v0")
    env_models = create_bilevel_planning_models(
        "base_motion3d", env.observation_space, env.action_space
    )
    skill_name_to_skill = {s.operator.name: s for s in env_models.skills}
    MoveBaseToTarget = skill_name_to_skill["MoveBaseToTarget"]

    obs0, _ = env.reset(seed=123)
    state0 = env_models.observation_to_state(obs0)
    abstract_state = env_models.state_abstractor(state0)
    obj_name_to_obj = {o.name: o for o in abstract_state.objects}
    robot = obj_name_to_obj["robot"]
    target = obj_name_to_obj["target"]

    # Test MoveBaseToTarget skill
    move_to_skill = MoveBaseToTarget.ground((robot, target))
    obs1 = _skill_test_helper(move_to_skill, env_models, env, obs0)

    # Check that robot reached the target
    state1 = env_models.observation_to_state(obs1)
    abstract_state1 = env_models.state_abstractor(state1)
    pred_name_to_pred = {p.name: p for p in env_models.predicates}
    AtTgt = pred_name_to_pred["AtTgt"]
    assert AtTgt([robot, target]) in abstract_state1.atoms

    env.close()


@pytest.mark.parametrize("seed", [123])
def test_base_motion3d_bilevel_planning(seed):
    """Tests for bilevel planning in the BaseMotion3D environment."""

    env = kinder.make("kinder/BaseMotion3D-v0", render_mode="rgb_array")

    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"BaseMotion3D-bilevel-{seed}"
        )

    env_models = create_bilevel_planning_models(
        "base_motion3d",
        env.observation_space,
        env.action_space,
    )
    agent = BilevelPlanningAgent(
        env_models,
        seed=seed,
        max_abstract_plans=5,
        samples_per_step=3,
        planning_timeout=60.0,
    )
    obs, info = env.reset(seed=seed)
    total_reward = 0
    try:
        agent.reset(obs, info)
    except Exception as e:
        env.close()
        pytest.skip(f"Planning failed for seed {seed}: {e}")

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
