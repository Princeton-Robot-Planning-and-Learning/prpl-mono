"""Tests for clutteredretrieval2d.py."""

import time

import imageio.v2 as iio
import numpy as np
import prbench
import pytest
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from prbench.envs.geom2d.structs import SE2Pose

from prbench_bilevel_planning.agent import BilevelPlanningAgent
from prbench_bilevel_planning.env_models import create_bilevel_planning_models

prbench.register_all_environments()


def test_clutteredretrieval2d_observation_to_state():
    """Tests for observation_to_state() in the ClutteredRetrieval2D environment."""
    env = prbench.make("prbench/ClutteredRetrieval2D-o1-v0")
    env_models = create_bilevel_planning_models(
        "clutteredretrieval2d",
        env.observation_space,
        env.action_space,
        num_obstructions=1,
    )
    observation_to_state = env_models.observation_to_state
    obs, _ = env.reset(seed=123)
    state = observation_to_state(obs)
    assert isinstance(hash(state), int)  # states are hashable for bilevel planning
    assert env_models.state_space.contains(state)
    assert env_models.observation_space == env.observation_space
    env.close()


def test_clutteredretrieval2d_transition_fn():
    """Tests for transition_fn() in the ClutteredRetrieval2D environment."""
    env = prbench.make("prbench/ClutteredRetrieval2D-o1-v0")
    env.action_space.seed(123)
    env_models = create_bilevel_planning_models(
        "clutteredretrieval2d",
        env.observation_space,
        env.action_space,
        num_obstructions=1,
    )
    transition_fn = env_models.transition_fn
    obs, _ = env.reset(seed=123)
    state = env_models.observation_to_state(obs)
    for _ in range(100):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        next_state = env_models.observation_to_state(obs)
        predicted_next_state = transition_fn(state, action)
        assert next_state == predicted_next_state
        state = next_state
    env.close()


def test_clutteredretrieval2d_goal_deriver():
    """Tests for goal_deriver() in the ClutteredRetrieval2D environment."""
    env = prbench.make("prbench/ClutteredRetrieval2D-o1-v0")
    env_models = create_bilevel_planning_models(
        "clutteredretrieval2d",
        env.observation_space,
        env.action_space,
        num_obstructions=1,
    )
    goal_deriver = env_models.goal_deriver
    obs, _ = env.reset(seed=123)
    state = env_models.observation_to_state(obs)
    goal = goal_deriver(state)
    assert len(goal.atoms) == 1
    goal_atom = next(iter(goal.atoms))
    assert str(goal_atom) == "(Inside target_block target_region)"


def test_clutteredretrieval2d_state_abstractor():
    """Tests for state_abstractor() in the ClutteredRetrieval2D environment."""
    env = prbench.make("prbench/ClutteredRetrieval2D-o1-v0")
    env_models = create_bilevel_planning_models(
        "clutteredretrieval2d",
        env.observation_space,
        env.action_space,
        num_obstructions=1,
    )
    state_abstractor = env_models.state_abstractor
    pred_name_to_pred = {p.name: p for p in env_models.predicates}
    HoldingTgt = pred_name_to_pred["HoldingTgt"]
    HandEmpty = pred_name_to_pred["HandEmpty"]
    InSide = pred_name_to_pred["Inside"]
    env.reset(seed=123)
    obs, _, _, _, _ = env.step((0, 0, 0, 0.1, 0.0))  # extend the arm
    state = env_models.observation_to_state(obs)
    abstract_state = state_abstractor(state)
    obj_name_to_obj = {o.name: o for o in abstract_state.objects}
    robot = obj_name_to_obj["robot"]
    target_block = obj_name_to_obj["target_block"]
    target_region = obj_name_to_obj["target_region"]
    assert len(abstract_state.atoms) == 1
    assert HandEmpty([robot]) in abstract_state.atoms

    # Create state where robot is holding the target block.
    state1 = state.copy()
    state1.set(robot, "vacuum", 1.0)
    # Move robot close to target block
    target_x = state.get(target_block, "x")
    target_y = state.get(target_block, "y")
    state1.set(robot, "x", target_x)
    state1.set(robot, "y", target_y + 0.2)  # position above target
    abstract_state1 = state_abstractor(state1)
    assert HoldingTgt([robot, target_block]) in abstract_state1.atoms

    # Create state where the target block is inside the target region
    state2 = state.copy()
    target_x = state.get(target_region, "x")
    target_y = state.get(target_region, "y")
    target_theta = state.get(target_region, "theta")
    target_width = state.get(target_region, "width")
    target_height = state.get(target_region, "height")
    target_block_width = state.get(target_block, "width")
    target_block_height = state.get(target_block, "height")
    target_center_pose = SE2Pose(target_x, target_y, target_theta) * SE2Pose(
        target_width / 2,
        target_height / 2,
        0.0,
    )
    target_block_bottom_pose = target_center_pose * SE2Pose(
        -target_block_width / 2,
        -target_block_height / 2,
        0.0,
    )

    state2.set(target_block, "x", target_block_bottom_pose.x)
    state2.set(target_block, "y", target_block_bottom_pose.y)
    state2.set(target_block, "theta", target_block_bottom_pose.theta)
    abstract_state2 = state_abstractor(state2)
    assert InSide([target_block, target_region]) in abstract_state2.atoms


def _skill_test_helper(ground_skill, env_models, env, obs, params=None, debug=False):
    rng = np.random.default_rng(123)
    state = env_models.observation_to_state(obs)
    abstract_state = env_models.state_abstractor(state)
    operator = ground_skill.operator
    assert operator.preconditions.issubset(abstract_state.atoms)
    controller = ground_skill.controller
    if params is None:
        params = controller.sample_parameters(state, rng)
    controller.reset(state, params)
    for _ in range(200):  # More steps for motion planning
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env_models.observation_to_state(obs)
        controller.observe(next_state)
        assert env_models.transition_fn(state, action) == next_state
        state = next_state
        if debug:
            img = env.render()
            iio.imsave(f"debug/debug-test-{int(time.time()*1000.0)}.png", img)

        if controller.terminated():
            break
    return obs


def test_clutteredretrieval2d_skills():
    """Tests for skills in the ClutteredRetrieval2D environment."""
    env = prbench.make("prbench/ClutteredRetrieval2D-o1-v0")
    env_models = create_bilevel_planning_models(
        "clutteredretrieval2d",
        env.observation_space,
        env.action_space,
        num_obstructions=1,
    )
    predicate_name_to_pred = {p.name: p for p in env_models.predicates}
    skill_name_to_skill = {s.operator.name: s for s in env_models.skills}
    PickTgt = skill_name_to_skill["PickTgt"]
    PickObstruction = skill_name_to_skill["PickObstruction"]
    PlaceObstruction = skill_name_to_skill["PlaceObstruction"]
    PlaceTgt = skill_name_to_skill["PlaceTgt"]
    obs0, _ = env.reset(seed=123)
    state0 = env_models.observation_to_state(obs0)
    abstract_state = env_models.state_abstractor(state0)
    obj_name_to_obj = {o.name: o for o in abstract_state.objects}
    robot = obj_name_to_obj["robot"]
    target_block = obj_name_to_obj["target_block"]
    obstruction = obj_name_to_obj["obstruction0"]
    pick_target_block = PickTgt.ground((robot, target_block))
    # Test picking the target block from left side.
    obs1 = _skill_test_helper(
        pick_target_block, env_models, env, obs0, params=(0.5, 0.1, 0.15)
    )
    state1 = env_models.observation_to_state(obs1)
    abstract_state1 = env_models.state_abstractor(state1)
    assert (
        predicate_name_to_pred["HoldingTgt"]([robot, target_block])
        in abstract_state1.atoms
    )
    # Test picking the obstruction from bottom side.
    obs0, _ = env.reset(seed=123)
    pick_obstruction = PickObstruction.ground((robot, obstruction))
    obs1 = _skill_test_helper(
        pick_obstruction, env_models, env, obs0, params=(0.5, 0.9, 0.15)
    )
    state1 = env_models.observation_to_state(obs1)
    abstract_state1 = env_models.state_abstractor(state1)
    assert (
        predicate_name_to_pred["HoldingObstruction"]([robot, obstruction])
        in abstract_state1.atoms
    )

    # Placing the obstruction to empty place.
    place_obstruction = PlaceObstruction.ground((robot, obstruction))
    obs1 = _skill_test_helper(place_obstruction, env_models, env, obs1)
    state1 = env_models.observation_to_state(obs1)
    abstract_state1 = env_models.state_abstractor(state1)
    assert predicate_name_to_pred["HandEmpty"]([robot]) in abstract_state1.atoms
    assert (
        predicate_name_to_pred["HoldingObstruction"]([robot, obstruction])
        not in abstract_state1.atoms
    )

    # Picking the target block from right side, which should be possible now.
    pick_target_block = PickTgt.ground((robot, target_block))
    obs2 = _skill_test_helper(
        pick_target_block, env_models, env, obs1, params=(0.9, 0.4, 0.15)
    )
    state2 = env_models.observation_to_state(obs2)
    abstract_state2 = env_models.state_abstractor(state2)
    assert (
        predicate_name_to_pred["HoldingTgt"]([robot, target_block])
        in abstract_state2.atoms
    )

    # Place the target block inside the target region.
    place_target_block = PlaceTgt.ground(
        (robot, target_block, obj_name_to_obj["target_region"])
    )
    obs3 = _skill_test_helper(place_target_block, env_models, env, obs2)
    state3 = env_models.observation_to_state(obs3)
    abstract_state3 = env_models.state_abstractor(state3)
    assert predicate_name_to_pred["HandEmpty"]([robot]) in abstract_state3.atoms
    assert (
        predicate_name_to_pred["Inside"](
            [target_block, obj_name_to_obj["target_region"]]
        )
        in abstract_state3.atoms
    )


@pytest.mark.parametrize(
    "num_obstructions, max_abstract_plans, samples_per_step",
    [
        (1, 2, 10),
    ],
)
def test_clutteredretrieval2d_bilevel_planning(
    num_obstructions, max_abstract_plans, samples_per_step
):
    """Tests for bilevel planning in the ClutteredRetrieval2D environment.

    Note that we only test a small number of obstructions to keep tests fast. Use
    experiment scripts to evaluate at scale.
    """

    env = prbench.make(
        f"prbench/ClutteredRetrieval2D-o{num_obstructions}-v0", render_mode="rgb_array"
    )

    if MAKE_VIDEOS:
        env = RecordVideo(
            env,
            "unit_test_videos",
            name_prefix=f"ClutteredRetrieval2D-o{num_obstructions}",
        )

    env_models = create_bilevel_planning_models(
        "clutteredretrieval2d",
        env.observation_space,
        env.action_space,
        num_obstructions=num_obstructions,
    )
    agent = BilevelPlanningAgent(
        env_models,
        seed=123,
        max_abstract_plans=max_abstract_plans,
        samples_per_step=samples_per_step,
    )

    obs, info = env.reset(seed=0)

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
