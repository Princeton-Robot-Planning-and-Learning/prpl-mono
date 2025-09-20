"""Tests for obstruction_2d.py."""

import numpy as np
import prbench
import pytest
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from prbench_models.geom2d.envs.obstruction2d.parameterized_skills import (
    get_robot_transfer_position,
)

from prbench_bilevel_planning.agent import BilevelPlanningAgent
from prbench_bilevel_planning.env_models import create_bilevel_planning_models

prbench.register_all_environments()


def test_obstruction2d_observation_to_state():
    """Tests for observation_to_state() in the Obstruction2D environment."""
    env = prbench.make("prbench/Obstruction2D-o1-v0")
    env_models = create_bilevel_planning_models(
        "obstruction2d", env.observation_space, env.action_space, num_obstructions=1
    )
    observation_to_state = env_models.observation_to_state
    obs, _ = env.reset(seed=123)
    state = observation_to_state(obs)
    assert isinstance(hash(state), int)  # states are hashable for bilevel planning
    assert env_models.state_space.contains(state)
    assert env_models.observation_space == env.observation_space
    env.close()


def test_obstruction2d_transition_fn():
    """Tests for transition_fn() in the Obstruction2D environment."""
    env = prbench.make("prbench/Obstruction2D-o1-v0")
    env.action_space.seed(123)
    env_models = create_bilevel_planning_models(
        "obstruction2d", env.observation_space, env.action_space, num_obstructions=1
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


def test_obstruction2d_goal_deriver():
    """Tests for goal_deriver() in the Obstruction2D environment."""
    env = prbench.make("prbench/Obstruction2D-o1-v0")
    env_models = create_bilevel_planning_models(
        "obstruction2d", env.observation_space, env.action_space, num_obstructions=1
    )
    goal_deriver = env_models.goal_deriver
    obs, _ = env.reset(seed=123)
    state = env_models.observation_to_state(obs)
    goal = goal_deriver(state)
    assert len(goal.atoms) == 1
    goal_atom = next(iter(goal.atoms))
    assert str(goal_atom) == "(OnTarget target_block)"


def test_obstruction2d_state_abstractor():
    """Tests for state_abstractor() in the Obstruction2D environment."""
    env = prbench.make("prbench/Obstruction2D-o1-v0")
    env_models = create_bilevel_planning_models(
        "obstruction2d", env.observation_space, env.action_space, num_obstructions=1
    )
    state_abstractor = env_models.state_abstractor
    pred_name_to_pred = {p.name: p for p in env_models.predicates}
    Holding = pred_name_to_pred["Holding"]
    HandEmpty = pred_name_to_pred["HandEmpty"]
    OnTable = pred_name_to_pred["OnTable"]
    OnTarget = pred_name_to_pred["OnTarget"]
    env.reset(seed=123)
    obs, _, _, _, _ = env.step((0, 0, 0, 0.1, 0.0))  # extend the arm
    state = env_models.observation_to_state(obs)
    abstract_state = state_abstractor(state)
    obj_name_to_obj = {o.name: o for o in abstract_state.objects}
    robot = obj_name_to_obj["robot"]
    target_block = obj_name_to_obj["target_block"]
    obstruction0 = obj_name_to_obj["obstruction0"]
    assert len(abstract_state.atoms) == 3
    assert HandEmpty([robot]) in abstract_state.atoms
    assert OnTable([target_block]) in abstract_state.atoms
    assert (
        OnTable([obstruction0]) in abstract_state.atoms
        or OnTarget([obstruction0]) in abstract_state.atoms
    )
    # Create state where robot is holding the target block.
    target_block_x = state.get(target_block, "x")
    robot_arm_joint = state.get(robot, "arm_joint")
    robot_x, robot_y = get_robot_transfer_position(
        target_block, state, target_block_x, robot_arm_joint
    )
    state1 = state.copy()
    state1.set(robot, "x", robot_x)
    state1.set(robot, "y", robot_y)
    state1.set(robot, "vacuum", 1.0)
    abstract_state1 = state_abstractor(state1)
    assert Holding([robot, target_block]) in abstract_state1.atoms

    # Uncomment to debug.
    # import imageio.v2 as iio
    # env.unwrapped._geom2d_env._current_state = state1
    # img = env.render()
    # iio.imsave(f"debug/robot-holding-block.png", img)
    # import ipdb; ipdb.set_trace()

    # Create state where robot is holding the obstruction.
    obstruction_x = state.get(obstruction0, "x")
    robot_x, robot_y = get_robot_transfer_position(
        obstruction0, state, obstruction_x, robot_arm_joint
    )
    state2 = state.copy()
    state2.set(robot, "x", robot_x)
    state2.set(robot, "y", robot_y)
    state2.set(robot, "vacuum", 1.0)
    abstract_state2 = state_abstractor(state2)
    assert Holding([robot, obstruction0]) in abstract_state2.atoms
    # Create state where the target block is on the target surface.
    target_surface = obj_name_to_obj["target_surface"]
    state3 = state.copy()
    state3.set(target_block, "x", state3.get(target_surface, "x"))
    abstract_state3 = state_abstractor(state3)
    assert OnTarget([target_block]) in abstract_state3.atoms


def _skill_test_helper(ground_skill, env_models, env, obs, params=None):
    rng = np.random.default_rng(123)
    state = env_models.observation_to_state(obs)
    abstract_state = env_models.state_abstractor(state)
    operator = ground_skill.operator
    assert operator.preconditions.issubset(abstract_state.atoms)
    predicted_next_atoms = (
        abstract_state.atoms - operator.delete_effects
    ) | operator.add_effects
    controller = ground_skill.controller
    if params is None:
        params = controller.sample_parameters(state, rng)
    controller.reset(state, params)
    for _ in range(100):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env_models.observation_to_state(obs)
        controller.observe(next_state)
        assert env_models.transition_fn(state, action) == next_state
        state = next_state

        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"
    next_abstract_state = env_models.state_abstractor(state)
    assert next_abstract_state.atoms == predicted_next_atoms
    return obs


def test_obstruction2d_skills():
    """Tests for skills in the Obstruction2D environment."""
    env = prbench.make("prbench/Obstruction2D-o0-v0")
    env_models = create_bilevel_planning_models(
        "obstruction2d", env.observation_space, env.action_space, num_obstructions=0
    )
    skill_name_to_skill = {s.operator.name: s for s in env_models.skills}
    PickFromTable = skill_name_to_skill["PickFromTable"]
    PickFromTarget = skill_name_to_skill["PickFromTarget"]
    PlaceOnTable = skill_name_to_skill["PlaceOnTable"]
    PlaceOnTarget = skill_name_to_skill["PlaceOnTarget"]
    obs0, _ = env.reset(seed=123)
    state0 = env_models.observation_to_state(obs0)
    abstract_state = env_models.state_abstractor(state0)
    obj_name_to_obj = {o.name: o for o in abstract_state.objects}
    robot = obj_name_to_obj["robot"]
    target_block = obj_name_to_obj["target_block"]
    pick_target_block_from_table = PickFromTable.ground((robot, target_block))
    pick_target_block_from_target = PickFromTarget.ground((robot, target_block))
    place_target_block_on_table = PlaceOnTable.ground((robot, target_block))
    place_target_block_on_target = PlaceOnTarget.ground((robot, target_block))
    # Test picking the target block from the table.
    obs1 = _skill_test_helper(pick_target_block_from_table, env_models, env, obs0)
    state1 = env_models.observation_to_state(obs1)
    # Test placing the target block back on the table in the exact same position.
    target_x = state0.get(target_block, "x")
    target_y = state0.get(target_block, "y")
    obs2 = _skill_test_helper(
        place_target_block_on_table,
        env_models,
        env,
        obs1,
        params=state1.get(robot, "x"),
    )
    state2 = env_models.observation_to_state(obs2)
    actual_x = state2.get(target_block, "x")
    actual_y = state2.get(target_block, "y")
    assert np.isclose(target_x, actual_x)
    assert np.isclose(target_y, actual_y)
    # Pick the block again.
    obs3 = _skill_test_helper(pick_target_block_from_table, env_models, env, obs2)
    # Place the block in the target.
    obs4 = _skill_test_helper(place_target_block_on_target, env_models, env, obs3)
    # Pick the block from the target.
    _skill_test_helper(pick_target_block_from_target, env_models, env, obs4)


@pytest.mark.parametrize(
    "num_obstructions, max_abstract_plans, samples_per_step",
    [
        (0, 1, 1),
        (1, 10, 1),
    ],
)
def test_obstruction2d_bilevel_planning(
    num_obstructions, max_abstract_plans, samples_per_step
):
    """Tests for bilevel planning in the Obstruction2D environment.

    Note that we only test a small number of obstructions to keep tests fast. Use
    experiment scripts to evaluate at scale.
    """

    env = prbench.make(
        f"prbench/Obstruction2D-o{num_obstructions}-v0", render_mode="rgb_array"
    )

    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"Obstruction2D-o{num_obstructions}"
        )

    env_models = create_bilevel_planning_models(
        "obstruction2d",
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
