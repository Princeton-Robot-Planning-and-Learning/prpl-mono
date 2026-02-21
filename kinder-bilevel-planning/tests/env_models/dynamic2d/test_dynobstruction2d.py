"""Tests for dynobstruction2d.py."""

import time

import imageio.v2 as iio
import kinder
import numpy as np
from kinder.envs.kinematic2d.structs import SE2Pose

from kinder_bilevel_planning.env_models import create_bilevel_planning_models

kinder.register_all_environments()


def test_dynobstruction2d_observation_to_state():
    """Tests for observation_to_state() in the DynObstruction2D environment."""
    env = kinder.make("kinder/DynObstruction2D-o1-v0")
    env_models = create_bilevel_planning_models(
        "dynobstruction2d",
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


def test_dynobstruction2d_transition_fn():
    """Tests for transition_fn() in the DynObstruction2D environment."""
    env = kinder.make("kinder/DynObstruction2D-o1-v0")
    env.action_space.seed(123)
    env_models = create_bilevel_planning_models(
        "dynobstruction2d",
        env.observation_space,
        env.action_space,
        num_obstructions=1,
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


def test_dynobstruction2d_goal_deriver():
    """Tests for goal_deriver() in the DynObstruction2D environment."""
    env = kinder.make("kinder/DynObstruction2D-o1-v0")
    env_models = create_bilevel_planning_models(
        "dynobstruction2d",
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
    assert str(goal_atom) == "(OnTgt target_block target_surface)"


def test_dynobstruction2d_state_abstractor():
    """Tests for state_abstractor() in the DynObstruction2D environment."""
    env = kinder.make("kinder/DynObstruction2D-o1-v0", render_mode="rgb_array")
    env_models = create_bilevel_planning_models(
        "dynobstruction2d",
        env.observation_space,
        env.action_space,
        num_obstructions=1,
    )

    state_abstractor = env_models.state_abstractor
    pred_name_to_pred = {p.name: p for p in env_models.predicates}
    HandEmpty = pred_name_to_pred["HandEmpty"]
    OnTgtSurface = pred_name_to_pred["OnTgt"]
    AboveTgtSurface = pred_name_to_pred["AboveTgt"]
    env.reset(seed=123)
    obs, _, _, _, _ = env.step((0, 0, 0, 0.1, 0.0))  # extend the arm

    state = env_models.observation_to_state(obs)
    abstract_state = state_abstractor(state)
    obj_name_to_obj = {o.name: o for o in abstract_state.objects}
    robot = obj_name_to_obj["robot"]
    target_block = obj_name_to_obj["target_block"]
    obstruction = obj_name_to_obj["obstruction0"]

    target_surface = obj_name_to_obj["target_surface"]
    assert len(abstract_state.atoms) == 1
    assert HandEmpty([robot]) in abstract_state.atoms

    # Create state where the target block is inside the target region
    state2 = state.copy()
    target_x = state.get(target_surface, "x")
    target_y = state.get(target_surface, "y")
    target_theta = state.get(target_surface, "theta")
    target_height = state.get(target_surface, "height")
    target_block_y = state.get(target_block, "y")
    target_block_height = state.get(target_block, "height")
    target_center_pose = SE2Pose(target_x, target_y, target_theta) * SE2Pose(
        0,
        target_height / 2 + target_block_height / 2,
        0.0,
    )

    # Move robot above the target location
    arm_length = state.get(robot, "arm_length")
    gripper_height = state.get(robot, "gripper_base_height")

    target_se2_pose = SE2Pose(target_x, target_block_y, target_theta) * SE2Pose(
        0, arm_length + gripper_height, -np.pi / 2
    )

    state2.set(robot, "x", target_se2_pose.x)
    state2.set(robot, "y", target_se2_pose.y)  # position above target
    state2.set(robot, "theta", target_se2_pose.theta)  # position above target

    # Move obstruction away from target location
    state2.set(obstruction, "x", 3.5 * target_se2_pose.x)

    # Move target on target location
    state2.set(target_block, "x", target_center_pose.x)
    state2.set(target_block, "y", target_center_pose.y)
    state2.set(target_block, "theta", target_center_pose.theta)
    abstract_state2 = state_abstractor(state2)

    assert OnTgtSurface([target_block, target_surface]) in abstract_state2.atoms
    assert AboveTgtSurface([robot]) in abstract_state2.atoms


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
        state = next_state
        if debug:
            img = env.render()
            iio.imsave(f"debug/debug-test-{int(time.time()*1000.0)}.png", img)

        if controller.terminated():
            break
    return obs


def test_dynobstruction2d_skills():
    """Tests for skills in the DynObstruction2D environment."""
    env = kinder.make("kinder/DynObstruction2D-o1-v0")
    env_models = create_bilevel_planning_models(
        "dynobstruction2d",
        env.observation_space,
        env.action_space,
        num_obstructions=1,
    )
    predicate_name_to_pred = {p.name: p for p in env_models.predicates}
    skill_name_to_skill = {s.operator.name: s for s in env_models.skills}
    PickTgt = skill_name_to_skill["PickTgt"]
    PlaceTgtSurface = skill_name_to_skill["PlaceTgtSurface"]
    obs0, _ = env.reset(seed=123)

    state0 = env_models.observation_to_state(obs0)
    abstract_state = env_models.state_abstractor(state0)
    obj_name_to_obj = {o.name: o for o in abstract_state.objects}
    robot = obj_name_to_obj["robot"]
    target_block = obj_name_to_obj["target_block"]
    target_surface = obj_name_to_obj["target_surface"]
    pick_target_block = PickTgt.ground((robot, target_block))
    # Test picking the target block from the top side.
    obs1 = _skill_test_helper(
        pick_target_block, env_models, env, obs0, params=(0, 0.6, 0.3)
    )
    state1 = env_models.observation_to_state(obs1)
    abstract_state1 = env_models.state_abstractor(state1)
    assert (
        predicate_name_to_pred["HoldingTgt"]([robot, target_block])
        in abstract_state1.atoms
    )

    # Test moving with the target block to be above target surface.
    # obs0, _ = env.reset(seed=123)
    place_target = PlaceTgtSurface.ground((robot, target_block, target_surface))
    obs2 = _skill_test_helper(place_target, env_models, env, obs1, params=0.25)
    state2 = env_models.observation_to_state(obs2)
    abstract_state2 = env_models.state_abstractor(state2)
    assert (
        predicate_name_to_pred["OnTgt"]([target_block, target_surface])
        in abstract_state2.atoms
    )
