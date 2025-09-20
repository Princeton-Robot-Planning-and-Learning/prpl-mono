"""Tests for stickbutton2d.py."""

import time

import imageio.v2 as iio
import numpy as np
import prbench
import pytest
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from prbench.envs.geom2d.stickbutton2d import StickButton2DEnvConfig

from prbench_bilevel_planning.agent import BilevelPlanningAgent
from prbench_bilevel_planning.env_models import create_bilevel_planning_models

prbench.register_all_environments()


def test_stickbutton2d_observation_to_state():
    """Tests for observation_to_state() in the StickButton2D environment."""
    env = prbench.make("prbench/StickButton2D-b1-v0")
    env_models = create_bilevel_planning_models(
        "stickbutton2d", env.observation_space, env.action_space, num_buttons=1
    )
    observation_to_state = env_models.observation_to_state
    obs, _ = env.reset(seed=123)
    state = observation_to_state(obs)
    assert isinstance(hash(state), int)
    assert env_models.state_space.contains(state)
    assert env_models.observation_space == env.observation_space
    env.close()


def test_stickbutton2d_transition_fn():
    """Tests for transition_fn() in the StickButton2D environment."""
    env = prbench.make("prbench/StickButton2D-b1-v0")
    env.action_space.seed(123)
    env_models = create_bilevel_planning_models(
        "stickbutton2d", env.observation_space, env.action_space, num_buttons=1
    )
    transition_fn = env_models.transition_fn
    obs, _ = env.reset(seed=123)
    state = env_models.observation_to_state(obs)
    for _ in range(100):
        executable = env.action_space.sample()
        obs, _, _, _, _ = env.step(executable)
        next_state = env_models.observation_to_state(obs)
        action = tuple(executable)
        predicted_next_state = transition_fn(state, action)
        assert next_state == predicted_next_state
        state = next_state
    env.close()


def test_stickbutton2d_goal_deriver():
    """Tests for goal_deriver() in the StickButton2D environment."""
    env = prbench.make("prbench/StickButton2D-b1-v0")
    env_models = create_bilevel_planning_models(
        "stickbutton2d", env.observation_space, env.action_space, num_buttons=1
    )
    goal_deriver = env_models.goal_deriver
    obs, _ = env.reset(seed=123)
    state = env_models.observation_to_state(obs)
    goal = goal_deriver(state)
    assert len(goal.atoms) == 1
    goal_atom = next(iter(goal.atoms))
    assert str(goal_atom) == "(Pressed button0)"


def test_stickbutton2d_state_abstractor():
    """Tests for state_abstractor() in the StickButton2D environment."""
    env = prbench.make("prbench/StickButton2D-b1-v0")
    env_models = create_bilevel_planning_models(
        "stickbutton2d", env.observation_space, env.action_space, num_buttons=1
    )
    state_abstractor = env_models.state_abstractor
    pred_name_to_pred = {p.name: p for p in env_models.predicates}
    Grasped = pred_name_to_pred["Grasped"]
    HandEmpty = pred_name_to_pred["HandEmpty"]
    Pressed = pred_name_to_pred["Pressed"]
    RobotAboveButton = pred_name_to_pred["RobotAboveButton"]
    StickAboveButton = pred_name_to_pred["StickAboveButton"]

    obs, _ = env.reset(seed=123)
    state = env_models.observation_to_state(obs)
    abstract_state = state_abstractor(state)
    obj_name_to_obj = {o.name: o for o in abstract_state.objects}
    robot = obj_name_to_obj["robot"]
    stick = obj_name_to_obj["stick"]
    button0 = obj_name_to_obj["button0"]

    # Initially robot should have empty hand and be above no button
    assert HandEmpty([robot]) in abstract_state.atoms
    # Check for AboveNoButton atom (with no parameters)
    above_no_button_atoms = [
        atom for atom in abstract_state.atoms if atom.predicate.name == "AboveNoButton"
    ]
    assert len(above_no_button_atoms) == 1
    # No button should be pressed initially
    assert not any(atom.predicate.name == "Pressed" for atom in abstract_state.atoms)

    # Create state where robot is holding the stick
    state1 = state.copy()
    # Position robot properly to grasp the stick
    stick_x = state1.get(stick, "x")
    stick_y = state1.get(stick, "y")
    robot_arm_joint = state1.get(robot, "arm_joint")
    # Calculate proper robot position for grasping (robot needs to be positioned
    # so its gripper tool tip reaches the stick)
    robot_x = stick_x
    robot_y = stick_y + robot_arm_joint + state1.get(robot, "gripper_width") / 2 + 0.01
    state1.set(robot, "x", robot_x)
    state1.set(robot, "y", robot_y)
    state1.set(robot, "vacuum", 1.0)

    abstract_state1 = state_abstractor(state1)
    assert Grasped([robot, stick]) in abstract_state1.atoms

    # Create state where button is pressed (green color)
    state2 = state.copy()
    state2.set(button0, "color_r", 0.0)
    state2.set(button0, "color_g", 0.9)
    state2.set(button0, "color_b", 0.0)
    abstract_state2 = state_abstractor(state2)
    assert Pressed([button0]) in abstract_state2.atoms

    # Test RobotAboveButton predicate
    state3 = state.copy()
    # Position robot directly over button0
    button_x = state3.get(button0, "x")
    button_y = state3.get(button0, "y")
    state3.set(robot, "x", button_x)
    state3.set(robot, "y", button_y)

    abstract_state3 = state_abstractor(state3)
    assert RobotAboveButton([robot, button0]) in abstract_state3.atoms
    # AboveNoButton should not be present when robot is above a button
    above_no_button_atoms3 = [
        atom for atom in abstract_state3.atoms if atom.predicate.name == "AboveNoButton"
    ]
    assert len(above_no_button_atoms3) == 0

    # Test StickAboveButton predicate
    state4 = state.copy()
    # Position stick directly over button0
    state4.set(stick, "x", button_x)
    state4.set(stick, "y", button_y)

    abstract_state4 = state_abstractor(state4)
    assert StickAboveButton([stick, button0]) in abstract_state4.atoms
    # AboveNoButton should not be present when stick is above a button
    above_no_button_atoms4 = [
        atom for atom in abstract_state4.atoms if atom.predicate.name == "AboveNoButton"
    ]
    assert len(above_no_button_atoms4) == 0

    # Test that AboveNoButton appears when neither robot nor stick is above any button
    state5 = state.copy()
    # Move both robot and stick away from button
    state5.set(robot, "x", 0.1)  # Far corner
    state5.set(robot, "y", 0.1)
    state5.set(stick, "x", 3.4)  # Opposite corner
    state5.set(stick, "y", 2.4)

    abstract_state5 = state_abstractor(state5)
    above_no_button_atoms5 = [
        atom for atom in abstract_state5.atoms if atom.predicate.name == "AboveNoButton"
    ]
    assert len(above_no_button_atoms5) == 1
    # No RobotAboveButton or StickAboveButton should be present
    assert not any(
        atom.predicate.name == "RobotAboveButton" for atom in abstract_state5.atoms
    )
    assert not any(
        atom.predicate.name == "StickAboveButton" for atom in abstract_state5.atoms
    )

    # Test that robot grasping stick and being above button works correctly
    state6 = state.copy()
    # First position the stick above button
    state6.set(stick, "x", button_x)
    state6.set(stick, "y", button_y)
    # Then position robot to grasp the stick at its new location
    robot_arm_joint = state6.get(robot, "arm_joint")
    robot_x = button_x  # Same x as stick
    robot_y = button_y + robot_arm_joint + state6.get(robot, "gripper_width") / 2 + 0.01
    state6.set(robot, "x", robot_x)
    state6.set(robot, "y", robot_y)
    state6.set(robot, "vacuum", 1.0)

    abstract_state6 = state_abstractor(state6)
    assert Grasped([robot, stick]) in abstract_state6.atoms
    assert StickAboveButton([stick, button0]) in abstract_state6.atoms
    # AboveNoButton should not be present when stick is above a button
    above_no_button_atoms6 = [
        atom for atom in abstract_state6.atoms if atom.predicate.name == "AboveNoButton"
    ]
    assert len(above_no_button_atoms6) == 0


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
    for _ in range(100):
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


def test_stickbutton2d_skills():
    """Tests for skills in the StickButton2D environment."""
    env = prbench.make("prbench/StickButton2D-b5-v0")
    env_config = StickButton2DEnvConfig()
    onfloor_y = (
        env_config.world_min_y
        + (env_config.world_max_y - env_config.world_min_y - env_config.table_shape[1])
        / 2
    )
    ontable_y = env_config.world_max_y - env_config.table_shape[1] / 2
    env_models = create_bilevel_planning_models(
        "stickbutton2d", env.observation_space, env.action_space, num_buttons=5
    )
    skill_name_to_skill = {s.operator.name: s for s in env_models.skills}
    PickStickFromNothing = skill_name_to_skill["PickStickFromNothing"]
    PlaceStick = skill_name_to_skill["PlaceStick"]
    StickPressButtonFromNothing = skill_name_to_skill["StickPressButtonFromNothing"]
    StickPressButtonFromButton = skill_name_to_skill["StickPressButtonFromButton"]
    RobotPressButtonFromNothing = skill_name_to_skill["RobotPressButtonFromNothing"]
    RobotPressButtonFromButton = skill_name_to_skill["RobotPressButtonFromButton"]

    # Test direct press buttons
    obs0, _ = env.reset(seed=123)
    state0 = env_models.observation_to_state(obs0)
    abstract_state = env_models.state_abstractor(state0)
    obj_name_to_obj = {o.name: o for o in abstract_state.objects}
    robot = obj_name_to_obj["robot"]
    stick = obj_name_to_obj["stick"]
    button0 = obj_name_to_obj["button0"]
    button1 = obj_name_to_obj["button1"]
    button2 = obj_name_to_obj["button2"]
    button3 = obj_name_to_obj["button3"]

    state1 = state0.copy()
    state1.set(button0, "y", onfloor_y)
    state1.set(button1, "y", ontable_y)
    state1.set(button2, "y", onfloor_y)
    state1.set(button3, "y", onfloor_y)
    reset_options = {"init_state": state1}
    obs1, _ = env.reset(seed=123, options=reset_options)
    robot_press_button_from_nothing = RobotPressButtonFromNothing.ground(
        (robot, button3)
    )
    obs2 = _skill_test_helper(robot_press_button_from_nothing, env_models, env, obs1)
    robot_press_button_from_button = RobotPressButtonFromButton.ground(
        (robot, button2, button3)
    )
    obs3 = _skill_test_helper(
        robot_press_button_from_button,
        env_models,
        env,
        obs2,
    )
    # Check that button3 and button2 are pressed
    state3 = env_models.observation_to_state(obs3)
    abstract_state3 = env_models.state_abstractor(state3)
    pred_name_to_pred = {p.name: p for p in env_models.predicates}
    Pressed = pred_name_to_pred["Pressed"]
    RobotAboveButton = pred_name_to_pred["RobotAboveButton"]
    assert Pressed([button3]) in abstract_state3.atoms
    assert Pressed([button2]) in abstract_state3.atoms
    assert RobotAboveButton([robot, button2]) in abstract_state3.atoms

    # Test stick press buttons
    obs0, _ = env.reset(seed=123)
    state0 = env_models.observation_to_state(obs0)
    abstract_state = env_models.state_abstractor(state0)
    obj_name_to_obj = {o.name: o for o in abstract_state.objects}
    robot = obj_name_to_obj["robot"]
    stick = obj_name_to_obj["stick"]
    button0 = obj_name_to_obj["button0"]
    button1 = obj_name_to_obj["button1"]
    button2 = obj_name_to_obj["button2"]
    button3 = obj_name_to_obj["button3"]
    button4 = obj_name_to_obj["button4"]

    state1 = state0.copy()
    state1.set(button0, "y", onfloor_y)
    state1.set(button1, "y", ontable_y)
    state1.set(button2, "y", onfloor_y)
    state1.set(button3, "y", onfloor_y)
    state1.set(button4, "y", ontable_y)
    reset_options = {"init_state": state1}
    obs1, _ = env.reset(seed=123, options=reset_options)
    # First try to directly press button4
    direct_press_button1 = RobotPressButtonFromNothing.ground((robot, button4))
    obs2 = _skill_test_helper(direct_press_button1, env_models, env, obs1)
    # Check that button4 is not pressed
    state2 = env_models.observation_to_state(obs2)
    abstract_state2 = env_models.state_abstractor(state2)
    assert Pressed([button4]) not in abstract_state2.atoms
    # Now try to press button4 with stick
    obs3 = _skill_test_helper(
        PickStickFromNothing.ground((robot, stick)), env_models, env, obs2
    )
    # Check that stick is grasped
    state3 = env_models.observation_to_state(obs3)
    abstract_state3 = env_models.state_abstractor(state3)
    Grasped = pred_name_to_pred["Grasped"]
    assert Grasped([robot, stick]) in abstract_state3.atoms

    # Now press button4 with stick
    obs4 = _skill_test_helper(
        StickPressButtonFromNothing.ground((robot, stick, button4)),
        env_models,
        env,
        obs3,
    )
    # Check that button4 is pressed
    state4 = env_models.observation_to_state(obs4)
    abstract_state4 = env_models.state_abstractor(state4)
    assert Pressed([button4]) in abstract_state4.atoms

    # Now press button1 with stick
    obs5 = _skill_test_helper(
        StickPressButtonFromButton.ground((robot, stick, button1, button4)),
        env_models,
        env,
        obs4,
    )
    # Check that button1 is pressed
    state5 = env_models.observation_to_state(obs5)
    abstract_state5 = env_models.state_abstractor(state5)
    assert Pressed([button1]) in abstract_state5.atoms
    # Finally Place the stick
    obs6 = _skill_test_helper(PlaceStick.ground((robot, stick)), env_models, env, obs5)
    # Check that the robot is no longer grasping the stick
    state6 = env_models.observation_to_state(obs6)
    abstract_state6 = env_models.state_abstractor(state6)
    assert Grasped([robot, stick]) not in abstract_state6.atoms


@pytest.mark.parametrize(
    "num_buttons, max_abstract_plans, samples_per_step",
    [
        (1, 2, 5),
        (2, 5, 5),
        (3, 30, 1),
    ],
)
def test_stickbutton2d_bilevel_planning(
    num_buttons, max_abstract_plans, samples_per_step
):
    """Tests for bilevel planning in the StickButton2D environment.

    Note that we only test a small number of buttons to keep tests fast. Use experiment
    scripts to evaluate at scale.
    """

    env = prbench.make(
        f"prbench/StickButton2D-b{num_buttons}-v0", render_mode="rgb_array"
    )

    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"StickButton2D-b{num_buttons}"
        )

    env_models = create_bilevel_planning_models(
        "stickbutton2d",
        env.observation_space,
        env.action_space,
        num_buttons=num_buttons,
    )
    agent = BilevelPlanningAgent(
        env_models,
        seed=123,
        max_abstract_plans=max_abstract_plans,
        samples_per_step=samples_per_step,
        planning_timeout=60.0,  # Increase timeout for more complex environment
    )
    obs, info = env.reset(seed=123)
    total_reward = 0
    agent.reset(obs, info)
    for _ in range(3000):  # Increase max steps for more complex task
        action = agent.step()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        agent.update(obs, reward, terminated or truncated, info)
        if terminated or truncated:
            break
    else:
        assert False, "Did not terminate successfully"

    env.close()
