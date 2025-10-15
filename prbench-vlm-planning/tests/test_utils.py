"""Tests for utils.py."""

# pylint: disable=unused-argument

from unittest.mock import Mock, patch

import numpy as np
import pytest
from bilevel_planning.structs import GroundParameterizedController

from prbench_vlm_planning.utils import (
    controller_and_param_plan_to_policy,
    create_vlm_by_name,
    option_policy_to_policy,
    parse_model_output_into_option_plan,
)


def test_create_vlm_by_name_success(tmp_path):
    """Test successful VLM model creation."""
    with patch("prbench_vlm_planning.utils.Path") as mock_path:
        mock_path.return_value = tmp_path / "vlm_cache"
        with patch("prbench_vlm_planning.utils.OpenAIModel") as mock_openai:
            mock_model = Mock()
            mock_openai.return_value = mock_model

            vlm = create_vlm_by_name("gpt-4o")

            assert vlm == mock_model
            mock_openai.assert_called_once()


def test_create_vlm_by_name_failure(tmp_path):
    """Test VLM model creation failure handling."""
    with patch("prbench_vlm_planning.utils.Path") as mock_path:
        mock_path.return_value = tmp_path / "vlm_cache"
        with patch("prbench_vlm_planning.utils.OpenAIModel") as mock_openai:
            mock_openai.side_effect = Exception("API Error")

            with pytest.raises(ValueError, match="Failed to create VLM model"):
                create_vlm_by_name("invalid-model")


def test_parse_model_output_valid(sample_objects, sample_types, sample_controllers):
    """Test parsing valid model output."""
    model_output = (
        "move(robot1:robot, loc1:location)[0.5, 0.5]\n"
        "pick(robot1:robot, obj1:object)[0.0]"
    )

    # Setup mocks for ground method
    sample_controllers["move"].ground.return_value = Mock()
    sample_controllers["pick"].ground.return_value = Mock()

    plan = parse_model_output_into_option_plan(
        model_output,
        set(sample_objects.values()),
        set(sample_types.values()),
        sample_controllers,
        parse_continuous_params=True,
    )

    assert len(plan) == 2
    # First action is move
    assert plan[0][0] == sample_controllers["move"]
    assert len(plan[0][1]) == 2  # 2 objects
    assert plan[0][2] == [0.5, 0.5]  # 2 continuous params

    # Second action is pick
    assert plan[1][0] == sample_controllers["pick"]
    assert len(plan[1][1]) == 2  # 2 objects
    assert plan[1][2] == [0.0]  # 1 continuous param


def test_parse_model_output_invalid_option_name(
    sample_objects, sample_types, sample_controllers
):
    """Test parsing with invalid option name."""
    model_output = "invalid_action(robot1:robot, loc1:location)[0.5, 0.5]"

    plan = parse_model_output_into_option_plan(
        model_output,
        set(sample_objects.values()),
        set(sample_types.values()),
        sample_controllers,
        parse_continuous_params=True,
    )

    assert len(plan) == 0  # Invalid action should be ignored


def test_parse_model_output_missing_params(
    sample_objects, sample_types, sample_controllers
):
    """Test parsing with missing continuous parameters."""
    model_output = "move(robot1:robot, loc1:location)"  # Missing [params]

    plan = parse_model_output_into_option_plan(
        model_output,
        set(sample_objects.values()),
        set(sample_types.values()),
        sample_controllers,
        parse_continuous_params=True,
    )

    assert len(plan) == 0  # Should terminate due to missing params


def test_parse_model_output_invalid_object(
    sample_objects, sample_types, sample_controllers
):
    """Test parsing with invalid object name."""
    model_output = "move(invalid_obj:robot, loc1:location)[0.5, 0.5]"

    plan = parse_model_output_into_option_plan(
        model_output,
        set(sample_objects.values()),
        set(sample_types.values()),
        sample_controllers,
        parse_continuous_params=True,
    )

    assert len(plan) == 0  # Invalid object should cause parsing to stop


def test_parse_model_output_empty_lines(
    sample_objects, sample_types, sample_controllers
):
    """Test parsing with empty lines."""
    model_output = "\nmove(robot1:robot, loc1:location)[0.5, 0.5]\n\n"

    sample_controllers["move"].ground.return_value = Mock()

    plan = parse_model_output_into_option_plan(
        model_output,
        set(sample_objects.values()),
        set(sample_types.values()),
        sample_controllers,
        parse_continuous_params=True,
    )

    assert len(plan) == 1  # Empty lines should be skipped


def test_controller_and_param_plan_to_policy(mock_controller, mock_observation_space):
    """Test converting controller plan to policy."""
    controller_plan = [(mock_controller, [0.5, 0.5]), (mock_controller, [0.3])]

    policy = controller_and_param_plan_to_policy(
        controller_plan, max_horizon=10, observation_space=mock_observation_space
    )

    # Policy should be callable
    assert callable(policy)

    # Execute policy
    obs = np.array([1.0, 2.0, 3.0])
    action = policy(obs)

    # Should get action from controller
    assert action is not None
    mock_controller.step.assert_called()


def test_option_policy_to_policy(mock_controller, mock_observation_space):
    """Test converting option policy to policy."""

    # Create a simple option policy that returns controller and params
    def option_policy(obs):
        return mock_controller, [0.5, 0.5]

    policy = option_policy_to_policy(
        option_policy, max_horizon=10, observation_space=mock_observation_space
    )

    # Policy should be callable
    assert callable(policy)

    # Execute policy
    obs = np.array([1.0, 2.0, 3.0])
    action = policy(obs)

    # Should get action from controller
    assert action is not None
    mock_controller.reset.assert_called_once()
    mock_controller.step.assert_called()


def test_option_policy_to_policy_controller_termination(
    mock_observation_space,
):
    """Test that policy gets new controller when current one terminates."""
    controller1 = Mock(spec=GroundParameterizedController)
    controller1.terminated.return_value = True
    controller1.step.return_value = np.array([0.1, 0.2])

    controller2 = Mock(spec=GroundParameterizedController)
    controller2.terminated.return_value = False
    controller2.step.return_value = np.array([0.3, 0.4])

    call_count = [0]

    def option_policy(obs):
        call_count[0] += 1
        if call_count[0] == 1:
            return controller1, [0.5]
        return controller2, [0.3]

    policy = option_policy_to_policy(
        option_policy, max_horizon=10, observation_space=mock_observation_space
    )

    obs = np.array([1.0, 2.0, 3.0])

    # First call should use controller1
    action1 = policy(obs)
    assert action1 is not None

    # Second call should get controller2 since controller1 terminated
    action2 = policy(obs)
    assert action2 is not None

    # Both controllers should have been reset
    controller1.reset.assert_called_once()
    controller2.reset.assert_called_once()


def test_option_policy_to_policy_max_horizon(mock_observation_space):
    """Test that policy raises exception when max horizon exceeded."""
    controller = Mock(spec=GroundParameterizedController)
    controller.terminated.return_value = False
    controller.step.return_value = np.array([0.1, 0.2])

    def option_policy(obs):
        return controller, [0.5]

    policy = option_policy_to_policy(
        option_policy, max_horizon=2, observation_space=mock_observation_space
    )

    obs = np.array([1.0, 2.0, 3.0])

    # First two calls should work
    policy(obs)
    policy(obs)

    # Mark controller as terminated to trigger new controller request
    controller.terminated.return_value = True

    # Third call should exceed max horizon
    with pytest.raises(Exception, match="Exceeded max controller steps"):
        policy(obs)
