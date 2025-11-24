"""Tests for agent.py."""

# pylint: disable=redefined-outer-name,protected-access

from unittest.mock import Mock, patch

import numpy as np
import pytest
from relational_structs.objects import Type

from prbench_vlm_planning.agent import VLMPlanningAgent, VLMPlanningAgentFailure


@pytest.fixture
def agent_kwargs(mock_observation_space, sample_controllers):
    """Common kwargs for creating VLMPlanningAgent."""
    return {
        "observation_space": mock_observation_space,
        "env_controllers": sample_controllers,
        "vlm_model_name": "gpt-5",
        "temperature": 0.0,
        "max_planning_horizon": 50,
        "seed": 0,
        "rgb_observation": False,
    }


@pytest.fixture
def mock_prompt_file(tmp_path):
    """Create a mock prompt file."""
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    prompt_file = prompt_dir / "vlm_planning_prompt.txt"
    prompt_file.write_text(
        "Controllers:\n{controllers}\n"
        "Objects:\n{typed_objects}\n"
        "Types:\n{type_hierarchy}\n"
        "Init State:\n{init_state_str}\n"
        "Goal:\n{goal_str}\n"
    )
    return prompt_file


def test_agent_initialization(agent_kwargs):
    """Test VLMPlanningAgent initialization."""
    with patch("prbench_vlm_planning.agent.create_vlm_by_name") as mock_create_vlm:
        mock_vlm = Mock()
        mock_create_vlm.return_value = mock_vlm

        with patch("builtins.open", create=True):
            with patch("os.path.dirname"):
                with patch("os.path.join"):
                    agent = VLMPlanningAgent(**agent_kwargs)

        assert agent._vlm_model_name == "gpt-5"
        assert agent._temperature == 0.0
        assert agent._max_planning_horizon == 50
        assert agent._seed == 0
        assert agent._rgb_observation is False
        assert agent._current_policy is None
        assert agent._next_action is None


def test_agent_reset_success(agent_kwargs, mock_env_info, sample_observation, mock_vlm):
    """Test successful agent reset with plan generation."""
    with patch("prbench_vlm_planning.agent.create_vlm_by_name") as mock_create_vlm:
        mock_create_vlm.return_value = mock_vlm

        with patch("builtins.open", create=True):
            with patch("os.path.dirname"):
                with patch("os.path.join"):
                    agent = VLMPlanningAgent(**agent_kwargs)

        # Mock the policy generation
        mock_policy = Mock()
        mock_policy.return_value = np.array([0.1, 0.2, 0.3])

        with patch.object(agent, "_generate_plan", return_value=mock_policy):
            agent.reset(sample_observation, mock_env_info)

        assert agent._current_policy is not None
        assert agent._next_action is not None
        assert agent._plan_step == 0


def test_agent_reset_failure(agent_kwargs, mock_env_info, sample_observation):
    """Test agent reset failure when plan generation fails."""
    with patch("prbench_vlm_planning.agent.create_vlm_by_name") as mock_create_vlm:
        mock_vlm = Mock()
        mock_create_vlm.return_value = mock_vlm

        with patch("builtins.open", create=True):
            with patch("os.path.dirname"):
                with patch("os.path.join"):
                    agent = VLMPlanningAgent(**agent_kwargs)

        # Mock plan generation to fail
        with patch.object(
            agent, "_generate_plan", side_effect=Exception("VLM query failed")
        ):
            with pytest.raises(VLMPlanningAgentFailure, match="Failed to generate"):
                agent.reset(sample_observation, mock_env_info)


def test_agent_get_action(agent_kwargs, mock_env_info, sample_observation, mock_vlm):
    """Test getting action from agent."""
    with patch("prbench_vlm_planning.agent.create_vlm_by_name") as mock_create_vlm:
        mock_create_vlm.return_value = mock_vlm

        with patch("builtins.open", create=True):
            with patch("os.path.dirname"):
                with patch("os.path.join"):
                    agent = VLMPlanningAgent(**agent_kwargs)

        # Setup agent with a policy
        mock_policy = Mock()
        expected_action = np.array([0.1, 0.2, 0.3])
        mock_policy.return_value = expected_action

        with patch.object(agent, "_generate_plan", return_value=mock_policy):
            agent.reset(sample_observation, mock_env_info)

        # Get action
        action = agent.step()

        assert np.array_equal(action, expected_action)
        assert agent._plan_step == 1


def test_agent_get_action_no_policy(agent_kwargs):
    """Test getting action when no policy is available."""
    with patch("prbench_vlm_planning.agent.create_vlm_by_name") as mock_create_vlm:
        mock_vlm = Mock()
        mock_create_vlm.return_value = mock_vlm

        with patch("builtins.open", create=True):
            with patch("os.path.dirname"):
                with patch("os.path.join"):
                    agent = VLMPlanningAgent(**agent_kwargs)

        with pytest.raises(VLMPlanningAgentFailure, match="No current plan available"):
            agent._get_action()


def test_agent_get_action_plan_exhausted(
    agent_kwargs, mock_env_info, sample_observation, mock_vlm
):
    """Test getting action when plan is exhausted."""
    with patch("prbench_vlm_planning.agent.create_vlm_by_name") as mock_create_vlm:
        mock_create_vlm.return_value = mock_vlm

        with patch("builtins.open", create=True):
            with patch("os.path.dirname"):
                with patch("os.path.join"):
                    agent = VLMPlanningAgent(**agent_kwargs)

        # Setup agent with a policy
        mock_policy = Mock()
        mock_policy.return_value = np.array([0.1, 0.2, 0.3])

        with patch.object(agent, "_generate_plan", return_value=mock_policy):
            agent.reset(sample_observation, mock_env_info)

        # Exhaust the plan
        agent._plan_step = agent._max_planning_horizon

        with pytest.raises(VLMPlanningAgentFailure, match="Plan exhausted"):
            agent._get_action()


def test_agent_update(agent_kwargs, mock_env_info, sample_observation, mock_vlm):
    """Test agent update with new observation."""
    with patch("prbench_vlm_planning.agent.create_vlm_by_name") as mock_create_vlm:
        mock_create_vlm.return_value = mock_vlm

        with patch("builtins.open", create=True):
            with patch("os.path.dirname"):
                with patch("os.path.join"):
                    agent = VLMPlanningAgent(**agent_kwargs)

        # Setup agent with a policy
        mock_policy = Mock()
        action1 = np.array([0.1, 0.2, 0.3])
        action2 = np.array([0.4, 0.5, 0.6])
        mock_policy.side_effect = [action1, action2]

        with patch.object(agent, "_generate_plan", return_value=mock_policy):
            agent.reset(sample_observation, mock_env_info)

        # Call step() first to set _last_action
        first_action = agent.step()
        assert np.array_equal(first_action, action1)

        # Update agent
        new_obs = np.array([2.0, 3.0, 4.0])
        agent.update(new_obs, reward=1.0, done=False, info={})

        # Next action should be updated
        assert np.array_equal(agent._next_action, action2)


def test_agent_get_goal_str(agent_kwargs, mock_env_info):
    """Test getting goal description string."""
    with patch("prbench_vlm_planning.agent.create_vlm_by_name") as mock_create_vlm:
        mock_vlm = Mock()
        mock_create_vlm.return_value = mock_vlm

        with patch("builtins.open", create=True):
            with patch("os.path.dirname"):
                with patch("os.path.join"):
                    agent = VLMPlanningAgent(**agent_kwargs)

        goal_str = agent._get_goal_str(mock_env_info)

        assert goal_str == "Test task description"


def test_agent_create_types_str_no_hierarchy(agent_kwargs):
    """Test creating types string without hierarchy."""
    with patch("prbench_vlm_planning.agent.create_vlm_by_name") as mock_create_vlm:
        mock_vlm = Mock()
        mock_create_vlm.return_value = mock_vlm

        with patch("builtins.open", create=True):
            with patch("os.path.dirname"):
                with patch("os.path.join"):
                    agent = VLMPlanningAgent(**agent_kwargs)

        types = [Type("object"), Type("robot"), Type("location")]
        types_str = agent.create_types_str(types)

        assert "location" in types_str
        assert "object" in types_str
        assert "robot" in types_str


def test_agent_create_types_str_with_hierarchy(agent_kwargs):
    """Test creating types string with hierarchy."""
    with patch("prbench_vlm_planning.agent.create_vlm_by_name") as mock_create_vlm:
        mock_vlm = Mock()
        mock_create_vlm.return_value = mock_vlm

        with patch("builtins.open", create=True):
            with patch("os.path.dirname"):
                with patch("os.path.join"):
                    agent = VLMPlanningAgent(**agent_kwargs)

        parent_type = Type("entity")
        child_type1 = Type("robot", parent=parent_type)
        child_type2 = Type("object", parent=parent_type)

        types = [parent_type, child_type1, child_type2]
        types_str = agent.create_types_str(types)

        assert "entity" in types_str
        assert "robot" in types_str
        assert "object" in types_str


def test_agent_get_controllers_str(agent_kwargs):
    """Test getting controllers string representation."""
    with patch("prbench_vlm_planning.agent.create_vlm_by_name") as mock_create_vlm:
        mock_vlm = Mock()
        mock_create_vlm.return_value = mock_vlm

        with patch("builtins.open", create=True):
            with patch("os.path.dirname"):
                with patch("os.path.join"):
                    agent = VLMPlanningAgent(**agent_kwargs)

        controllers_str = agent._get_controllers_str()

        assert "move" in controllers_str
        assert "pick" in controllers_str


def test_agent_generate_plan_vlm_failure(
    agent_kwargs, mock_env_info, sample_observation
):
    """Test plan generation when VLM query fails."""
    with patch("prbench_vlm_planning.agent.create_vlm_by_name") as mock_create_vlm:
        mock_vlm = Mock()
        mock_vlm.query.side_effect = Exception("API Error")
        mock_create_vlm.return_value = mock_vlm

        with patch("builtins.open", create=True):
            with patch("os.path.dirname"):
                with patch("os.path.join"):
                    agent = VLMPlanningAgent(**agent_kwargs)

        with pytest.raises(VLMPlanningAgentFailure, match="VLM query failed"):
            agent._generate_plan(sample_observation, mock_env_info)


def test_agent_with_image_observation(agent_kwargs, mock_env_info, mock_vlm):
    """Test agent with image observation."""
    agent_kwargs["rgb_observation"] = True

    with patch("prbench_vlm_planning.agent.create_vlm_by_name") as mock_create_vlm:
        mock_create_vlm.return_value = mock_vlm

        with patch("builtins.open", create=True):
            with patch("os.path.dirname"):
                with patch("os.path.join"):
                    agent = VLMPlanningAgent(**agent_kwargs)

        # Create observation with RGB image
        rgb_image = np.zeros((100, 100, 3), dtype=np.uint8)
        obs_dict = {"rgb": rgb_image, "state": np.array([1.0, 2.0])}

        mock_policy = Mock()
        mock_policy.return_value = np.array([0.1, 0.2])

        with patch("prbench_vlm_planning.agent.PIL.Image.fromarray") as mock_from_array:
            mock_pil_image = Mock()
            mock_from_array.return_value = mock_pil_image

            # Mock the state that will be returned by devectorize
            mock_state = Mock()
            mock_state.data = []
            mock_state.type_features = []
            mock_state.pretty_str.return_value = "state"

            with (
                patch.object(
                    agent._observation_space, "devectorize", return_value=mock_state
                ) as mock_devectorize,
                patch.object(agent, "_get_controllers_str") as mock_get_controllers,
                patch.object(agent, "_get_goal_str") as mock_get_goal,
                patch.object(agent, "create_types_str") as mock_create_types,
                patch(
                    "prbench_vlm_planning.agent.controller_and_param_plan_to_policy",
                    return_value=mock_policy,
                ),
            ):
                mock_get_controllers.return_value = "controllers"
                mock_get_goal.return_value = "goal"
                mock_create_types.return_value = "types"

                with patch(
                    "prbench_vlm_planning.agent.parse_model_output_into_option_plan",
                    return_value=[],
                ):
                    try:
                        agent._generate_plan(obs_dict, mock_env_info)
                    except Exception:
                        pass  # Expected to fail due to empty plan

                # Verify VLM was called with images
                mock_vlm.query.assert_called_once()
                call_kwargs = mock_vlm.query.call_args[1]
                assert "imgs" in call_kwargs
                # Verify devectorize was called
                mock_devectorize.assert_called_once()
