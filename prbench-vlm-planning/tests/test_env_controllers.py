"""Tests for env_controllers.py."""

from unittest.mock import Mock, patch

import pytest

from prbench_vlm_planning.env_controllers import (
    _import_lifted_controllers,
    get_controllers_for_environment,
)


def test_get_controllers_for_environment_success():
    """Test successful controller loading."""
    with patch(
        "prbench_vlm_planning.env_controllers._import_lifted_controllers"
    ) as mock_import:
        mock_controllers = {"move": Mock(), "pick": Mock()}
        mock_import.return_value = mock_controllers

        controllers = get_controllers_for_environment(
            "geom2d", "motion2d", action_space=Mock()
        )

        assert controllers == mock_controllers
        mock_import.assert_called_once()


def test_get_controllers_for_environment_not_found():
    """Test controller loading when module not found."""
    with patch(
        "prbench_vlm_planning.env_controllers._import_lifted_controllers"
    ) as mock_import:
        mock_import.return_value = None

        controllers = get_controllers_for_environment(
            "invalid_env", "invalid_name", action_space=None
        )

        assert controllers is None


def test_import_lifted_controllers_success():
    """Test successful import of lifted controllers."""
    mock_module = Mock()
    mock_controllers = {"move": Mock(), "pick": Mock()}

    def mock_create_lifted_controllers(
        action_space, init_constant_state
    ):  # pylint: disable=unused-argument
        return mock_controllers

    mock_module.create_lifted_controllers = mock_create_lifted_controllers

    with patch("importlib.import_module", return_value=mock_module):
        controllers = _import_lifted_controllers(
            "prbench_models.geom2d.envs.motion2d.parameterized_skills",
            "motion2d",
            action_space=Mock(),
        )

        assert controllers == mock_controllers


def test_import_lifted_controllers_missing_method():
    """Test import when create_lifted_controllers method is missing."""
    mock_module = Mock()
    del mock_module.create_lifted_controllers

    with patch("importlib.import_module", return_value=mock_module):
        with pytest.raises(
            NotImplementedError,
            match="does not have a create_lifted_controllers method",
        ):
            _import_lifted_controllers(
                "prbench_models.geom2d.envs.motion2d.parameterized_skills",
                "motion2d",
                action_space=None,
            )


def test_import_lifted_controllers_import_error():
    """Test import when module cannot be imported."""
    with patch("importlib.import_module", side_effect=ImportError("Module not found")):
        controllers = _import_lifted_controllers(
            "nonexistent.module.path", "test_env", action_space=None
        )

        assert controllers is None


def test_import_lifted_controllers_exception():
    """Test import when an unexpected exception occurs."""
    with patch("importlib.import_module", side_effect=Exception("Unexpected error")):
        controllers = _import_lifted_controllers(
            "prbench_models.geom2d.envs.motion2d.parameterized_skills",
            "motion2d",
            action_space=None,
        )

        assert controllers is None


def test_import_lifted_controllers_empty_result():
    """Test import when create_lifted_controllers returns None."""
    mock_module = Mock()
    mock_module.create_lifted_controllers.return_value = None

    with patch("importlib.import_module", return_value=mock_module):
        controllers = _import_lifted_controllers(
            "prbench_models.geom2d.envs.motion2d.parameterized_skills",
            "motion2d",
            action_space=None,
        )

        assert controllers is None


def test_import_lifted_controllers_with_action_space():
    """Test import with action space parameter."""
    mock_module = Mock()
    mock_action_space = Mock()
    mock_controllers = {"move": Mock()}

    def mock_create_lifted_controllers(action_space, init_constant_state):
        assert action_space == mock_action_space
        assert init_constant_state is None
        return mock_controllers

    mock_module.create_lifted_controllers = mock_create_lifted_controllers

    with patch("importlib.import_module", return_value=mock_module):
        controllers = _import_lifted_controllers(
            "prbench_models.geom2d.envs.motion2d.parameterized_skills",
            "motion2d",
            action_space=mock_action_space,
        )

        assert controllers == mock_controllers
