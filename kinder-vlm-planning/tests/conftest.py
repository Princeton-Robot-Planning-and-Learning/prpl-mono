"""Shared test fixtures and utilities."""

from unittest.mock import Mock

import numpy as np
import pytest
from bilevel_planning.structs import (
    GroundParameterizedController,
    LiftedParameterizedController,
)
from relational_structs import ObjectCentricState
from relational_structs.objects import Object, Type


@pytest.fixture
def mock_vlm():
    """Create a mock VLM model."""
    vlm = Mock()
    response = Mock()
    response.text = """Here is the plan:

Plan:
move(robot:robot, target:location)[0.5, 0.5]
pick(robot:robot, obj:object)[0.0]
"""
    vlm.query.return_value = response
    return vlm


@pytest.fixture
def sample_types():
    """Create sample types for testing."""
    object_type = Type("object")
    robot_type = Type("robot")
    location_type = Type("location")
    return {"object": object_type, "robot": robot_type, "location": location_type}


@pytest.fixture
def sample_objects(sample_types):  # pylint: disable=redefined-outer-name
    """Create sample objects for testing."""
    robot = Object("robot1", sample_types["robot"])
    obj1 = Object("obj1", sample_types["object"])
    loc1 = Object("loc1", sample_types["location"])
    return {"robot": robot, "obj1": obj1, "loc1": loc1}


@pytest.fixture
def sample_state(sample_objects, sample_types):  # pylint: disable=redefined-outer-name
    """Create a sample object-centric state."""
    # Create object data - map each object to a feature array
    object_data = {
        sample_objects["robot"]: np.array([1.0, 2.0]),
        sample_objects["obj1"]: np.array([3.0, 4.0]),
        sample_objects["loc1"]: np.array([5.0, 6.0]),
    }

    # Create type features - map each type to its feature names
    type_features = {
        sample_types["robot"]: ["x", "y"],
        sample_types["object"]: ["x", "y"],
        sample_types["location"]: ["x", "y"],
    }

    state = ObjectCentricState(object_data, type_features)
    return state


@pytest.fixture
def mock_observation_space(sample_state):  # pylint: disable=redefined-outer-name
    """Create a mock observation space."""
    obs_space = Mock()
    obs_space.devectorize = Mock(return_value=sample_state)
    return obs_space


@pytest.fixture
def mock_controller():
    """Create a mock GroundParameterizedController."""
    controller = Mock(spec=GroundParameterizedController)
    controller.terminated.return_value = False
    controller.step.return_value = np.array([0.1, 0.2, 0.3])
    controller.reset = Mock()
    controller.observe = Mock()
    return controller


@pytest.fixture
def mock_lifted_controller():
    """Create a mock LiftedParameterizedController."""
    controller = Mock(spec=LiftedParameterizedController)
    controller.types = []
    controller.params_space = Mock()
    controller.params_space.shape = (2,)
    controller.ground = Mock()
    controller.var_str = "()"
    return controller


@pytest.fixture
def sample_controllers(sample_types):  # pylint: disable=redefined-outer-name
    """Create sample lifted controllers."""
    move_controller = Mock(spec=LiftedParameterizedController)
    move_controller.types = [sample_types["robot"], sample_types["location"]]
    move_controller.params_space = Mock()
    move_controller.params_space.shape = (2,)
    move_controller.var_str = "(robot, location)"
    move_controller.ground = Mock()

    pick_controller = Mock(spec=LiftedParameterizedController)
    pick_controller.types = [sample_types["robot"], sample_types["object"]]
    pick_controller.params_space = Mock()
    pick_controller.params_space.shape = (1,)
    pick_controller.var_str = "(robot, object)"
    pick_controller.ground = Mock()

    return {"move": move_controller, "pick": pick_controller}


@pytest.fixture
def mock_action_space():
    """Create a mock action space."""
    action_space = Mock()
    action_space.shape = (3,)
    action_space.sample.return_value = np.array([0.0, 0.0, 0.0])
    return action_space


@pytest.fixture
def sample_observation():
    """Create a sample observation."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def sample_model_output():
    """Sample valid model output for plan parsing."""
    return """move(robot1:robot, loc1:location)[0.5, 0.5]
pick(robot1:robot, obj1:object)[0.0]
"""


@pytest.fixture
def mock_env_info():
    """Create mock environment info."""
    return {"description": "Test task description"}
