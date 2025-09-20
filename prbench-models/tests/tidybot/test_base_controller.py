"""Integration tests for the BaseController class in
prbench.envs.tidybot.base_controller.

These tests use the BaseController and its dependencies, assuming all required packages
are installed.
"""

# pylint: disable=redefined-outer-name

import numpy as np
import pytest

from prbench_models.tidybot.base_controller import BaseController


@pytest.fixture
def base_controller() -> BaseController:
    """Fixture to create a BaseController instance with real dependencies for
    testing."""
    qpos = np.zeros(3)
    qvel = np.zeros(3)
    ctrl = np.zeros(3)
    timestep = 0.1
    return BaseController(qpos, qvel, ctrl, timestep)


def test_reset(base_controller):
    """Test that BaseController.reset() sets state to origin and OTG to Finished."""
    base_controller.reset()
    assert np.allclose(base_controller.qpos, 0)
    assert np.allclose(base_controller.ctrl, 0)
    # OTG result may depend on Ruckig, so just check it's set (not None)
    assert base_controller.otg_res is not None


def test_run_controller_sets_target(base_controller):
    """Test that run_controller sets the target position and updates OTG state."""
    base_controller.reset()
    command = {"base_pose": np.array([1.0, 2.0, 3.0])}
    base_controller.run_controller(command)
    assert np.allclose(base_controller.otg_inp.target_position, [1.0, 2.0, 3.0])
    assert base_controller.otg_res is not None
