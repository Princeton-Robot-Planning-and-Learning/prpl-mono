"""Tests for utils.py."""

import numpy as np
import pytest
import spatialmath

from prpl_tidybot.constants import BASE_CAMERA_DIMS, WRIST_CAMERA_DIMS
from prpl_tidybot.structs import TidyBotAction, TidyBotObservation


def test_tidybot_observation():
    """Tests for TidyBotObservation()."""
    obs = TidyBotObservation(
        arm_conf=[0.0] * 7,
        base_pose=spatialmath.SE2(x=0, y=0, theta=0),
        gripper=0.0,
        wrist_camera=np.zeros(WRIST_CAMERA_DIMS, dtype=np.uint8),
        base_camera=np.zeros(BASE_CAMERA_DIMS, dtype=np.uint8),
    )
    assert np.allclose(obs.arm_conf, [0.0] * 7)
    # Compare homogeneous transform matrices for the SE2 poses
    assert np.allclose(obs.base_pose.A, spatialmath.SE2(x=0, y=0, theta=0).A)
    assert np.isclose(obs.gripper, 0.0)
    # Test with incorrect image size, should throw assertion error.
    with pytest.raises(AssertionError):
        TidyBotObservation(
            arm_conf=[0.0] * 7,
            base_pose=spatialmath.SE2(x=0, y=0, theta=0),
            gripper=0.0,
            wrist_camera=np.zeros((1, 1, 1), dtype=np.uint8),
            base_camera=np.zeros(BASE_CAMERA_DIMS, dtype=np.uint8),
        )


def test_tidybot_action():
    """Tests for TidyBotAction()."""
    arm_goal = [1.0, 0.5, -0.5, 0.0, 0.1, -0.1, 0.2]
    base_goal = spatialmath.SE2(x=1.0, y=-2.0, theta=0.5)
    action = TidyBotAction(
        arm_goal=arm_goal, base_local_goal=base_goal, gripper_goal=1.0
    )
    assert np.allclose(action.arm_goal, arm_goal)
    assert np.allclose(action.base_local_goal.A, base_goal.A)
    assert action.gripper_goal == 1.0
