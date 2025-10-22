"""Tests for utils.py."""

from prpl_tidybot.structs import TidyBotObservation, TidyBotAction
import spatialmath
import numpy as np

def test_tidybot_observation():
    """Tests for TidyBotObservation()."""
    obs = TidyBotObservation(
        arm_conf=[0.0] * 7,
        base_pose=spatialmath.SE2(x=0, y=0, theta=0),
        gripper=0.0
    )
    assert np.allclose(obs.arm_conf, [0.0] * 7)
    # Compare homogeneous transform matrices for the SE2 poses
    assert np.allclose(obs.base_pose.A, spatialmath.SE2(x=0, y=0, theta=0).A)
    assert np.isclose(obs.gripper, 0.0)


def test_tidybot_action():
    """Tests for TidyBotAction()."""
    arm_goal = [1.0, 0.5, -0.5, 0.0, 0.1, -0.1, 0.2]
    base_goal = spatialmath.SE2(x=1.0, y=-2.0, theta=0.5)
    action = TidyBotAction(
        arm_goal=arm_goal,
        base_local_goal=base_goal,
        gripper_goal=1.0
    )
    assert np.allclose(action.arm_goal, arm_goal)
    assert np.allclose(action.base_local_goal.A, base_goal.A)
    assert action.gripper_goal == 1.0
