"""Tests for prbench_ground_perceiver.py."""

import numpy as np
import spatialmath

from prpl_tidybot.interfaces.interface import FakeInterface
from prpl_tidybot.perceivers.prbench_ground_perceiver import PRBenchGroundPerceiver


def test_prbench_ground_perceiver():
    """Tests for PRBenchGroundPerceiver()."""
    interface = FakeInterface()
    interface.arm_interface.arm_state = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    interface.base_interface.base_state = spatialmath.SE2(x=1.0, y=0.0, theta=0.0)
    perceiver = PRBenchGroundPerceiver(interface)
    state = perceiver.get_state()
    robot_obj = state.get_object_from_name("robot")
    assert np.isclose(state.get(robot_obj, "pos_arm_joint1"), 1.0)
    assert np.isclose(state.get(robot_obj, "pos_arm_joint2"), 0.0)
    assert np.isclose(state.get(robot_obj, "pos_arm_joint3"), 0.0)
    assert np.isclose(state.get(robot_obj, "pos_arm_joint4"), 0.0)
    assert np.isclose(state.get(robot_obj, "pos_arm_joint5"), 0.0)
    assert np.isclose(state.get(robot_obj, "pos_arm_joint6"), 0.0)
    assert np.isclose(state.get(robot_obj, "pos_base_x"), 1.0)
    assert np.isclose(state.get(robot_obj, "pos_base_y"), 0.0)
    assert np.isclose(state.get(robot_obj, "pos_base_rot"), 0.0)
