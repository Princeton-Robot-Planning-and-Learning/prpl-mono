"""Tests for kinder_ground_perceiver.py."""

import numpy as np
import spatialmath
from kinder.envs.dynamic3d.object_types import MujocoTidyBotRobotObjectType
from relational_structs import ObjectCentricState

from prpl_tidybot.interfaces.interface import FakeInterface
from prpl_tidybot.perceivers.kinder_ground_perceiver import KinDERGroundPerceiver


def _get_robot_from_state(state: ObjectCentricState):
    """Helper to get robot object from state by type."""
    robots = state.get_objects(MujocoTidyBotRobotObjectType)
    assert len(robots) == 1, f"Expected 1 robot, got {len(robots)}"
    return list(robots)[0]


def test_kinder_ground_perceiver():
    """Tests for KinDERGroundPerceiver()."""
    interface = FakeInterface()
    interface.arm_interface.arm_state = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    interface.base_interface.map_base_state = spatialmath.SE2(x=1.0, y=0.0, theta=0.0)
    perceiver = KinDERGroundPerceiver(interface)
    state = perceiver.get_state()
    robot_obj = _get_robot_from_state(state)
    assert np.isclose(state.get(robot_obj, "pos_arm_joint1"), 1.0)
    assert np.isclose(state.get(robot_obj, "pos_arm_joint2"), 0.0)
    assert np.isclose(state.get(robot_obj, "pos_arm_joint3"), 0.0)
    assert np.isclose(state.get(robot_obj, "pos_arm_joint4"), 0.0)
    assert np.isclose(state.get(robot_obj, "pos_arm_joint5"), 0.0)
    assert np.isclose(state.get(robot_obj, "pos_arm_joint6"), 0.0)
    assert np.isclose(state.get(robot_obj, "pos_base_x"), 1.0)
    assert np.isclose(state.get(robot_obj, "pos_base_y"), 0.0)
    assert np.isclose(state.get(robot_obj, "pos_base_rot"), 0.0)
