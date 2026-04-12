"""Tests for kinder_ground_perceiver.py."""

import numpy as np
import spatialmath
from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv
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


def test_real_to_sim_kinder_ground():
    """Tests real-to-sim in the KinDER Ground environment."""

    # Create fake interface to real.
    interface = FakeInterface()
    interface.arm_interface.arm_state = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    interface.base_interface.base_state = spatialmath.SE2(x=1.0, y=1.0, theta=0.0)
    perceiver = KinDERGroundPerceiver(interface)

    # Create sim.
    sim = ObjectCentricTidyBot3DEnv(
        scene_type="base_motion",
        num_objects=1,
        allow_state_access=True,
    )

    # Get the real state from the perceiver.
    state = perceiver.get_state()

    # Set the sim.
    sim.reset(seed=123)
    sim.set_state(state)

    # Uncomment to visualize.
    # sim._robot_env.sim.forward()  # pylint: disable=protected-access
    # img = sim.render()
    # import imageio.v2 as iio
    # iio.imsave("real_to_sim_ground_image.png", img)
