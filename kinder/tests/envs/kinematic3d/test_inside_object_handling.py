"""Tests for inside_object handling in kinematic3d environments."""

import pytest
from pybullet_helpers.geometry import Pose, get_pose, multiply_poses, set_pose

from kinder.envs.kinematic3d.transport3d import (
    ObjectCentricTransport3DEnv,
    Transport3DEnv,
    Transport3DObjectCentricState,
)


@pytest.fixture(scope="module")
def env():
    """Create a shared environment for all tests in this module."""
    environment = Transport3DEnv(
        num_cubes=2,
        num_boxes=1,
        use_gui=False,
        render_mode="rgb_array",
        realistic_bg=False,
    )
    yield environment
    environment.close()


def test_inside_object_state_restoration(env):  # pylint: disable=redefined-outer-name
    """Test that inside objects are correctly derived after state restoration.

    This tests the bug where inside_object_list was not serialized, so set_state()
    didn't restore it. The fix derives inside objects on-demand from poses.
    """
    config = (
        env.unwrapped._object_centric_env.config  # pylint: disable=protected-access
    )

    vec_obs, _ = env.reset(seed=42)
    oc_obs = env.observation_space.devectorize(vec_obs)
    obs = Transport3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Create a simulator.
    sim = ObjectCentricTransport3DEnv(
        num_cubes=2,
        num_boxes=1,
        config=config,
        realistic_bg=False,
        allow_state_access=True,
    )
    sim.set_state(obs)

    # Manually position a cube inside the box
    box_pose = obs.get_object_pose("box0")
    cube_inside_pose = Pose(
        (box_pose.position[0], box_pose.position[1], box_pose.position[2] + 0.02),
        (0, 0, 0, 1),
    )
    cube_id = sim._cubes["cube0"]  # pylint: disable=protected-access
    set_pose(cube_id, cube_inside_pose, sim.physics_client_id)

    # Manually set grasp state (simulating having grasped the box)
    sim._grasped_object = "box0"  # pylint: disable=protected-access
    world_to_robot = sim.robot.arm.get_end_effector_pose()
    box_id = sim._boxes["box0"]  # pylint: disable=protected-access
    world_to_object = get_pose(box_id, sim.physics_client_id)
    sim._grasped_object_transform = multiply_poses(  # pylint: disable=protected-access
        world_to_robot.invert(), world_to_object
    )

    # Get the current observation with the grasp state
    obs_with_grasp = sim._get_obs()  # pylint: disable=protected-access

    # Verify inside objects are correctly computed
    inside_names, inside_transforms = (
        sim._get_inside_objects()  # pylint: disable=protected-access
    )
    assert "cube0" in inside_names, f"cube0 should be inside box0, got: {inside_names}"
    assert "cube0" in inside_transforms, "cube0 transform should be available"

    # Now create a NEW simulator and restore state to it
    sim2 = ObjectCentricTransport3DEnv(
        num_cubes=2,
        num_boxes=1,
        config=config,
        realistic_bg=False,
        allow_state_access=True,
    )
    sim2.set_state(obs_with_grasp)

    # After state restoration, inside objects should be correctly derived
    # from the current poses (this was the bug - inside_object_list was not
    # serialized, so set_state() didn't restore it)
    inside_names2, inside_transforms2 = (
        sim2._get_inside_objects()  # pylint: disable=protected-access
    )

    assert (
        "cube0" in inside_names2
    ), f"cube0 should be inside box0 after state restoration, got: {inside_names2}"
    assert (
        "cube0" in inside_transforms2
    ), "cube0 transform should be available after state restoration"

    sim.close()
    sim2.close()
