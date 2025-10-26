"""Test utils for dynamic3d models."""

import numpy as np
import prbench
from relational_structs.spaces import ObjectCentricBoxSpace

from prbench_models.dynamic3d.utils import get_overhead_object_se2_pose

prbench.register_all_environments()


def test_get_overhead_object_se2_pose():
    """Tests for get_overhead_object_se2_pose()."""

    # Get a real object-centric state.
    env = prbench.make("prbench/TidyBot3D-ground-o1-v0")
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    obs, _ = env.reset(seed=123)
    state1 = env.observation_space.devectorize(obs)
    cube = state1.get_object_from_name("cube1")

    # Extract the initial SE2 pose.
    pose1 = get_overhead_object_se2_pose(state1, cube)

    # Moving the object z shouldn't change anything.
    state2 = state1.copy()
    state2.set(cube, "z", 1000)
    pose2 = get_overhead_object_se2_pose(state2, cube)
    assert np.allclose(pose1.A, pose2.A, atol=1e-5)

    # Move the object x should have an effect.
    state3 = state1.copy()
    state3.set(cube, "x", state1.get(cube, "x") + 1.0)
    pose3 = get_overhead_object_se2_pose(state3, cube)
    assert np.isclose(pose1.x + 1, pose3.x)
    assert np.isclose(pose1.y, pose3.y)
    assert np.isclose(pose1.theta(), pose3.theta())
