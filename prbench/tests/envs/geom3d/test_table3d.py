"""Tests for ground3d.py."""

import numpy as np
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from prpl_utils.utils import wrap_angle
from pybullet_helpers.geometry import Pose, SE2Pose
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    remap_joint_position_plan_to_constant_distance,
    run_single_arm_mobile_base_motion_planning,
    smoothly_follow_end_effector_path,
)
from relational_structs.spaces import ObjectCentricBoxSpace

from prbench.envs.geom3d.table3d import (
    Table3DEnv,
    Table3DObjectCentricState,
    ObjectCentricTable3DEnv,
)


def test_base_table3d_env():
    """Tests for basic methods in base table3D env."""

    env = Table3DEnv(use_gui=True)  # set use_gui=True to debug
    obs, _ = env.reset(seed=123)
    assert isinstance(obs, np.ndarray)

    for _ in range(10):
        act = env.action_space.sample()
        assert isinstance(act, np.ndarray)
        obs, _, _, _, _ = env.step(act)

    # Uncomment to debug.
    import pybullet as p

    while True:
        p.getMouseEvents(env._object_centric_env.physics_client_id)

