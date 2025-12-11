"""Tests for ground3d.py."""

import numpy as np
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from relational_structs.spaces import ObjectCentricBoxSpace

from prbench.envs.geom3d.ground3d import (
    Ground3DEnv,
    Ground3DObjectCentricState,
    ObjectCentricGround3DEnv,
)


def test_base_motion3d_env():
    """Tests for basic methods in base motion3D env."""

    env = Ground3DEnv(use_gui=False)  # set use_gui=True to debug
    obs, _ = env.reset(seed=123)
    assert isinstance(obs, np.ndarray)

    for _ in range(10):
        act = env.action_space.sample()
        assert isinstance(act, np.ndarray)
        obs, _, _, _, _ = env.step(act)

    # Uncomment to debug.
    # import pybullet as p

    # while True:
    #     p.getMouseEvents(env._object_centric_env.physics_client_id)
