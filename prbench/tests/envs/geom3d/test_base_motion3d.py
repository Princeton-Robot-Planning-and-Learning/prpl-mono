"""Tests for base_motion3d.py."""

import numpy as np
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from pybullet_helpers.motion_planning import (
    run_single_arm_mobile_base_motion_planning,
)
from relational_structs.spaces import ObjectCentricBoxSpace

from prbench.envs.geom3d.base_motion3d import (
    BaseMotion3DEnv,
    BaseMotion3DObjectCentricState,
    ObjectCentricBaseMotion3DEnv,
)


def test_base_motion3d_env():
    """Tests for basic methods in base motion3D env."""

    env = BaseMotion3DEnv(use_gui=False)  # set use_gui=True to debug
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


def test_motion_planning_in_base_motion3d_env():
    """Proof of concept that motion planning works in this environment."""

    # Create the real environment.
    env = BaseMotion3DEnv(render_mode="rgb_array")
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    config = env._object_centric_env.config  # pylint: disable=protected-access
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    vec_obs, _ = env.reset(seed=123)
    # NOTE: we should soon make this smoother.
    oc_obs = env.observation_space.devectorize(vec_obs)
    obs = BaseMotion3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Create a simulator for planning.
    sim = ObjectCentricBaseMotion3DEnv(config=config)

    base_plan = run_single_arm_mobile_base_motion_planning(
        sim.robot,
        sim.robot.base.get_pose(),
        obs.target_base_pose,
        collision_bodies=set(),
        seed=123,
    )
    assert base_plan is not None

    env.action_space.seed(123)
    for target_base_pose in base_plan[1:]:
        current_base_pose = obs.base_pose
        delta = target_base_pose - current_base_pose
        delta_lst = [delta.x, delta.y, delta.rot]
        action_lst = delta_lst + [0.0] * 7 + [0.0]
        action = np.array(action_lst, dtype=np.float32)
        vec_obs, _, done, _, _ = env.step(action)
        # NOTE: we should soon make this smoother.
        oc_obs = env.observation_space.devectorize(vec_obs)
        obs = BaseMotion3DObjectCentricState(oc_obs.data, oc_obs.type_features)
        if done:
            break
    else:
        assert False, "Plan did not reach goal"
    env.close()
