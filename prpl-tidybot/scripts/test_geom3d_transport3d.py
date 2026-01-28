"""Tests for real world base-arm motion planning."""

import math
import time

import numpy as np
import prbench
from gymnasium.wrappers import RecordVideo
from prbench_models.geom3d.transport3d.parameterized_skills import (
    create_lifted_controllers,
)
from relational_structs.spaces import ObjectCentricBoxSpace
from spatialmath import SE2

from prbench.envs.geom3d.transport3d import ObjectCentricTransport3DEnv

from prpl_tidybot.base_movement import reach_target_pose
from prpl_tidybot.constants import POLICY_CONTROL_PERIOD
from prpl_tidybot.coord_converter import CoordFrameConverter
from prpl_tidybot.interfaces.interface import RealInterface
from prpl_tidybot.perceivers.prbench_ground_perceiver import PRBenchTransport3DPerceiver
from prpl_tidybot.structs import TidyBotAction

prbench.register_all_environments()


def real2sim() -> None:
    """Test move-base-arm to the target object in ground environment with 1 cube."""

    try:
        # Create the environment.
        num_cubes = 1
        env = prbench.make(
            f"prbench/Transport3D-o{num_cubes}-v0", render_mode="rgb_array"
        )

        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"Transport3D-o{num_cubes}"
        )

        # Reset the environment and get the initial state.
        obs, _ = env.reset(seed=125)  # type: ignore
        assert isinstance(env.observation_space, ObjectCentricBoxSpace)

        ### real interface
        interface = RealInterface()

        # initialization
        pose_map = SE2(0, 0, 0)
        pose_odom = SE2(0, 0, 0)
        map_to_odom_converter = CoordFrameConverter(pose_map, pose_odom)
        odom_to_map_converter = CoordFrameConverter(pose_odom, pose_map)

        # get initial pose
        observation = interface.get_observation()
        map_to_odom_converter.update(observation.map_base_pose, observation.base_pose)
        odom_to_map_converter.update(observation.base_pose, observation.map_base_pose)

        perceiver = PRBenchTransport3DPerceiver(interface)
        state = perceiver.get_state()
        env.unwrapped._object_centric_env.set_state(state)  # type: ignore # pylint: disable=protected-access

        sim = ObjectCentricTransport3DEnv(num_cubes=num_cubes)
        controllers = create_lifted_controllers(
            env.action_space,
            sim,
        )
        lifted_controller = controllers["pick"]
        robot = state.get_object_from_name("robot")
        target = state.get_object_from_name("cube0")
        object_parameters = (robot, target)
        controller = lifted_controller.ground(object_parameters)

        params = np.array([0.5, 0.0])

        controller.reset(state, params)
        for _ in range(500):
            action = controller.step()
            obs, _, _, _, _ = env.step(action)
            next_state = env.observation_space.devectorize(obs)
            controller.observe(next_state)
            state = next_state
            if controller.terminated():
                break
        else:
            assert False, "Controller did not terminate"

        lifted_controller = controllers["place"]
        robot = state.get_object_from_name("robot")
        target = state.get_object_from_name("cube0")
        target_box = state.get_object_from_name("box0")
        object_parameters = (robot, target, target_box)
        controller = lifted_controller.ground(object_parameters)

        params = np.array([0.0, -0.06])

        controller.reset(state, params)
        for _ in range(500):
            action = controller.step()
            obs, _, _, _, _ = env.step(action)
            next_state = env.observation_space.devectorize(obs)
            controller.observe(next_state)
            state = next_state
            if controller.terminated():
                break
        else:
            assert False, "Controller did not terminate"

    finally:
        env.close()  # type: ignore
        interface.close()

if __name__ == "__main__":
    real2sim()