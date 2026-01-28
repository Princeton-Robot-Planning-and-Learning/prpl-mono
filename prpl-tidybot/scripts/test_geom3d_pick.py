"""Tests for real world base-arm motion planning."""

import math
import time

import numpy as np
import prbench
from gymnasium.wrappers import RecordVideo
from prbench_models.geom3d.ground3d.parameterized_skills import (
    create_lifted_controllers,
)
from relational_structs.spaces import ObjectCentricBoxSpace
from spatialmath import SE2

from prbench.envs.geom3d.ground3d import ObjectCentricGround3DEnv

from prpl_tidybot.base_movement import reach_target_pose
from prpl_tidybot.constants import POLICY_CONTROL_PERIOD
from prpl_tidybot.coord_converter import CoordFrameConverter
from prpl_tidybot.interfaces.interface import RealInterface
from prpl_tidybot.perceivers.prbench_ground_perceiver import PRBenchGeom3DPerceiver
from prpl_tidybot.structs import TidyBotAction

prbench.register_all_environments()


def real2sim() -> None:
    """Test move-base-arm to the target object in ground environment with 1 cube."""

    try:
        # Create the environment.
        num_cubes = 1
        env = prbench.make(
            f"prbench/Ground3D-o{num_cubes}-v0", render_mode="rgb_array"
        )

        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"Ground3D-o{num_cubes}"
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

        perceiver = PRBenchGeom3DPerceiver(interface)
        state = perceiver.get_state()
        env.unwrapped._object_centric_env.set_state(state)  # type: ignore # pylint: disable=protected-access

        sim = ObjectCentricGround3DEnv(num_cubes=num_cubes)
        controllers = create_lifted_controllers(
            env.action_space,
            sim,
        )
        lifted_controller = controllers["pick"]
        robot = state.get_object_from_name("robot")
        target = state.get_object_from_name("cube0")
        object_parameters = (robot, target)
        controller = lifted_controller.ground(object_parameters)

        rng = np.random.default_rng(123)
        params = controller.sample_parameters(state, rng)

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
        object_parameters = (robot, target)
        controller = lifted_controller.ground(object_parameters)

        rng = np.random.default_rng(123)
        params = controller.sample_parameters(state, rng)

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

def real2sim2real() -> None:
    """Test move-base-arm to the target object in ground environment with 1 cube."""

    try:
        # Create the environment.
        num_cubes = 1
        env = prbench.make(
            f"prbench/Ground3D-o{num_cubes}-v0", render_mode="rgb_array"
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

        perceiver = PRBenchGeom3DPerceiver(interface)
        state = perceiver.get_state()
        env.unwrapped._object_centric_env.set_state(state)  # type: ignore # pylint: disable=protected-access

        sim = ObjectCentricGround3DEnv(num_cubes=num_cubes)
        controllers = create_lifted_controllers(
            env.action_space,
            sim,
        )
        lifted_controller = controllers["pick"]
        robot = state.get_object_from_name("robot")
        target = state.get_object_from_name("cube0")
        object_parameters = (robot, target)
        controller = lifted_controller.ground(object_parameters)

        rng = np.random.default_rng(123)
        params = controller.sample_parameters(state, rng)

        controller.reset(state, params)

        controller.get_base_motion_plan()
        
        # real execution
        for t in range(
            1, len(controller._current_plan)  # type: ignore  # pylint: disable=protected-access
        ):
            pose_temp = controller._current_plan[  # type: ignore # pylint: disable=protected-access
                t
            ]
            pose = SE2(pose_temp.x, pose_temp.y, pose_temp.rot)
            print(f"Target pose: {pose.x}, {pose.y}, {pose.theta()}")
            if (
                t != len(controller._current_plan) - 1  # type: ignore # pylint: disable=protected-access
            ):
                reach_target_pose(
                    interface,
                    pose,
                    map_to_odom_converter,
                    odom_to_map_converter,
                    tolerance=0.05,
                )
            else:
                reach_target_pose(
                    interface, pose, map_to_odom_converter, odom_to_map_converter
                )
            time.sleep(0.1)

        # get new observation
        observation = interface.get_observation()
        map_to_odom_converter.update(observation.map_base_pose, observation.base_pose)
        odom_to_map_converter.update(observation.base_pose, observation.map_base_pose)

        perceiver = PRBenchGeom3DPerceiver(interface)
        state = perceiver.get_state()

        controller.observe(state)
        controller.get_arm_motion_plan(real=True)

        # real execution
        for t in range(
            1, len(controller._current_arm_joint_plan)  # type: ignore # pylint: disable=protected-access
        ):
            tidybot_action = TidyBotAction(
                base_local_goal=interface.get_base_state(),
                arm_goal=controller._current_arm_joint_plan[t][  # type: ignore # pylint: disable=protected-access
                    :7
                ],
                gripper_goal=interface.get_gripper_state(),
            )
            interface.execute_arm_action(tidybot_action)
            time.sleep(POLICY_CONTROL_PERIOD)

        time.sleep(POLICY_CONTROL_PERIOD)

        # close the gripper
        for _ in range(5):
            tidybot_action = TidyBotAction(
                base_local_goal=interface.get_base_state(),
                arm_goal=interface.get_arm_state(),
                gripper_goal=1.0,
            )
            interface.execute_gripper_action(tidybot_action)
            time.sleep(POLICY_CONTROL_PERIOD)

        # get new observation
        observation = interface.get_observation()
        map_to_odom_converter.update(observation.map_base_pose, observation.base_pose)
        odom_to_map_converter.update(observation.base_pose, observation.map_base_pose)

        perceiver = PRBenchGeom3DPerceiver(interface)
        state = perceiver.get_state()
        controller.observe(state)
        controller.get_retract_motion_plan(real=True)

        # real execution
        for t in range(
            1, len(controller._current_retract_plan)  # type: ignore # pylint: disable=protected-access
        ):
            tidybot_action = TidyBotAction(
                base_local_goal=interface.get_base_state(),
                arm_goal=controller._current_retract_plan[t][  # type: ignore # pylint: disable=protected-access
                    :7
                ],
                gripper_goal=interface.get_gripper_state(),
            )
            interface.execute_arm_action(tidybot_action)
            time.sleep(POLICY_CONTROL_PERIOD)

        time.sleep(POLICY_CONTROL_PERIOD)

        lifted_controller = controllers["place"]
        robot = state.get_object_from_name("robot")
        target = state.get_object_from_name("cube0")
        object_parameters = (robot, target)
        controller = lifted_controller.ground(object_parameters)

        rng = np.random.default_rng(123)
        params = controller.sample_parameters(state, rng)

        controller.reset(state, params)

        controller.get_base_motion_plan(real=True)
        
        # real execution
        for t in range(
            1, len(controller._current_plan)  # type: ignore  # pylint: disable=protected-access
        ):
            pose_temp = controller._current_plan[  # type: ignore # pylint: disable=protected-access
                t
            ]
            pose = SE2(pose_temp.x, pose_temp.y, pose_temp.rot)
            print(f"Target pose: {pose.x}, {pose.y}, {pose.theta()}")
            if (
                t != len(controller._current_plan) - 1  # type: ignore # pylint: disable=protected-access
            ):
                reach_target_pose(
                    interface,
                    pose,
                    map_to_odom_converter,
                    odom_to_map_converter,
                    tolerance=0.05,
                )
            else:
                reach_target_pose(
                    interface, pose, map_to_odom_converter, odom_to_map_converter
                )
            time.sleep(0.1)

        # get new observation
        observation = interface.get_observation()
        map_to_odom_converter.update(observation.map_base_pose, observation.base_pose)
        odom_to_map_converter.update(observation.base_pose, observation.map_base_pose)

        perceiver = PRBenchGeom3DPerceiver(interface)
        state = perceiver.get_state()

        controller.observe(state)
        controller.get_arm_motion_plan(real=True)

        # real execution
        for t in range(
            1, len(controller._current_arm_joint_plan)  # type: ignore # pylint: disable=protected-access
        ):
            tidybot_action = TidyBotAction(
                base_local_goal=interface.get_base_state(),
                arm_goal=controller._current_arm_joint_plan[t][  # type: ignore # pylint: disable=protected-access
                    :7
                ],
                gripper_goal=interface.get_gripper_state(),
            )
            interface.execute_arm_action(tidybot_action)
            time.sleep(POLICY_CONTROL_PERIOD)

        time.sleep(POLICY_CONTROL_PERIOD)

        # open the gripper
        for _ in range(5):
            tidybot_action = TidyBotAction(
                base_local_goal=interface.get_base_state(),
                arm_goal=interface.get_arm_state(),
                gripper_goal=0.0,
            )
            interface.execute_gripper_action(tidybot_action)
            time.sleep(POLICY_CONTROL_PERIOD)
        
        controller.get_retract_motion_plan()

        # real execution
        for t in range(
            1, len(controller._current_retract_plan)  # type: ignore # pylint: disable=protected-access
        ):
            tidybot_action = TidyBotAction(
                base_local_goal=interface.get_base_state(),
                arm_goal=controller._current_retract_plan[t][  # type: ignore # pylint: disable=protected-access
                    :7
                ],
                gripper_goal=interface.get_gripper_state(),
            )
            interface.execute_arm_action(tidybot_action)
            time.sleep(POLICY_CONTROL_PERIOD)
        
        time.sleep(POLICY_CONTROL_PERIOD)


    finally:
        env.close()  # type: ignore
        interface.close()


if __name__ == "__main__":
    # real2sim()
    real2sim2real()