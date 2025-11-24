"""Tests for real world base-arm motion planning."""

import math
import time

import numpy as np
import prbench
from gymnasium.wrappers import RecordVideo
from prbench_models.dynamic3d.ground.parameterized_skills import (
    create_lifted_controllers,
)
from relational_structs.spaces import ObjectCentricBoxSpace
from spatialmath import SE2

from prpl_tidybot.base_movement import reach_target_pose
from prpl_tidybot.constants import POLICY_CONTROL_PERIOD
from prpl_tidybot.coord_converter import CoordFrameConverter
from prpl_tidybot.interfaces.interface import RealInterface
from prpl_tidybot.perceivers.prbench_ground_perceiver import PRBenchGroundPerceiver
from prpl_tidybot.structs import TidyBotAction

prbench.register_all_environments()


def real2sim() -> None:
    """Test move-base-arm to the target object in ground environment with 1 cube."""

    try:
        # Create the environment.
        num_cubes = 1
        env = prbench.make(
            f"prbench/TidyBot3D-cupboard-o{num_cubes}-v0", render_mode="rgb_array"
        )

        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"TidyBot3D-cupboard-o{num_cubes}"
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

        perceiver = PRBenchGroundPerceiver(interface)
        state = perceiver.get_state()
        env.unwrapped._object_centric_env.set_state(state)  # type: ignore # pylint: disable=protected-access

        # Create the move-base controller.
        controllers = create_lifted_controllers(env.action_space)  # type: ignore
        lifted_controller = controllers["move_to_target"]
        robot = state.get_object_from_name("robot")
        cube = state.get_object_from_name("cube1")
        object_parameters = (robot, cube)
        controller = lifted_controller.ground(object_parameters)
        target_distance = 0.6
        target_rotation = 0
        params = np.array([target_distance, target_rotation])

        # Reset and execute the controller until it terminates.
        controller.reset(state, params)
        for _ in range(200):
            action = controller.step()
            obs, _, _, _, _ = env.step(action)  # type: ignore
            next_state = env.observation_space.devectorize(obs)
            controller.observe(next_state)
            state = next_state
            if controller.terminated():
                break
        else:
            assert False, "Controller did not terminate"

        # create the move-arm controller.
        lifted_controller = controllers["move_arm_to_end_effector"]
        robot = state.get_object_from_name("robot")
        object_parameters = (robot,)  # type: ignore
        controller = lifted_controller.ground(object_parameters)
        target_end_effector_pose = np.array(
            [
                0.50,
                0.0,
                -0.35,
                0.707,
                0.707,
                0,
                0,
            ]
        )  # x, y, z, rw, rx, ry, rz
        params = target_end_effector_pose

        # Reset and execute the controller until it terminates.
        controller.reset(state, params)
        for _ in range(200):
            action = controller.step()
            obs, _, _, _, _ = env.step(action)  # type: ignore
            next_state = env.observation_space.devectorize(obs)
            controller.observe(next_state)
            state = next_state
            if controller.terminated():
                break
        else:
            assert False, "Controller did not terminate"

        # Create the controller.
        controllers = create_lifted_controllers(env.action_space)  # type: ignore
        lifted_controller = controllers["close_gripper"]
        robot = state.get_object_from_name("robot")
        object_parameters = (robot,)  # type: ignore
        controller = lifted_controller.ground(object_parameters)

        # Reset and execute the controller until it terminates.
        controller.reset(state)  # type: ignore
        for _ in range(20):
            action = controller.step()
            obs, _, _, _, _ = env.step(action)  # type: ignore
            next_state = env.observation_space.devectorize(obs)
            controller.observe(next_state)
            state = next_state
            if controller.terminated():
                break
        else:
            assert False, "Controller did not terminate"

        # move the arm to the target configuration
        lifted_controller = controllers["move_arm_to_conf"]
        robot = state.get_object_from_name("robot")
        object_parameters = (robot,)  # type: ignore
        controller = lifted_controller.ground(object_parameters)
        target_conf = np.deg2rad(
            [0, -20, 180, -146, 0, -50, 90]
        )  # retract configuration
        params = target_conf

        # Reset and execute the controller until it terminates.
        controller.reset(state, params)
        for _ in range(200):
            action = controller.step()
            obs, _, _, _, _ = env.step(action)  # type: ignore
            next_state = env.observation_space.devectorize(obs)
            controller.observe(next_state)
            state = next_state
            if controller.terminated():
                break
        else:
            assert False, "Controller did not terminate"

        # Create the move-base controller.
        controllers = create_lifted_controllers(env.action_space)  # type: ignore
        lifted_controller = controllers["move_to_target"]
        robot = state.get_object_from_name("robot")
        cube = state.get_object_from_name("cupboard_1")
        object_parameters = (robot, cube)
        controller = lifted_controller.ground(object_parameters)
        target_distance = 0.9
        target_rotation = -np.pi / 2  # type: ignore
        params = np.array([target_distance, target_rotation])

        # Reset and execute the controller until it terminates.
        controller.reset(state, params, disable_collision_objects=["cube1"])  # type: ignore # pylint: disable=line-too-long
        for _ in range(200):
            action = controller.step()
            obs, _, _, _, _ = env.step(action)  # type: ignore
            next_state = env.observation_space.devectorize(obs)
            controller.observe(next_state)
            state = next_state
            if controller.terminated():
                break
        else:
            assert False, "Controller did not terminate"

        # create the move-arm controller.
        lifted_controller = controllers["move_arm_to_end_effector"]
        robot = state.get_object_from_name("robot")
        object_parameters = (robot,)  # type: ignore
        controller = lifted_controller.ground(object_parameters)
        target_end_effector_pose = np.array(
            [
                0.7,
                0.0,
                0.08,
                0.5,
                0.5,
                0.5,
                0.5,
            ]
        )  # x, y, z, rw, rx, ry, rz
        params = target_end_effector_pose

        # Reset and execute the controller until it terminates.
        controller.reset(state, params)
        for _ in range(200):
            action = controller.step()
            obs, _, _, _, _ = env.step(action)  # type: ignore
            next_state = env.observation_space.devectorize(obs)
            controller.observe(next_state)
            state = next_state
            if controller.terminated():
                break
        else:
            assert False, "Controller did not terminate"

        # Create the controller.
        controllers = create_lifted_controllers(env.action_space)  # type: ignore
        lifted_controller = controllers["open_gripper"]
        robot = state.get_object_from_name("robot")
        object_parameters = (robot,)  # type: ignore
        controller = lifted_controller.ground(object_parameters)

        # Reset and execute the controller until it terminates.
        controller.reset(state)  # type: ignore
        for _ in range(20):
            action = controller.step()
            obs, _, _, _, _ = env.step(action)  # type: ignore
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
    """Real to sim to real."""

    try:
        # Create the environment.
        num_cubes = 1
        env = prbench.make(
            f"prbench/TidyBot3D-cupboard-o{num_cubes}-v0", render_mode="rgb_array"
        )

        # Reset the environment and get the initial state.
        _, _ = env.reset(seed=125)
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

        perceiver = PRBenchGroundPerceiver(interface)
        temp_state = perceiver.get_state()
        env.unwrapped._object_centric_env.set_state(temp_state)  # type: ignore # pylint: disable=protected-access
        state = env.unwrapped._object_centric_env._get_object_centric_state()  # type: ignore # pylint: disable=protected-access

        # Create the move-base controller.
        controllers = create_lifted_controllers(env.action_space)  # type: ignore
        lifted_controller = controllers["move_to_target"]
        robot = state.get_object_from_name("robot")
        cube = state.get_object_from_name("cube1")
        object_parameters = (robot, cube)
        controller = lifted_controller.ground(object_parameters)
        target_distance = 0.6
        target_rotation = 0
        params = np.array([target_distance, target_rotation])

        # Reset and execute the controller until it terminates.
        controller.reset(  # type: ignore
            state, params, extend_xy_magnitude=0.2, extend_rot_magnitude=math.pi / 2
        )

        # real execution
        for t in range(
            1, len(controller._current_base_motion_plan)  # type: ignore  # pylint: disable=protected-access
        ):
            pose = controller._current_base_motion_plan[  # type: ignore # pylint: disable=protected-access
                t
            ]
            print(f"Target pose: {pose.x}, {pose.y}, {pose.theta()}")
            if (
                t != len(controller._current_base_motion_plan) - 1  # type: ignore # pylint: disable=protected-access
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

        perceiver = PRBenchGroundPerceiver(interface)
        temp_state = perceiver.get_state()
        env.unwrapped._object_centric_env.set_state(temp_state)  # type: ignore # pylint: disable=protected-access
        state = env.unwrapped._object_centric_env._get_object_centric_state()  # type: ignore # pylint: disable=protected-access

        # create the move-arm controller.
        lifted_controller = controllers["move_arm_to_end_effector"]
        robot = state.get_object_from_name("robot")
        object_parameters = (robot,)  # type: ignore
        controller = lifted_controller.ground(object_parameters)
        target_end_effector_pose = np.array(
            [
                0.50,
                0.0,
                -0.35,
                0.707,
                0.707,
                0,
                0,
            ]
        )  # x, y, z, rw, rx, ry, rz
        params = target_end_effector_pose

        # Reset and execute the controller until it terminates.
        controller.reset(state, params)

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

        perceiver = PRBenchGroundPerceiver(interface)
        temp_state = perceiver.get_state()
        env.unwrapped._object_centric_env.set_state(temp_state)  # type: ignore # pylint: disable=protected-access
        state = env.unwrapped._object_centric_env._get_object_centric_state()  # type: ignore # pylint: disable=protected-access

        # move the arm to the target configuration
        lifted_controller = controllers["move_arm_to_conf"]
        robot = state.get_object_from_name("robot")
        object_parameters = (robot,)  # type: ignore
        controller = lifted_controller.ground(object_parameters)
        target_conf = np.deg2rad(
            [0, -20, 180, -146, 0, -50, 90]
        )  # retract configuration
        params = target_conf

        # Reset and execute the controller until it terminates.
        controller.reset(state, params)

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

        # get new observation
        observation = interface.get_observation()
        map_to_odom_converter.update(observation.map_base_pose, observation.base_pose)
        odom_to_map_converter.update(observation.base_pose, observation.map_base_pose)

        perceiver = PRBenchGroundPerceiver(interface)
        temp_state = perceiver.get_state()
        env.unwrapped._object_centric_env.set_state(temp_state)  # type: ignore # pylint: disable=protected-access
        state = env.unwrapped._object_centric_env._get_object_centric_state()  # type: ignore # pylint: disable=protected-access

        # Create the move-base controller.
        controllers = create_lifted_controllers(env.action_space)  # type: ignore
        lifted_controller = controllers["move_to_target"]
        robot = state.get_object_from_name("robot")
        cube = state.get_object_from_name("cupboard_1")  # type: ignore # pylint: disable=protected-access
        object_parameters = (robot, cube)
        controller = lifted_controller.ground(object_parameters)
        target_distance = 0.9
        target_rotation = -np.pi / 2  # type: ignore
        params = np.array([target_distance, target_rotation])

        # Reset and execute the controller until it terminates.
        controller.reset(  # type: ignore
            state,
            params,
            extend_xy_magnitude=0.2,
            extend_rot_magnitude=math.pi / 2,
            disable_collision_objects=["cube1"],
        )

        # real execution
        for t in range(
            1, len(controller._current_base_motion_plan)  # type: ignore  # pylint: disable=protected-access
        ):
            pose = controller._current_base_motion_plan[  # type: ignore # pylint: disable=protected-access
                t
            ]
            print(f"Target pose: {pose.x}, {pose.y}, {pose.theta()}")
            if (
                t != len(controller._current_base_motion_plan) - 1  # type: ignore # pylint: disable=protected-access
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

        perceiver = PRBenchGroundPerceiver(interface)
        temp_state = perceiver.get_state()
        env.unwrapped._object_centric_env.set_state(temp_state)  # type: ignore # pylint: disable=protected-access
        state = env.unwrapped._object_centric_env._get_object_centric_state()  # type: ignore # pylint: disable=protected-access

        # create the move-arm controller.
        lifted_controller = controllers["move_arm_to_end_effector"]
        robot = state.get_object_from_name("robot")
        object_parameters = (robot,)  # type: ignore
        controller = lifted_controller.ground(object_parameters)
        target_end_effector_pose = np.array(
            [
                0.7,
                0.0,
                0.08,
                0.5,
                0.5,
                0.5,
                0.5,
            ]
        )  # x, y, z, rw, rx, ry, rz
        params = target_end_effector_pose

        # Reset and execute the controller until it terminates.
        controller.reset(state, params)

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

    finally:
        env.close()  # type: ignore
        interface.close()


if __name__ == "__main__":
    # real2sim()
    real2sim2real()
