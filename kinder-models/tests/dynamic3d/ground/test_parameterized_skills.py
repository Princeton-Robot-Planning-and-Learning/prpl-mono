"""Tests for ground parameterized skills."""

import kinder
import numpy as np
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from kinder.envs.dynamic3d.object_types import MujocoTidyBotRobotObjectType
from prpl_tidybot.interfaces.interface import FakeInterface
from prpl_tidybot.perceivers.kinder_ground_perceiver import KinDERGroundPerceiver
from relational_structs import ObjectCentricState
from relational_structs.spaces import ObjectCentricBoxSpace
from spatialmath import SE2

from kinder_models.dynamic3d.ground.parameterized_skills import (
    PyBulletSim,
    create_lifted_controllers,
    get_target_robot_pose_from_parameters,
)

kinder.register_all_environments()


def _get_robot_from_state(state: ObjectCentricState):
    """Helper to get robot object from state by type."""
    robots = state.get_objects(MujocoTidyBotRobotObjectType)
    assert len(robots) == 1, f"Expected 1 robot, got {len(robots)}"
    return list(robots)[0]


def test_get_target_robot_pose_from_parameters():
    """Tests for get_target_robot_pose_from_parameters()."""

    target = SE2(1.0, 0.0, 0.0)
    robot_pose = get_target_robot_pose_from_parameters(
        target, target_distance=1.0, target_rot=0.0
    )

    # Robot should be 1m behind the target, facing it
    assert np.isclose(robot_pose.x, 0.0)
    assert np.isclose(robot_pose.y, 0.0)
    assert np.isclose(robot_pose.theta(), 0.0)

    # With a rotation offset of 90 degrees (pi/2)
    robot_pose2 = get_target_robot_pose_from_parameters(
        target, target_distance=1.0, target_rot=np.pi / 2
    )
    assert np.isclose(robot_pose2.x, 1.0)
    assert np.isclose(robot_pose2.y, -1.0)
    assert np.isclose(robot_pose2.theta(), np.pi / 2)

    # Uncomment to debug.
    # import imageio.v2 as iio
    # from matplotlib import pyplot as plt
    # from prpl_utils.utils import fig2data

    # from kinder_models.dynamic3d.utils import get_overhead_object_se2_pose, \
    #     plot_overhead_scene

    # env = kinder.make("kinder/TidyBot3D-ground-o1-v0", render_mode="rgb_array")
    # assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    # obs, _ = env.reset(seed=123)
    # state = env.observation_space.devectorize(obs)
    # fig, ax = plot_overhead_scene(state, min_x=-1.5, max_x=1.5, min_y=-1.5, max_y=1.5)

    # target_distance = 0.75
    # target_object = state.get_object_from_name("cube1")
    # for target_rot in np.linspace(-np.pi, np.pi, num=24):
    #     target_object_pose = get_overhead_object_se2_pose(state, target_object)
    #     robot_pose = get_target_robot_pose_from_parameters(
    #         target_object_pose, target_distance, target_rot
    #     )
    #     th = robot_pose.theta()
    #     ax.arrow(
    #         robot_pose.x, robot_pose.y, 0.1 * np.cos(th), 0.1 * np.sin(th), width=0.01
    #     )

    # ax.set_title("Examples for get_target_robot_pose_from_parameters().")
    # plt.tight_layout()
    # plt.axis("equal")
    # img = fig2data(fig)
    # outfile = "get_target_robot_pose_from_parameters.png"
    # iio.imsave(outfile, img)
    # print(f"Wrote out to {outfile}")


def test_move_to_target_controller_one_cube():
    """Test move-to-target controller in ground environment with 1 cube."""

    # Create the environment.
    num_cubes = 1
    env = kinder.make(
        f"kinder/TidyBot3D-ground-o{num_cubes}-v0", render_mode="rgb_array"
    )
    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"TidyBot3D-ground-o{num_cubes}"
        )

    # Reset the environment and get the initial state.
    obs, _ = env.reset(seed=123)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    # Create the controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["move_to_target"]
    robot = _get_robot_from_state(state)
    cube = state.get_object_from_name("cube1")
    object_parameters = (robot, cube)
    controller = lifted_controller.ground(object_parameters)
    target_distance = 0.5
    target_rotation = 0.0
    params = np.array([target_distance, target_rotation])

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(200):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    env.close()


def test_move_to_target_arm_configuration():
    """Test move-arm-to-conf controller in ground environment with 1 cube."""

    # Create the environment.
    num_cubes = 1
    env = kinder.make(
        f"kinder/TidyBot3D-ground-o{num_cubes}-v0", render_mode="rgb_array"
    )
    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"TidyBot3D-ground-o{num_cubes}"
        )

    # Reset the environment and get the initial state.
    obs, _ = env.reset(seed=124)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    # Create the controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["move_arm_to_conf"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)
    target_conf = np.zeros(7)
    params = target_conf

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(200):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    env.close()


def test_move_to_target_arm_end_effector():
    """Test move-arm-to-end-effector controller in ground environment with 1 cube."""

    # Create the environment.
    num_cubes = 1
    env = kinder.make(
        f"kinder/TidyBot3D-ground-o{num_cubes}-v0", render_mode="rgb_array"
    )
    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"TidyBot3D-ground-o{num_cubes}"
        )

    # Reset the environment and get the initial state.
    obs, _ = env.reset(seed=124)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    # Create the controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["move_arm_to_end_effector"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)
    relative_target_end_effector_pose = np.array(
        [
            0.5,
            0,
            -0.1,
            1,
            0,
            0,
            0,
            0.0,
        ]
    )  # x, y, z, rw, rx, ry, rz, yaw for relative rotation of target object
    params = relative_target_end_effector_pose

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(200):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    env.close()


def test_close_gripper_controller():
    """Test close-gripper controller in ground environment with 1 cube."""

    # Create the environment.
    num_cubes = 1
    env = kinder.make(
        f"kinder/TidyBot3D-ground-o{num_cubes}-v0", render_mode="rgb_array"
    )
    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"TidyBot3D-ground-o{num_cubes}"
        )

    # Reset the environment and get the initial state.
    obs, _ = env.reset(seed=125)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    # Create the controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["close_gripper"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)

    # Reset and execute the controller until it terminates.
    controller.reset(state)
    for _ in range(20):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # move the arm to the target configuration
    lifted_controller = controllers["move_arm_to_conf"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)
    target_conf = np.deg2rad([0, -20, 180, -146, 0, -50, 90])  # retract configuration
    params = target_conf

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(200):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # Create the move-base controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["move_to_target"]
    robot = _get_robot_from_state(state)
    cube = state.get_object_from_name("cube1")
    object_parameters = (robot, cube)
    controller = lifted_controller.ground(object_parameters)
    target_distance = 0.5
    target_rotation = np.pi / 2
    params = np.array([target_distance, target_rotation])

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(200):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # Create the controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["open_gripper"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)

    # Reset and execute the controller until it terminates.
    controller.reset(state)
    for _ in range(20):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    env.close()


def test_pick_place_ground():
    """Test pick and place in ground environment with 1 cube."""

    # Create the environment.
    num_cubes = 1
    env = kinder.make(
        f"kinder/TidyBot3D-ground-o{num_cubes}-v0",
        render_mode="rgb_array",
        allow_state_access=True,
    )
    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"TidyBot3D-ground-o{num_cubes}"
        )

    # Reset the environment and get the initial state.
    _, _ = env.reset(seed=125)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)

    interface = FakeInterface()
    interface.arm_interface.arm_state = np.deg2rad(
        [0, -20, 180, -146, 0, -50, 90]
    ).tolist()
    interface.arm_interface.gripper_state = 0.0
    interface.base_interface.map_base_state = SE2(x=0.8, y=0.0, theta=0.0)
    perceiver = KinDERGroundPerceiver(interface)
    temp_state = perceiver.get_state()
    env.unwrapped._object_centric_env.set_state(temp_state)  # type: ignore # pylint: disable=protected-access
    state = (
        env.unwrapped._object_centric_env._get_object_centric_state()  # pylint: disable=protected-access
    )

    # Create the move-base controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["move_to_target"]
    robot = _get_robot_from_state(state)
    cube = state.get_object_from_name("cube1")
    object_parameters = (robot, cube)
    controller = lifted_controller.ground(object_parameters)
    target_distance = 0.5
    target_rotation = np.pi
    params = np.array([target_distance, target_rotation])

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(200):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # create the move-arm controller.
    lifted_controller = controllers["move_arm_to_end_effector"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)
    target_end_effector_pose = np.array(
        [
            0.39,
            0.0,
            -0.35,
            0.707,
            0.707,
            0,
            0,
            0.0,
        ]
    )  # x, y, z, rw, rx, ry, rz, yaw for relative rotation of target object
    params = target_end_effector_pose

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(200):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # Create the controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["close_gripper"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)

    # Reset and execute the controller until it terminates.
    controller.reset(state)
    for _ in range(20):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # move the arm to the target configuration
    lifted_controller = controllers["move_arm_to_conf"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)
    target_conf = np.deg2rad([0, -20, 180, -146, 0, -50, 90])  # retract configuration
    params = target_conf

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(200):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # Create the move-base controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["move_to_target"]
    robot = _get_robot_from_state(state)
    cube = state.get_object_from_name("cube1")
    object_parameters = (robot, cube)
    controller = lifted_controller.ground(object_parameters)
    target_distance = 0.5
    target_rotation = np.pi / 2
    params = np.array([target_distance, target_rotation])

    # Reset and execute the controller until it terminates.
    controller.reset(state, params, disable_collision_objects=["cube1"])
    for _ in range(200):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # create the move-arm controller.
    lifted_controller = controllers["move_arm_to_end_effector"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)
    target_end_effector_pose = np.array(
        [
            0.40,
            0.0,
            -0.3,
            0.707,
            0.707,
            0,
            0,
            0.0,
        ]
    )  # x, y, z, rw, rx, ry, rz, yaw for relative rotation of target object
    params = target_end_effector_pose

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(200):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # Create the controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["open_gripper"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)

    # Reset and execute the controller until it terminates.
    controller.reset(state)
    for _ in range(20):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    env.close()


def test_pick_place_shelf():
    """Test fake interface in ground environment with 1 cube."""

    # Create the environment.
    num_cubes = 1
    env = kinder.make(
        f"kinder/TidyBot3D-cupboard_real-o{num_cubes}-v0",
        render_mode="rgb_array",
        allow_state_access=True,
    )
    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"TidyBot3D-cupboard-o{num_cubes}-real"
        )

    # Reset the environment and get the initial state.
    _, _ = env.reset(seed=125)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)

    interface = FakeInterface()
    interface.arm_interface.arm_state = np.deg2rad(
        [0, -20, 180, -146, 0, -50, 90]
    ).tolist()
    interface.arm_interface.gripper_state = 0.0
    interface.base_interface.map_base_state = SE2(x=-0.7, y=0.0, theta=0.0)
    perceiver = KinDERGroundPerceiver(interface)
    temp_state = perceiver.get_state()
    env.unwrapped._object_centric_env.set_state(temp_state)  # type: ignore # pylint: disable=protected-access
    state = (
        env.unwrapped._object_centric_env._get_object_centric_state()  # pylint: disable=protected-access
    )

    # Create the move-base controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["move_to_target"]
    robot = _get_robot_from_state(state)
    cube = state.get_object_from_name("cube1")
    object_parameters = (robot, cube)
    controller = lifted_controller.ground(object_parameters)
    target_distance = 0.5
    target_rotation = 0
    params = np.array([target_distance, target_rotation])

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(200):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # create the move-arm controller.
    lifted_controller = controllers["move_arm_to_end_effector"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)
    target_end_effector_pose = np.array(
        [
            0.40,
            0.0,
            -0.35,
            0.707,
            0.707,
            0,
            0,
            0.0,
        ]
    )  # x, y, z, rw, rx, ry, rz, yaw for relative rotation of target object
    params = target_end_effector_pose

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(200):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # Create the controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["close_gripper"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)

    # Reset and execute the controller until it terminates.
    controller.reset(state)
    for _ in range(20):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # move the arm to the target configuration
    lifted_controller = controllers["move_arm_to_conf"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)
    target_conf = np.deg2rad([0, -20, 180, -146, 0, -50, 90])  # retract configuration
    params = target_conf

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(200):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # Create the move-base controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["move_to_target"]
    robot = _get_robot_from_state(state)
    cube = state.get_object_from_name("cupboard_1")
    object_parameters = (robot, cube)
    controller = lifted_controller.ground(object_parameters)
    target_distance = 0.9
    target_rotation = -np.pi / 2
    params = np.array([target_distance, target_rotation])

    # Reset and execute the controller until it terminates.
    controller.reset(state, params, disable_collision_objects=["cube1"])
    for _ in range(200):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # create the move-arm controller.
    lifted_controller = controllers["move_arm_to_end_effector"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)
    target_end_effector_pose = np.array(
        [
            0.7,
            0.0,
            0.0,
            0.5,
            0.5,
            0.5,
            0.5,
            0.0,
        ]
    )  # x, y, z, rw, rx, ry, rz, yaw for relative rotation of target object
    params = target_end_effector_pose

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(200):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # Create the controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["open_gripper"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)

    # Reset and execute the controller until it terminates.
    controller.reset(state)
    for _ in range(20):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    env.close()


def test_pick_place_skill():
    """Test pick and place skill in ground environment with 1 cube."""

    # Create the environment.
    num_cubes = 1
    env = kinder.make(
        f"kinder/TidyBot3D-cupboard_real-o{num_cubes}-v0", render_mode="rgb_array"
    )
    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"TidyBot3D-cupboard-o{num_cubes}-real"
        )

    # Reset the environment and get the initial state.
    obs, _ = env.reset(seed=123)
    for _ in range(5):
        obs, _, _, _, _ = env.step(np.zeros(11))
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    assert state is not None
    pybullet_sim = PyBulletSim(state, rendering=False)

    controllers = create_lifted_controllers(env.action_space, pybullet_sim=pybullet_sim)

    # create the pick ground controller.
    lifted_controller = controllers["pick_ground"]
    robot = _get_robot_from_state(state)
    cube = state.get_object_from_name("cube1")
    object_parameters = (robot, cube)
    controller = lifted_controller.ground(object_parameters)
    # params = controller.sample_parameters(state, np.random.default_rng(123))
    params = np.array([0.6, 0.0])

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(400):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # create the place ground controller.
    lifted_controller = controllers["place_ground"]
    robot = _get_robot_from_state(state)
    cube = state.get_object_from_name("cube1")
    cupboard = state.get_object_from_name("cupboard_1")
    object_parameters = (robot, cube, cupboard)
    controller = lifted_controller.ground(object_parameters)
    params = controller.sample_parameters(state, np.random.default_rng(123))

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(400):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    env.close()


def test_pick_place_two_cubes_skill():
    """Test pick and place skill in ground environment with 1 cube."""

    # Create the environment.
    num_cubes = 2
    env = kinder.make(
        f"kinder/TidyBot3D-cupboard_real-o{num_cubes}-v0", render_mode="rgb_array"
    )
    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"TidyBot3D-cupboard-o{num_cubes}-real"
        )

    # Reset the environment and get the initial state.
    obs, _ = env.reset(seed=123)
    for _ in range(5):
        obs, _, _, _, _ = env.step(np.zeros(11))
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    assert state is not None
    pybullet_sim = PyBulletSim(state, rendering=False)

    # Create the move-base controller.
    controllers = create_lifted_controllers(env.action_space, pybullet_sim=pybullet_sim)
    # create the pick ground controller.
    lifted_controller = controllers["pick_ground"]
    robot = _get_robot_from_state(state)
    cube = state.get_object_from_name("cube1")
    object_parameters = (robot, cube)
    controller = lifted_controller.ground(object_parameters)
    target_distance = 0.6
    target_rotation = 0.0
    params = np.array([target_distance, target_rotation])

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(400):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # create the place ground controller.
    lifted_controller = controllers["place_ground"]
    robot = _get_robot_from_state(state)
    cube = state.get_object_from_name("cube1")
    cupboard = state.get_object_from_name("cupboard_1")
    object_parameters = (robot, cube, cupboard)
    controller = lifted_controller.ground(object_parameters)
    target_distance = 0.85
    offset = 0.0
    target_rotation = -np.pi / 2
    params = np.array([target_distance, offset, target_rotation])

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(800):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # create the pick ground controller.
    lifted_controller = controllers["pick_ground"]
    robot = _get_robot_from_state(state)
    cube = state.get_object_from_name("cube2")
    object_parameters = (robot, cube)
    controller = lifted_controller.ground(object_parameters)
    target_distance = 0.6
    target_rotation = 0.0
    params = np.array([target_distance, target_rotation])

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(400):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # create the place ground controller.
    lifted_controller = controllers["place_ground"]
    robot = _get_robot_from_state(state)
    cube = state.get_object_from_name("cube2")
    cupboard = state.get_object_from_name("cupboard_1")
    object_parameters = (robot, cube, cupboard)
    controller = lifted_controller.ground(object_parameters)
    target_distance = 0.92
    offset = 0.0
    target_rotation = -np.pi / 2
    params = np.array([target_distance, offset, target_rotation])

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(400):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    env.close()


def test_velocity_tracking_mode():
    """Test pick and place in ground environment with 1 cube."""

    # Create the environment.
    num_cubes = 1
    env = kinder.make(
        f"kinder/TidyBot3D-ground-o{num_cubes}-v0",
        render_mode="rgb_array",
        allow_state_access=True,
    )
    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"TidyBot3D-ground-o{num_cubes}"
        )

    # Reset the environment and get the initial state.
    _, _ = env.reset(seed=125)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)

    interface = FakeInterface()
    interface.arm_interface.arm_state = np.deg2rad(
        [0, -20, 180, -146, 0, -50, 90]
    ).tolist()
    interface.arm_interface.gripper_state = 0.0
    interface.base_interface.map_base_state = SE2(x=0.8, y=0.0, theta=0.0)
    perceiver = KinDERGroundPerceiver(interface)
    temp_state = perceiver.get_state()
    env.unwrapped._object_centric_env.set_state(temp_state)  # type: ignore # pylint: disable=protected-access
    state = (
        env.unwrapped._object_centric_env._get_object_centric_state()  # pylint: disable=protected-access
    )

    # Create the move-base controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["move_to_target"]
    robot = _get_robot_from_state(state)
    cube = state.get_object_from_name("cube1")
    object_parameters = (robot, cube)
    controller = lifted_controller.ground(object_parameters)
    target_distance = 0.5
    target_rotation = np.pi
    params = np.array([target_distance, target_rotation])

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(200):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # create the move-arm controller.
    lifted_controller = controllers["move_arm_to_end_effector"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)
    target_end_effector_pose = np.array(
        [
            0.39,
            0.0,
            -0.35,
            0.707,
            0.707,
            0,
            0,
            0.0,
        ]
    )  # x, y, z, rw, rx, ry, rz, yaw for relative rotation of target object
    params = target_end_effector_pose

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(200):
        action = controller.step()
        action_18 = np.zeros(18)
        action_18[:10] = action[:10]
        action_18[10] = action[10]
        action_18[11:18] = 0.2 * action[3:10] * _ * np.ones(7)
        obs, _, _, _, _ = env.step(action_18)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    env.close()


def test_pick_toss():
    """Test pick and place in ground environment with 1 cube."""

    # Create the environment.
    num_cubes = 1
    env = kinder.make(
        f"kinder/Tossing3D-o{num_cubes}-v0",
        render_mode="rgb_array",
        scene_bg=False,
    )
    if MAKE_VIDEOS:
        env.unwrapped._object_centric_env.set_render_camera("task_view")  # type: ignore # pylint: disable=protected-access
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"TidyBot3D-ground-o{num_cubes}"
        )

    # Reset the environment and get the initial state.
    obs, _ = env.reset(seed=125)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)

    state = env.observation_space.devectorize(obs)

    # Create the move-base controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["move_to_target"]
    robot = _get_robot_from_state(state)
    cube = state.get_object_from_name("cube_0")
    object_parameters = (robot, cube)
    controller = lifted_controller.ground(object_parameters)
    target_distance = 0.5
    target_rotation = 0
    params = np.array([target_distance, target_rotation])

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(200):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # create the move-arm controller.
    lifted_controller = controllers["move_arm_to_end_effector"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)
    target_end_effector_pose = np.array(
        [
            0.39,
            0.0,
            -0.35,
            0.707,
            0.707,
            0,
            0,
            0.0,
        ]
    )  # x, y, z, rw, rx, ry, rz, yaw for relative rotation of target object
    params = target_end_effector_pose

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(200):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # Create the controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["close_gripper"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)

    # Reset and execute the controller until it terminates.
    controller.reset(state)
    for _ in range(20):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # move the arm to the target configuration
    lifted_controller = controllers["move_arm_to_conf"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)
    target_conf = np.deg2rad([0, -20, 180, -146, 0, -50, 90])  # retract configuration
    params = target_conf

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(200):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # Create the move-base controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["move_to_target"]
    robot = _get_robot_from_state(state)
    cube = state.get_object_from_name("bin_0")
    object_parameters = (robot, cube)
    controller = lifted_controller.ground(object_parameters)
    target_distance = 1.35
    target_rotation = 0.0
    params = np.array([target_distance, target_rotation])

    # Reset and execute the controller until it terminates.
    controller.reset(state, params, disable_collision_objects=["cube_0"])
    for _ in range(200):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # move the arm to the target configuration
    lifted_controller = controllers["move_arm_to_conf"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)
    target_conf = np.deg2rad([0, 50, 180, -110, 0, -100, 90])  # pre toss
    params = target_conf

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(200):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # move the arm to the target configuration
    lifted_controller = controllers["toss"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)
    target_conf = np.deg2rad([0, 20, 180, -35, 0, 25, 90])  # toss
    params = target_conf

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(200):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"
    cube_position = [state.get(cube, "x"), state.get(cube, "y"), state.get(cube, "z")]
    cube_orientation = [
        state.get(cube, "qx"),
        state.get(cube, "qy"),
        state.get(cube, "qz"),
        state.get(cube, "qw"),
    ]
    robot_base_position = [
        state.get(robot, "pos_base_x"),
        state.get(robot, "pos_base_y"),
    ]
    distance = np.linalg.norm(
        np.array(cube_position[:2]) - np.array(robot_base_position[:2])
    )
    print("cube_position", cube_position)
    print("cube_orientation", cube_orientation)
    print("robot base position", robot_base_position)
    print("distance", distance)

    env.close()
