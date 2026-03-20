"""Dataset collection using bilevel planning parameterized skills for kinematic3d
environments."""

import argparse

import kinder
import numpy as np
from episode_storage import EpisodeWriter
from kinder.envs.kinematic3d.base_motion3d import ObjectCentricBaseMotion3DEnv
from kinder.envs.kinematic3d.ground3d import ObjectCentricGround3DEnv
from kinder.envs.kinematic3d.motion3d import ObjectCentricMotion3DEnv
from kinder.envs.kinematic3d.obstruction3d import ObjectCentricObstruction3DEnv
from kinder.envs.kinematic3d.shelf3d import ObjectCentricShelf3DEnv
from kinder.envs.kinematic3d.transport3d import ObjectCentricTransport3DEnv
from relational_structs.spaces import ObjectCentricBoxSpace

from kinder_models.dynamic3d.fk_solver import TidybotFKSolver
from kinder_models.teleop_utils import _visualize_image_in_window

kinder.register_all_environments()


def collect_data(
    output_dir: str = "data/demos",
    num_cubes: int = 1,
    env_name: str = "Shelf3D-o1-v0",
    seed: int = 123,
    save: bool = True,
    show_images: bool = False,
    use_qpos: bool = False,
    use_delta_qpos: bool = False,
):
    """Collect pick and place demonstration data in ground environment.

    Args:
        output_dir: Directory to save episode data.
        seed: Random seed for reproducibility.
        save: Whether to save the episode data to disk.
    """

    # Create the environment.
    env = kinder.make(
        f"kinder/{env_name}", render_mode="rgb_array", realistic_bg=True, use_gui=False
    )

    # Create episode writer if saving is enabled.
    writer = EpisodeWriter(output_dir) if save else None

    # Reset the environment and get the initial state.
    obs, _ = env.reset(seed=seed)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    assert state is not None

    fk_solver = TidybotFKSolver(ee_offset=0.12)

    # Create the pick ground controller.
    if "Shelf3D" in env_name:
        sim = ObjectCentricShelf3DEnv(num_cubes=num_cubes, allow_state_access=True)
        from kinder_models.kinematic3d.shelf3d.parameterized_skills import (  # pylint: disable=import-outside-toplevel
            create_lifted_controllers,
        )
    elif "Ground3D" in env_name:
        sim = ObjectCentricGround3DEnv(  # type: ignore
            num_cubes=num_cubes, allow_state_access=True
        )
        from kinder_models.kinematic3d.ground3d.parameterized_skills import (  # type: ignore # pylint: disable=import-outside-toplevel
            create_lifted_controllers,
        )
    elif "Transport3D" in env_name:
        sim = ObjectCentricTransport3DEnv(  # type: ignore
            num_cubes=num_cubes, allow_state_access=True
        )
        from kinder_models.kinematic3d.transport3d.parameterized_skills import (  # type: ignore # pylint: disable=import-outside-toplevel
            create_lifted_controllers,
        )
    elif "BaseMotion3D" in env_name:
        sim = ObjectCentricBaseMotion3DEnv(allow_state_access=True)  # type: ignore
        from kinder_models.kinematic3d.base_motion3d.parameterized_skills import (  # type: ignore # pylint: disable=import-outside-toplevel
            create_lifted_controllers,
        )
    elif "Motion3D" in env_name:
        sim = ObjectCentricMotion3DEnv(allow_state_access=True)  # type: ignore
        from kinder_models.kinematic3d.motion3d.parameterized_skills import (  # type: ignore # pylint: disable=import-outside-toplevel
            create_lifted_controllers,
        )
    elif "Obstruction3D" in env_name:
        sim = ObjectCentricObstruction3DEnv(  # type: ignore
            num_obstructions=num_cubes, allow_state_access=True
        )
        from kinder_models.kinematic3d.obstruction3d.parameterized_skills import (  # type: ignore # pylint: disable=import-outside-toplevel
            create_lifted_controllers,
        )
    else:
        raise ValueError(f"Environment {env_name} not supported")

    # Create lifted controllers
    controllers = create_lifted_controllers(
        env.action_space,  # type: ignore
        sim,
    )
    if "Shelf3D" in env_name or "Ground3D" in env_name:
        target_object_key = f"cube{num_cubes - 1}"
        lifted_controller = controllers["pick"]
        robot = state.get_object_from_name("robot")
        cube = state.get_object_from_name(target_object_key)
        object_parameters = (robot, cube)
    elif "Transport3D" in env_name:
        target_object_key = "box0"
        lifted_controller = controllers["pick"]
        robot = state.get_object_from_name("robot")
        target = state.get_object_from_name(target_object_key)
        object_parameters = (robot, target)
    elif "BaseMotion3D" in env_name:
        target_object_key = "target"
        lifted_controller = controllers["move_base_to_target"]
        robot = state.get_object_from_name("robot")
        target = state.get_object_from_name("target")
        object_parameters = (robot, target)
    elif "Motion3D" in env_name:
        target_object_key = "target"
        lifted_controller = controllers["move_to_target"]
        robot = state.get_object_from_name("robot")
        target = state.get_object_from_name("target")
        object_parameters = (robot, target)
    elif "Obstruction3D" in env_name:
        target_object_key = "target_block"
        lifted_controller = controllers["pick"]
        robot = state.get_object_from_name("robot")
        target = state.get_object_from_name(target_object_key)
        object_parameters = (robot, target)
    else:
        raise ValueError(f"Environment {env_name} not supported")
    controller = lifted_controller.ground(object_parameters)

    params = np.array([0.5, 0.0])

    controller.reset(state, params)
    for step_idx in range(400):
        action = controller.step()
        robot = state.get_object_from_name("robot")
        current_joints = [
            state.get(robot, "joint_1"),
            state.get(robot, "joint_2"),
            state.get(robot, "joint_3"),
            state.get(robot, "joint_4"),
            state.get(robot, "joint_5"),
            state.get(robot, "joint_6"),
            state.get(robot, "joint_7"),
        ]
        current_position, current_orientation = fk_solver.forward_kinematics(
            np.array(current_joints)
        )

        target_base_pose = np.array(
            [
                state.get(robot, "pos_base_x") + action[0],
                state.get(robot, "pos_base_y") + action[1],
                state.get(robot, "pos_base_rot") + action[2],
            ]
        )
        target_joints = current_joints + action[3:10]
        target_position, target_orientation = fk_solver.forward_kinematics(
            np.array(target_joints)
        )

        all_images = env.unwrapped._object_centric_env.render_all_cameras()  # type: ignore # pylint: disable=protected-access
        if show_images:
            _visualize_image_in_window(all_images["overview"], "overview")
            _visualize_image_in_window(all_images["base"], "base")
            _visualize_image_in_window(all_images["wrist"], "wrist")

        # Record observation and action before stepping
        if writer is not None:
            target_shelf = state.get_object_from_name("shelf")
            target_cube_list = [
                state.get_object_from_name(f"cube{i}") for i in range(num_cubes)
            ]
            target_cube_list_pose = []
            for cube in target_cube_list:
                target_cube_list_pose.append(
                    np.array(
                        [
                            state.get(cube, "pose_x"),
                            state.get(cube, "pose_y"),
                            state.get(cube, "pose_z"),
                            state.get(cube, "pose_qx"),
                            state.get(cube, "pose_qy"),
                            state.get(cube, "pose_qz"),
                            state.get(cube, "pose_qw"),
                        ]
                    )
                )

            # Create observation dict with state vector and images
            if use_qpos:
                obs_dict = {
                    "base_pose": np.array(
                        [
                            state.get(robot, "pos_base_x"),
                            state.get(robot, "pos_base_y"),
                            state.get(robot, "pos_base_rot"),
                        ]
                    ),
                    "arm_qpos": np.array(current_joints),
                    "gripper_pos": np.array([state.get(robot, "finger_state")]),
                    "base_image": all_images["base"],
                    "wrist_image": all_images["wrist"],
                    "overview_image": all_images["overview"],
                }

                if use_delta_qpos:
                    # Convert action to dict format
                    action_dict = {
                        "base_pose": np.array(action[:3]),
                        "arm_qpos": np.array(action[3:10]),
                        "gripper_pos": np.array([action[-1]]),
                    }
                else:
                    # Convert action to dict format
                    action_dict = {
                        "base_pose": target_base_pose,
                        "arm_qpos": np.array(target_joints),
                        "gripper_pos": np.array([action[-1]]),
                    }
            else:
                obs_dict = {
                    "base_pose": np.array(
                        [
                            state.get(robot, "pos_base_x"),
                            state.get(robot, "pos_base_y"),
                            state.get(robot, "pos_base_rot"),
                        ]
                    ),
                    "arm_pos": current_position,
                    "arm_quat": current_orientation,
                    "gripper_pos": np.array([state.get(robot, "finger_state")]),
                    "base_image": all_images["base"],
                    "wrist_image": all_images["wrist"],
                    "overview_image": all_images["overview"],
                }

                # Convert action to dict format
                action_dict = {
                    "base_pose": target_base_pose,
                    "arm_pos": target_position,
                    "arm_quat": target_orientation,
                    "gripper_pos": np.array([action[-1]]),
                }
            writer.step(obs_dict, action_dict, target_object_key)

        obs, _, _, _, _ = env.step(action)  # type: ignore
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            print(f"Pick controller terminated after {step_idx + 1} steps")
            break
    else:
        print("Warning: Pick controller did not terminate within 400 steps")

    add_place = True

    if add_place:
        lifted_controller = controllers["place"]
        robot = state.get_object_from_name("robot")
        target = state.get_object_from_name(target_object_key)
        target_shelf = state.get_object_from_name("shelf")
        object_parameters = (robot, target, target_shelf)  # type: ignore
        controller = lifted_controller.ground(object_parameters)

        params = np.array([0.0, -0.10])

        controller.reset(state, params)
        for step_idx in range(400):
            action = controller.step()
            robot = state.get_object_from_name("robot")
            current_joints = [
                state.get(robot, "joint_1"),
                state.get(robot, "joint_2"),
                state.get(robot, "joint_3"),
                state.get(robot, "joint_4"),
                state.get(robot, "joint_5"),
                state.get(robot, "joint_6"),
                state.get(robot, "joint_7"),
            ]
            current_position, current_orientation = fk_solver.forward_kinematics(
                np.array(current_joints)
            )

            target_base_pose = np.array(
                [
                    state.get(robot, "pos_base_x") + action[0],
                    state.get(robot, "pos_base_y") + action[1],
                    state.get(robot, "pos_base_rot") + action[2],
                ]
            )
            target_joints = current_joints + action[3:10]
            target_position, target_orientation = fk_solver.forward_kinematics(
                np.array(target_joints)
            )

            all_images = env.unwrapped._object_centric_env.render_all_cameras()  # type: ignore # pylint: disable=protected-access
            if show_images:
                _visualize_image_in_window(all_images["overview"], "overview")
                _visualize_image_in_window(all_images["base"], "base")
                _visualize_image_in_window(all_images["wrist"], "wrist")

            # Record observation and action before stepping
            if writer is not None:
                target_shelf = state.get_object_from_name("shelf")
                target_cube_list = [
                    state.get_object_from_name(f"cube{i}") for i in range(num_cubes)
                ]
                target_cube_list_pose = []
                for cube in target_cube_list:
                    target_cube_list_pose.append(
                        np.array(
                            [
                                state.get(cube, "pose_x"),
                                state.get(cube, "pose_y"),
                                state.get(cube, "pose_z"),
                                state.get(cube, "pose_qx"),
                                state.get(cube, "pose_qy"),
                                state.get(cube, "pose_qz"),
                                state.get(cube, "pose_qw"),
                            ]
                        )
                    )

                # Create observation dict with state vector and images
                if use_qpos:
                    obs_dict = {
                        "base_pose": np.array(
                            [
                                state.get(robot, "pos_base_x"),
                                state.get(robot, "pos_base_y"),
                                state.get(robot, "pos_base_rot"),
                            ]
                        ),
                        "arm_qpos": np.array(current_joints),
                        "gripper_pos": np.array([state.get(robot, "finger_state")]),
                        "base_image": all_images["base"],
                        "wrist_image": all_images["wrist"],
                        "overview_image": all_images["overview"],
                    }

                    if use_delta_qpos:
                        # Convert action to dict format
                        action_dict = {
                            "base_pose": np.array(action[:3]),
                            "arm_qpos": np.array(action[3:10]),
                            "gripper_pos": np.array([action[-1]]),
                        }
                    else:
                        # Convert action to dict format
                        action_dict = {
                            "base_pose": target_base_pose,
                            "arm_qpos": np.array(target_joints),
                            "gripper_pos": np.array([action[-1]]),
                        }
                else:
                    obs_dict = {
                        "base_pose": np.array(
                            [
                                state.get(robot, "pos_base_x"),
                                state.get(robot, "pos_base_y"),
                                state.get(robot, "pos_base_rot"),
                            ]
                        ),
                        "arm_pos": current_position,
                        "arm_quat": current_orientation,
                        "gripper_pos": np.array([state.get(robot, "finger_state")]),
                        "base_image": all_images["base"],
                        "wrist_image": all_images["wrist"],
                        "overview_image": all_images["overview"],
                    }

                    # Convert action to dict format
                    action_dict = {
                        "base_pose": target_base_pose,
                        "arm_pos": target_position,
                        "arm_quat": target_orientation,
                        "gripper_pos": np.array([action[-1]]),
                    }
                writer.step(obs_dict, action_dict, target_object_key)

                obs, _, terminated, truncated, _ = env.step(action)  # type: ignore
                next_state = env.observation_space.devectorize(obs)
                controller.observe(next_state)
                state = next_state
                if terminated or truncated:
                    print("env terminated or truncated")
                    break
                if controller.terminated():
                    print(f"Place controller terminated after {step_idx + 1} steps")
                    break
            else:
                print("Warning: Place controller did not terminate within 400 steps")

    # Save episode data to disk
    if writer is not None and len(writer) > 0:
        writer.flush_async()
        writer.wait_for_flush()
        print(f"Episode saved with {len(writer)} steps")

    env.close()  # type: ignore


def main() -> None:
    """Main function to collect demonstration data."""
    parser = argparse.ArgumentParser(description="Collect demonstration data")
    parser.add_argument("--output-dir", default="data/demos", help="Output dir")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--num-cubes", type=int, default=1, help="Number of cubes")
    parser.add_argument(
        "--env-name", type=str, default="Shelf3D-o1-v0", help="Environment name"
    )
    parser.add_argument("--show-images", action="store_true", default=False)
    parser.add_argument("--no-save", dest="save", action="store_false")
    parser.add_argument(
        "--n-demos", type=int, default=1, help="Number of demos to collect"
    )
    parser.add_argument("--use-qpos", action="store_true", default=False)
    parser.add_argument("--use-delta-qpos", action="store_true", default=False)
    args = parser.parse_args()
    for demo_idx in range(args.n_demos):
        collect_data(
            output_dir=args.output_dir,
            seed=args.seed + demo_idx,
            save=args.save,
            num_cubes=args.num_cubes,
            env_name=args.env_name,
            show_images=args.show_images,
            use_qpos=args.use_qpos,
            use_delta_qpos=args.use_delta_qpos,
        )


if __name__ == "__main__":
    main()
