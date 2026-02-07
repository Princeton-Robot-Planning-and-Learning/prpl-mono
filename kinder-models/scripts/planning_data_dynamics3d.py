"""Dataset collection using bilevel planning parameterized skills for dynamics3d
environments."""

import argparse

import kinder
import numpy as np
from episode_storage import EpisodeWriter
from relational_structs.spaces import ObjectCentricBoxSpace

from kinder_models.dynamic3d.fk_solver import TidybotFKSolver
from kinder_models.dynamic3d.ground.parameterized_skills import (
    PyBulletSim,
    create_lifted_controllers,
)
from kinder_models.teleop_utils import _visualize_image_in_window

kinder.register_all_environments()


def collect_data(
    output_dir: str = "data/demos",
    seed: int = 123,
    save: bool = True,
    grasping_only: bool = False,
    show_images: bool = False,
):
    """Collect pick and place demonstration data in ground environment.

    Args:
        output_dir: Directory to save episode data.
        seed: Random seed for reproducibility.
        save: Whether to save the episode data to disk.
    """

    # Create the environment.
    num_cubes = 2
    env = kinder.make(
        f"kinder/TidyBot3D-cupboard_real-o{num_cubes}-v0", render_mode="rgb_array"
    )

    # Create episode writer if saving is enabled.
    writer = EpisodeWriter(output_dir) if save else None

    # Reset the environment and get the initial state.
    obs, _ = env.reset(seed=seed)  # type: ignore
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    assert state is not None
    pybullet_sim = PyBulletSim(state, rendering=False)

    controllers = create_lifted_controllers(env.action_space, pybullet_sim=pybullet_sim)  # type: ignore # pylint: disable=line-too-long

    fk_solver = TidybotFKSolver(ee_offset=0.12)

    # Target object for this episode
    target_object_key = "cube1"

    # Create the pick ground controller.
    lifted_controller = controllers["pick_ground"]
    robot = state.get_object_from_name("robot")
    cube = state.get_object_from_name(target_object_key)
    object_parameters = (robot, cube)
    controller = lifted_controller.ground(object_parameters)
    params = controller.sample_parameters(state, np.random.default_rng(seed))

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for step_idx in range(400):
        action = controller.step()
        robot = state.get_object_from_name("robot")
        current_joints = [
            state.get(robot, "pos_arm_joint1"),
            state.get(robot, "pos_arm_joint2"),
            state.get(robot, "pos_arm_joint3"),
            state.get(robot, "pos_arm_joint4"),
            state.get(robot, "pos_arm_joint5"),
            state.get(robot, "pos_arm_joint6"),
            state.get(robot, "pos_arm_joint7"),
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
        # print('target_position: ', target_position)
        # print('target_orientation: ', target_orientation)
        # print('current_position: ', current_position)
        # print('current_orientation: ', current_orientation)
        # print('action: ', action)

        env.unwrapped._object_centric_env.set_render_camera("overview")  # type: ignore # pylint: disable=protected-access
        overview_image = env.unwrapped._object_centric_env.render()  # type: ignore # pylint: disable=protected-access
        env.unwrapped._object_centric_env.set_render_camera("base")  # type: ignore # pylint: disable=protected-access
        base_image = env.unwrapped._object_centric_env.render()  # type: ignore # pylint: disable=protected-access
        env.unwrapped._object_centric_env.set_render_camera("wrist")  # type: ignore # pylint: disable=protected-access
        wrist_image = env.unwrapped._object_centric_env.render()  # type: ignore # pylint: disable=protected-access
        if show_images:
            _visualize_image_in_window(overview_image, "overview")
            _visualize_image_in_window(base_image, "base")
            _visualize_image_in_window(wrist_image, "wrist")
        # Record observation and action before stepping
        if writer is not None:
            # Create observation dict with state vector and images
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
                "gripper_pos": np.array([state.get(robot, "pos_gripper")]),
                "base_image": base_image,
                "wrist_image": wrist_image,
                "overview_image": overview_image,
            }
            # Convert action to dict format
            action_dict = {
                "base_pose": target_base_pose,
                "arm_pos": target_position,
                "arm_quat": target_orientation,
                "gripper_pos": np.array([action[-1]]),
            }
            writer.step(obs_dict, action_dict, target_object_key)

        obs, _, _, _, _, _ = env.step_with_images(action)  # type: ignore
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            print(f"Pick controller terminated after {step_idx + 1} steps")
            break
    else:
        print("Warning: Pick controller did not terminate within 400 steps")

    if not grasping_only:
        # Create the place ground controller.
        lifted_controller = controllers["place_ground"]
        robot = state.get_object_from_name("robot")
        cube = state.get_object_from_name(target_object_key)
        cupboard = state.get_object_from_name("cupboard_1")
        object_parameters = (robot, cube, cupboard)  # type: ignore
        controller = lifted_controller.ground(object_parameters)
        params = controller.sample_parameters(state, np.random.default_rng(seed))

        # Reset and execute the controller until it terminates.
        controller.reset(state, params)
        for step_idx in range(400):
            action = controller.step()
            robot = state.get_object_from_name("robot")
            current_joints = [
                state.get(robot, "pos_arm_joint1"),
                state.get(robot, "pos_arm_joint2"),
                state.get(robot, "pos_arm_joint3"),
                state.get(robot, "pos_arm_joint4"),
                state.get(robot, "pos_arm_joint5"),
                state.get(robot, "pos_arm_joint6"),
                state.get(robot, "pos_arm_joint7"),
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

            env.unwrapped._object_centric_env.set_render_camera("overview")  # type: ignore # pylint: disable=protected-access
            overview_image = env.unwrapped._object_centric_env.render()  # type: ignore # pylint: disable=protected-access
            env.unwrapped._object_centric_env.set_render_camera("base")  # type: ignore # pylint: disable=protected-access
            base_image = env.unwrapped._object_centric_env.render()  # type: ignore # pylint: disable=protected-access
            env.unwrapped._object_centric_env.set_render_camera("wrist")  # type: ignore # pylint: disable=protected-access
            wrist_image = env.unwrapped._object_centric_env.render()  # type: ignore # pylint: disable=protected-access
            if show_images:
                _visualize_image_in_window(overview_image, "overview")
                _visualize_image_in_window(base_image, "base")
                _visualize_image_in_window(wrist_image, "wrist")
            # Record observation and action before stepping
            if writer is not None:
                # Create observation dict with state vector and images
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
                    "gripper_pos": np.array([state.get(robot, "pos_gripper")]),
                    "base_image": base_image,
                    "wrist_image": wrist_image,
                    "overview_image": overview_image,
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
    parser.add_argument("--grasping-only", action="store_true", default=True)
    parser.add_argument("--show-images", action="store_true", default=False)
    parser.add_argument("--no-save", dest="save", action="store_false")
    parser.add_argument(
        "--n-demos", type=int, default=1, help="Number of demos to collect"
    )
    args = parser.parse_args()
    for demo_idx in range(args.n_demos):
        collect_data(
            output_dir=args.output_dir,
            seed=args.seed + demo_idx,
            save=args.save,
            grasping_only=args.grasping_only,
            show_images=args.show_images,
        )


if __name__ == "__main__":
    main()
