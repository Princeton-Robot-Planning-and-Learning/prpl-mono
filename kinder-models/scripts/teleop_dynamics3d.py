"""Teleoperation script for kinder dynamics3d environments."""

import argparse
import time

import kinder
import numpy as np
from episode_storage import EpisodeWriter
from relational_structs.spaces import ObjectCentricBoxSpace

from kinder_models.dynamic3d.fk_solver import TidybotFKSolver
from kinder_models.dynamic3d.ik_solver import TidybotIKSolver
from kinder_models.policy_constants import POLICY_CONTROL_PERIOD
from kinder_models.teleop_utils import TeleopPolicy, _visualize_image_in_window

kinder.register_all_environments()


def run_teleop(
    output_dir: str = "data/teleop",
    seed: int = 123,
    save: bool = True,
    num_episodes: int = 1,
    max_steps: int = 1000,
    enable_web_server: bool = True,
    port: int = 5000,
    show_images: bool = False,
    env_name: str = "TidyBot3D-cupboard_real-o2-v0",
) -> None:
    """Run teleoperation in the kinder environment.

    Args:
        output_dir: Directory to save episode data.
        seed: Random seed for reproducibility.
        save: Whether to save the episode data to disk.
        num_episodes: Number of episodes to run.
        max_steps: Maximum steps per episode.
        enable_web_server: Whether to enable the WebXR web server.
        port: Port for the WebXR web server.
    """
    # Create the environment
    env = kinder.make(
        f"kinder/{env_name}",
        render_mode="rgb_array",
        scene_bg=True,
    )

    # Create FK/IK solvers for computing end-effector pose
    fk_solver = TidybotFKSolver(ee_offset=0.12)
    ik_solver = TidybotIKSolver(ee_offset=0.12)

    # Create teleop policy
    policy = TeleopPolicy(enable_web_server=enable_web_server, port=port)

    try:
        for episode_idx in range(num_episodes):
            print(f"\n=== Episode {episode_idx + 1}/{num_episodes} ===")
            print("Waiting for user to start episode via WebXR interface...")

            # Create episode writer if saving is enabled
            writer = EpisodeWriter(output_dir) if save else None

            # Reset the environment
            episode_seed = seed + episode_idx
            obs, _ = env.reset(seed=episode_seed)  # type: ignore
            assert isinstance(env.observation_space, ObjectCentricBoxSpace)
            state = env.observation_space.devectorize(obs)

            # Target object for this episode
            target_object_key = "cube0"

            # Reset the policy (waits for user to start if web server enabled)
            policy.reset()
            print("Episode started!")

            start_time = time.time()
            for step_idx in range(max_steps):
                # Enforce desired control frequency
                step_end_time = start_time + step_idx * POLICY_CONTROL_PERIOD
                while time.time() < step_end_time:
                    time.sleep(0.0001)

                # Get robot state
                robot = state.get_object_from_name("robot")
                current_joints = np.array(
                    [state.get(robot, f"pos_arm_joint{i}") for i in range(1, 8)]
                )
                current_position, current_orientation = fk_solver.forward_kinematics(
                    current_joints
                )

                robot_name = env.unwrapped._object_centric_env.robot_name  # type: ignore # pylint: disable=protected-access
                env.unwrapped._object_centric_env.set_render_camera("agent_overview")  # type: ignore # pylint: disable=protected-access
                overview_image = env.unwrapped._object_centric_env.render()  # type: ignore # pylint: disable=protected-access
                env.unwrapped._object_centric_env.set_render_camera(  # type: ignore # pylint: disable=protected-access
                    robot_name + "_base"
                )
                base_image = env.unwrapped._object_centric_env.render()  # type: ignore # pylint: disable=protected-access
                env.unwrapped._object_centric_env.set_render_camera(  # type: ignore # pylint: disable=protected-access
                    robot_name + "_wrist"
                )
                wrist_image = env.unwrapped._object_centric_env.render()  # type: ignore # pylint: disable=protected-access
                env.unwrapped._object_centric_env.set_render_camera("agentview_1")  # type: ignore # pylint: disable=protected-access
                agent_image = env.unwrapped._object_centric_env.render()  # type: ignore # pylint: disable=protected-access
                if show_images:
                    _visualize_image_in_window(overview_image, "agent_overview")
                    _visualize_image_in_window(base_image, "base")
                    _visualize_image_in_window(wrist_image, "wrist")
                    _visualize_image_in_window(agent_image, "agentview_1")
                # Create observation dict for policy
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

                # Get action from policy
                action_result = policy.step(obs_dict)

                # Handle control signals
                if action_result == "end_episode":
                    print(f"User ended episode after {step_idx + 1} steps")
                    break
                if action_result == "reset_env":
                    print("User requested environment reset")
                    break
                if action_result is None:
                    # No action from teleop, hold current pose
                    continue

                action_dict = action_result

                # Convert action dict to env action
                qpos = ik_solver.solve(
                    action_dict["arm_pos"],  # type: ignore
                    action_dict["arm_quat"],  # type: ignore
                    current_joints,
                )
                delta_qpos = (
                    np.mod((qpos - current_joints) + np.pi, 2 * np.pi) - np.pi
                )  # Unwrapped joint angles

                action = np.concatenate(
                    [
                        action_dict["base_pose"] - obs_dict["base_pose"],  # type: ignore
                        delta_qpos,
                        action_dict["gripper_pos"],  # type: ignore
                    ]
                )

                # Record observation and action before stepping
                if writer is not None:
                    writer.step(obs_dict, action_dict, target_object_key)  # type: ignore

                # Execute action in environment
                obs, reward, terminated, truncated, _ = env.step(  # type: ignore # pylint: disable=line-too-long
                    action
                )
                next_state = env.observation_space.devectorize(obs)
                state = next_state

                # Check for episode end
                if terminated or truncated:
                    print(f"Episode ended after {step_idx + 1} steps")
                    print(
                        f"  Reward: {reward}, Terminated: {terminated}, "
                        f"Truncated: {truncated}"
                    )
                    break

            else:
                print(f"Episode reached max steps ({max_steps})")

            # Save episode data to disk
            if writer is not None and len(writer) > 0:
                writer.flush_async()
                writer.wait_for_flush()
                print(f"Episode saved with {len(writer)} steps")

    finally:
        policy.close()
        env.close()  # type: ignore


def main() -> None:
    """Main function to run teleoperation in kinder."""
    parser = argparse.ArgumentParser(
        description="Run teleoperation in kinder environment"
    )
    parser.add_argument(
        "--output-dir", default="data/teleop", help="Directory to save episodes"
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save episodes"
    )
    parser.add_argument("--no-save", dest="save", action="store_false")
    parser.add_argument(
        "--num-episodes", type=int, default=1, help="Number of episodes to run"
    )
    parser.add_argument(
        "--max-steps", type=int, default=1000, help="Maximum steps per episode"
    )
    parser.add_argument(
        "--no-web-server",
        dest="enable_web_server",
        action="store_false",
        default=True,
        help="Disable WebXR web server (for testing)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for WebXR web server (default: 5000)",
    )
    parser.add_argument(
        "--show-images",
        action="store_true",
        default=False,
        help="Show images in OpenCV windows",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="TidyBot3D-tool_use-lab2_kitchen-o5-sweep_the_blocks_into_the_top_drawer_of_the_kitchen_island-v0",  # pylint: disable=line-too-long
        help="Name of the environment",
    )

    args = parser.parse_args()

    run_teleop(
        output_dir=args.output_dir,
        seed=args.seed,
        save=args.save,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        env_name=args.env_name,
        enable_web_server=args.enable_web_server,
        port=args.port,
        show_images=args.show_images,
    )


if __name__ == "__main__":
    main()
