"""Teleoperation script for prbench environments using WebXR phone app."""

import argparse
import time
from typing import Any

import cv2 as cv
import numpy as np
import prbench
from constants import POLICY_CONTROL_PERIOD
from episode_storage import EpisodeWriter
from numpy.typing import NDArray
from relational_structs.spaces import ObjectCentricBoxSpace

from prbench_models.dynamic3d.fk_solver import TidybotFKSolver
from prbench_models.dynamic3d.ik_solver import TidybotIKSolver
from prbench_models.teleop_utils import TeleopPolicy

prbench.register_all_environments()


def _visualize_image_in_window(image: NDArray[np.uint8], window_name: str) -> None:
    """Visualize an image in an OpenCV window."""
    if image.dtype == np.uint8 and len(image.shape) == 3:
        # Convert RGB to BGR for proper color display in OpenCV
        display_image = cv.cvtColor(  # pylint: disable=no-member
            image, cv.COLOR_RGB2BGR  # pylint: disable=no-member
        )
        cv.imshow(window_name, display_image)  # pylint: disable=no-member
        cv.waitKey(1)  # pylint: disable=no-member


def run_teleop(
    output_dir: str = "data/teleop",
    seed: int = 123,
    save: bool = True,
    num_episodes: int = 1,
    max_steps: int = 1000,
    num_cubes: int = 2,
    enable_web_server: bool = True,
    port: int = 5000,
    show_images: bool = False,
) -> None:
    """Run teleoperation in the prbench environment.

    Args:
        output_dir: Directory to save episode data.
        seed: Random seed for reproducibility.
        save: Whether to save the episode data to disk.
        num_episodes: Number of episodes to run.
        max_steps: Maximum steps per episode.
        num_cubes: Number of cubes in the environment.
        enable_web_server: Whether to enable the WebXR web server.
        port: Port for the WebXR web server.
    """
    # Create the environment
    env = prbench.make(
        f"prbench/Shelf3D-o{num_cubes}-v0",
        use_gui=True,
        render_mode="rgb_array",
    )

    # Create FK/IK solvers for computing end-effector pose
    fk_solver = TidybotFKSolver(ee_offset=0.0)
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
                # 10hz -> 5hz due to pybullet rendering speed.
                step_end_time = start_time + step_idx * POLICY_CONTROL_PERIOD * 2
                while time.time() < step_end_time:
                    time.sleep(0.0001)

                # Get robot state
                robot = state.get_object_from_name("robot")
                current_joints = np.array(
                    [state.get(robot, f"joint_{i}") for i in range(1, 8)]
                )
                current_position, current_orientation = fk_solver.forward_kinematics(
                    current_joints
                )

                # Create observation dict for policy
                all_images = env.unwrapped._object_centric_env.render_all_cameras()  # type: ignore # pylint: disable=protected-access
                if show_images:
                    _visualize_image_in_window(all_images["overview"], "overview")
                    _visualize_image_in_window(all_images["base"], "base")
                    _visualize_image_in_window(all_images["wrist"], "wrist")
                obs_dict: dict[str, Any] = {
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
                    "overview_image": all_images["overview"],
                    "base_image": all_images["base"],
                    "wrist_image": all_images["wrist"],
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
    """Main function to run teleoperation in prbench."""
    parser = argparse.ArgumentParser(
        description="Run teleoperation in prbench environment"
    )
    parser.add_argument(
        "--output-dir", default="data/teleop_geom3d", help="Directory to save episodes"
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save episodes"
    )
    parser.add_argument(
        "--show-images",
        action="store_true",
        default=False,
        help="Show images in OpenCV windows",
    )
    parser.add_argument("--no-save", dest="save", action="store_false")
    parser.add_argument(
        "--num-episodes", type=int, default=1, help="Number of episodes to run"
    )
    parser.add_argument(
        "--max-steps", type=int, default=1000, help="Maximum steps per episode"
    )
    parser.add_argument(
        "--num-cubes", type=int, default=2, help="Number of cubes in environment"
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

    args = parser.parse_args()

    run_teleop(
        output_dir=args.output_dir,
        seed=args.seed,
        save=args.save,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        num_cubes=args.num_cubes,
        enable_web_server=args.enable_web_server,
        port=args.port,
        show_images=args.show_images,
    )


if __name__ == "__main__":
    main()
