"""Teleoperation script for kinder dynamics3d environments.

Saves demonstrations in the same format as collect_demos_ds.py for compatibility with
generate_demo_video.py and other kinder tools.

Example usage:
    python teleop_dynamics3d_prbench.py \
        --teleop-device vr \
        --env-name Tossing3D-o1-v0 \
        --show-images \
        --max-steps 10000
"""

import argparse
import time
from pathlib import Path
from typing import Any

import dill as pkl  # type: ignore[import-untyped]
import kinder
import numpy as np
from relational_structs.spaces import ObjectCentricBoxSpace

from kinder_models.dynamic3d.fk_solver import TidybotFKSolver
from kinder_models.dynamic3d.ik_solver import TidybotIKSolver
from kinder_models.policy_constants import POLICY_CONTROL_PERIOD
from kinder_models.teleop_utils import (
    QuestTeleopPolicy,
    TeleopPolicy,
    _visualize_image_in_window,
)

kinder.register_all_environments()

# Default demos directory: ../kinder/demos relative to this script
# Script: prpl-mono/kinder-models/scripts/teleop_dynamics3d_kinder.py
# Demos:  prpl-mono/kinder/demos
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_DEMOS_DIR = _SCRIPT_DIR.parent.parent / "kinder" / "demos"


def sanitize_env_id(env_id: str) -> str:
    """Remove unnecessary stuff from the env ID.

    Mirrors the function in kinder/scripts/generate_env_docs.py and collect_demos_ds.py
    for consistent directory naming.
    """
    if env_id.startswith("kinder/"):
        env_id = env_id[len("kinder/") :]
    env_id = env_id.replace("/", "_")
    if len(env_id) >= 3 and env_id[-3:-1] == "-v":
        return env_id[:-3]
    return env_id


def save_demo(
    demo_dir: Path,
    env_id: str,
    seed: int,
    observations: list[Any],
    actions: list[Any],
    rewards: list[float],
    terminated: bool,
    truncated: bool,
) -> Path:
    """Save a demo to disk in the same format as collect_demos_ds.py.

    Directory structure: {demo_dir}/{sanitized_env_id}/{seed}/{timestamp}.p
    """
    timestamp = int(time.time())
    demo_subdir = demo_dir / sanitize_env_id(env_id) / str(seed)
    demo_subdir.mkdir(parents=True, exist_ok=True)
    demo_path = demo_subdir / f"{timestamp}.p"
    demo_data = {
        "env_id": env_id,
        "timestamp": timestamp,
        "seed": seed,
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminated": terminated,
        "truncated": truncated,
    }
    with open(demo_path, "wb") as f:
        pkl.dump(demo_data, f)
    return demo_path


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
    teleop_device: str = "phone",
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
        show_images: Whether to show images in OpenCV windows.
        env_name: Name of the kinder environment.
        teleop_device: Type of teleop interface ("phone" for WebXR or "vr"
            for Quest).
    """
    env_id = f"kinder/{env_name}"
    demo_dir = Path(output_dir)

    # Create the environment
    env = kinder.make(
        env_id,
        render_mode="rgb_array",
        scene_bg=True,
    )

    # Create FK/IK solvers for computing end-effector pose
    fk_solver = TidybotFKSolver(ee_offset=0.12)
    ik_solver = TidybotIKSolver(ee_offset=0.12)

    # Create teleop policy based on type
    if teleop_device == "phone":
        policy: TeleopPolicy | QuestTeleopPolicy = TeleopPolicy(
            enable_web_server=enable_web_server, port=port
        )
    elif teleop_device == "vr":
        policy = QuestTeleopPolicy(debug=False)
    else:
        raise ValueError(
            f"Invalid teleop_device '{teleop_device}'. Must be 'phone' or 'vr'."
        )

    try:
        for episode_idx in range(num_episodes):
            print(f"\n=== Episode {episode_idx + 1}/{num_episodes} ===")
            print("Waiting for user to start episode via WebXR interface...")

            # Reset the environment
            episode_seed = seed + episode_idx
            obs, _ = env.reset(seed=episode_seed)  # type: ignore
            assert isinstance(env.observation_space, ObjectCentricBoxSpace)
            state = env.observation_space.devectorize(obs)

            # Initialize demo collection lists (same format as collect_demos_ds.py)
            observations: list[Any] = [obs]  # Start with initial observation
            actions: list[Any] = []
            rewards: list[float] = []
            terminated = False
            truncated = False

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
                robot_name = env.unwrapped._object_centric_env.robot_name  # type: ignore[attr-defined] # pylint: disable=protected-access
                robot_type = env.unwrapped._object_centric_env.robot_type  # type: ignore[attr-defined] # pylint: disable=protected-access
                robot = state.get_object_from_name(robot_name)
                assert (
                    robot is not None
                ), f"Robot with name '{robot_name}' not found in state"
                assert (
                    robot_type == "tidybot"
                ), f"Expected robot type 'tidybot', but got '{robot_type}'"
                current_joints = np.array(
                    [state.get(robot, f"pos_arm_joint{i}") for i in range(1, 8)]
                )
                current_position, current_orientation = fk_solver.forward_kinematics(
                    current_joints
                )

                # Render images
                images = env.unwrapped._object_centric_env.render_all_cameras()  # type: ignore[attr-defined] # pylint: disable=protected-access
                task_view_image = images["task_view_image"]
                base_image = images[robot_name + "_base_image"]
                wrist_image = images[robot_name + "_wrist_image"]

                if show_images:
                    _visualize_image_in_window(task_view_image, "task_view")

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
                    "task_view_image": task_view_image,
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

                # Execute action in environment
                obs, reward, ep_terminated, ep_truncated, _ = env.step(  # type: ignore
                    action
                )

                # Record data for demo (same format as collect_demos_ds.py)
                observations.append(obs)
                actions.append(action)
                rewards.append(float(reward))

                next_state = env.observation_space.devectorize(obs)
                state = next_state

                # Check for episode end
                if ep_terminated:
                    terminated = True
                    print(f"Episode terminated after {step_idx + 1} steps")
                    print(f"  Reward: {reward}")
                    break
                if ep_truncated:
                    truncated = True
                    print(f"Episode truncated after {step_idx + 1} steps")
                    print(f"  Reward: {reward}")
                    break

            else:
                print(f"Episode reached max steps ({max_steps})")
                truncated = True

            # Save episode data to disk (same format as collect_demos_ds.py)
            if save and len(actions) > 0:
                demo_path = save_demo(
                    demo_dir,
                    env_id,
                    episode_seed,
                    observations,
                    actions,
                    rewards,
                    terminated,
                    truncated,
                )
                print(f"Episode saved to {demo_path}")
                print(f"  Observations: {len(observations)}, Actions: {len(actions)}")
            elif save:
                print("No actions recorded, episode not saved")

    finally:
        policy.close()
        env.close()  # type: ignore


def main() -> None:
    """Main function to run teleoperation in kinder."""
    parser = argparse.ArgumentParser(
        description="Run teleoperation in kinder environment"
    )
    parser.add_argument(
        "--output-dir",
        default=str(_DEFAULT_DEMOS_DIR),
        help="Directory to save episodes (default: kinder/demos)",
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
        default="Tossing3D-o1-v0",
        help="Name of the environment",
    )
    parser.add_argument(
        "--teleop-device",
        type=str,
        default="phone",
        choices=["phone", "vr"],
        help="Type of teleoperation interface: 'phone' for WebXR or 'vr' for Quest VR",
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
        teleop_device=args.teleop_device,
    )


if __name__ == "__main__":
    main()
