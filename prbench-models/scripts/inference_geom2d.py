"""Policy inference script for running remote policies in prbench environments."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import imageio as iio
import cv2 as cv
import numpy as np
import prbench
import zmq
from relational_structs.spaces import ObjectCentricBoxSpace

from prbench_models.policy_constants import (
    POLICY_CONTROL_PERIOD,
    POLICY_IMAGE_HEIGHT,
    POLICY_IMAGE_WIDTH,
    POLICY_SERVER_HOST,
    POLICY_SERVER_PORT,
)
from prbench_models.teleop_utils import _visualize_image_in_window

prbench.register_all_environments()


class RemotePolicy:
    """Execute policy running on remote server via ZMQ."""

    def __init__(
        self,
        host: str = POLICY_SERVER_HOST,
        port: int = POLICY_SERVER_PORT,
        image_width: int = POLICY_IMAGE_WIDTH,
        image_height: int = POLICY_IMAGE_HEIGHT,
    ):
        self.image_width = image_width
        self.image_height = image_height

        # Connection to policy server
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")
        print(f"Connected to policy server at {host}:{port}")

    def reset(self, target_object_key: str):
        """Reset the policy on the server."""
        # Check connection to policy server and reset policy
        default_timeout = self.socket.getsockopt(zmq.RCVTIMEO)
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)  # Temporarily set 1000 ms timeout
        self.socket.send_pyobj({"reset": True, "target_object_key": target_object_key})
        try:
            self.socket.recv_pyobj()  # Note: Not secure. Only unpickle data you trust.
        except zmq.error.Again as e:
            raise Exception("Could not communicate with policy server") from e
        self.socket.setsockopt(
            zmq.RCVTIMEO, default_timeout
        )  # Put default timeout back
        print("Policy reset successful")

    def step(self, obs: dict) -> dict:
        """Get action from policy server.

        Args:
            obs: Observation dictionary with state and image keys.

        Returns:
            Action dictionary from the policy server.
        """
        # Encode images
        encoded_obs = {}
        for k, v in obs.items():
            if isinstance(v, np.ndarray) and v.ndim == 3:
                # Resize image to resolution expected by policy server
                v = cv.resize(  # pylint: disable=no-member
                    v, (self.image_width, self.image_height)
                )
                # Encode image as JPEG
                _, v = cv.imencode(  # pylint: disable=no-member
                    ".jpg", v
                )  # Note: Interprets RGB as BGR
                encoded_obs[k] = v
            else:
                encoded_obs[k] = v

        # Send obs to policy server
        req = {"obs": encoded_obs}
        self.socket.send_pyobj(req)

        # Get action from policy server
        rep = (
            self.socket.recv_pyobj()
        )  # Note: Not secure. Only unpickle data you trust.
        action = rep["action"]

        return action

    def close(self):
        """Close the connection to policy server."""
        self.socket.close()


def run_inference(
    output_dir: Path = Path("data/inference"),
    seed: int = 123,
    num_episodes: int = 1,
    max_steps: int = 200,
    policy_host: str = POLICY_SERVER_HOST,
    policy_port: int = POLICY_SERVER_PORT,
    env_name: str = "Shelf3D-o1-v0",
    render: bool = False,
    num_cubes: int = 1,
    show_images: bool = False,
    use_qpos: bool = False,
    use_delta_qpos: bool = False,
    use_env_state: bool = False,
    save_videos: bool = False,
):
    """Run policy inference in the prbench environment.

    Args:
        output_dir: Directory to save episode data.
        seed: Random seed for reproducibility.
        num_episodes: Number of episodes to run.
        max_steps: Maximum steps per episode.
        policy_host: Policy server hostname.
        policy_port: Policy server port.
        env_name: Name of the environment.
        render: Whether to render the environment.
        num_cubes: Number of cubes in the environment.
        show_images: Whether to show images in a window.
        use_qpos: Whether to use qpos for the policy.
        use_delta_qpos: Whether to use delta qpos for the policy.
        use_env_state: Whether to use env state for the policy.
        save_videos: Whether to save videos for evaluation.
    """
    

    # Episode tracking
    successes = 0
    episode_rewards = []
    episode_lengths = []
    episode_terminated = []
    episode_truncated = []
    episode_seeds = []
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_parent_dir = output_dir / f"videos_{env_name}_{timestamp}"
        video_parent_dir.mkdir(parents=True, exist_ok=True)
        for episode_idx in range(num_episodes):
            # Create the environment
            render_mode = "rgb_array" if render or save_videos else None
            if "TidyBot" in env_name:
                env = prbench.make(
                    f"prbench/{env_name}",
                    render_mode=render_mode,
                    scene_bg=True,
                )
            else:
                env = prbench.make(
                    f"prbench/{env_name}",
                    render_mode=render_mode,
                )

            
            if save_videos:
                video_dir = video_parent_dir / f"eval_episode_{episode_idx}" 
                video_dir.mkdir(parents=True, exist_ok=True)
                overview_images = []
                base_images = []
                wrist_images = []
                images_2d = []
                fps = 30
            # Create remote policy
            policy = RemotePolicy(host=policy_host, port=policy_port)

            print(f"\n=== Episode {episode_idx + 1}/{num_episodes} ===")

            # Reset the environment
            episode_seed = seed + episode_idx
            episode_seeds.append(episode_seed)
            (
                obs,
                _,
            ) = env.reset(
                seed=episode_seed
            )  # type: ignore
            assert isinstance(env.observation_space, ObjectCentricBoxSpace)
            state = env.observation_space.devectorize(obs)

            # Target object for this episode (can be detected or specified)
            if "DynPushPullHook2D" in env_name or "DynObstruction2D" in env_name or "Motion2D" in env_name or "StickButton2D" in env_name:
                target_object_key = "target_agent"
            elif "Shelf3D" in env_name or "Ground3D" in env_name:
                target_object_key = f"cube{num_cubes - 1}"
            elif "Transport3D" in env_name:
                target_object_key = "box0"
            elif "BaseMotion3D" in env_name or "TidyBot" in env_name or "Transport3D" in env_name:
                target_object_key = "target"
            elif "Motion3D" in env_name:
                target_object_key = "target"
            elif "Obstruction3D" in env_name:
                target_object_key = "target_block"
            else:
                raise ValueError(f"Environment {env_name} not supported")

            # Reset the policy
            policy.reset(target_object_key)  # type: ignore

            # Episode metrics
            episode_reward = 0.0
            ep_terminated = False
            ep_truncated = False
            
            start_time = time.time()
            for step_idx in range(max_steps):
                # Enforce desired control frequency
                step_end_time = start_time + step_idx * POLICY_CONTROL_PERIOD
                while time.time() < step_end_time:
                    time.sleep(0.0001)

                # Get robot state
                robot = state.get_object_from_name("robot")

                if "BaseMotion3D" in env_name or "Transport3D" in env_name:
                    all_images = env.unwrapped._object_centric_env.render_all_cameras()
                    overview_image = all_images["overview"]
                    base_image = all_images["base"]
                    wrist_image = all_images["wrist"]
                    if save_videos:
                        overview_images.append(overview_image)
                        base_images.append(base_image)
                        wrist_images.append(wrist_image)
                    if show_images:
                        _visualize_image_in_window(overview_image, "overview")
                        _visualize_image_in_window(base_image, "base")
                        _visualize_image_in_window(wrist_image, "wrist")
                elif "TidyBot" in env_name:
                    robot_name = env.unwrapped._object_centric_env.robot_name
                    env.unwrapped._object_centric_env.set_render_camera("agentview_1")
                    overview_image = env.unwrapped._object_centric_env.render()
                    env.unwrapped._object_centric_env.set_render_camera(robot_name + "_base")
                    base_image = env.unwrapped._object_centric_env.render()
                    env.unwrapped._object_centric_env.set_render_camera(robot_name+ "_wrist")
                    wrist_image = env.unwrapped._object_centric_env.render()
                    if save_videos:
                        overview_images.append(overview_image)
                        base_images.append(base_image)
                        wrist_images.append(wrist_image)
                    if show_images:
                        _visualize_image_in_window(overview_image, "overview")
                        _visualize_image_in_window(base_image, "base")
                        _visualize_image_in_window(wrist_image, "wrist")
                else:
                    image = env.unwrapped._object_centric_env.render()
                    if save_videos:
                        images_2d.append(image)
                    if show_images:
                        _visualize_image_in_window(image, "overview")

                # Create observation dict for policy
                if use_env_state:
                    if "TidyBot" in env_name or "BaseMotion3D" in env_name or "Transport3D" in env_name:
                        obs_dict = {
                            "robot_state": env.observation_space.get_object_subvector(obs, "robot"),
                            "env_state": env.observation_space.get_vector_excluding_object(obs, "robot"),
                            "overview_image": overview_image,
                            "base_image": base_image,
                            "wrist_image": wrist_image,
                        }
                    else:
                        obs_dict = {
                            "robot_state": env.observation_space.get_object_subvector(obs, "robot"),
                            "env_state": env.observation_space.get_vector_excluding_object(obs, "robot"),
                            "image": image,
                        }
                else:
                    if "TidyBot" in env_name or "BaseMotion3D" in env_name or "Transport3D" in env_name:
                        obs_dict = {
                            "robot_state": env.observation_space.get_object_subvector(obs, "robot"),
                            "env_state": env.observation_space.get_vector_excluding_object(obs, "robot"),
                            "overview_image": overview_image,
                            "base_image": base_image,
                            "wrist_image": wrist_image,
                        }
                    else:
                        obs_dict = {
                            "robot_state": env.observation_space.get_object_subvector(obs, "robot"),
                            "env_state": env.observation_space.get_vector_excluding_object(obs, "robot"),
                            "image": image,
                        }
                
                if "TidyBot" in env_name:
                    assert obs_dict["robot_state"].shape == obs[-22:].shape
                    if "env_state" in obs_dict:
                        assert obs_dict["env_state"].shape == obs[:-22].shape
                elif "BaseMotion3D" in env_name or "Transport3D" in env_name:
                    assert obs_dict["robot_state"].shape == obs[:19].shape
                    if "env_state" in obs_dict:
                        assert obs_dict["env_state"].shape == obs[19:].shape
                elif "DynPushPullHook2D" in env_name:
                    assert obs_dict["robot_state"].shape == obs[:24].shape
                    if "env_state" in obs_dict:
                        assert obs_dict["env_state"].shape == obs[24:].shape
                elif "DynObstruction2D" in env_name:
                    assert obs_dict["robot_state"].shape == obs[-24:].shape
                    if "env_state" in obs_dict:
                        assert obs_dict["env_state"].shape == obs[:-24].shape
                elif "Motion2D" in env_name or "StickButton2D" in env_name:
                    assert obs_dict["robot_state"].shape == obs[:9].shape
                    if "env_state" in obs_dict:
                        assert obs_dict["env_state"].shape == obs[9:].shape
                
                # Get action from policy
                action_dict = policy.step(obs_dict)

                
                if action_dict is None:
                    action_dict = {
                        "robot_actions": np.zeros(env.action_space.shape[0], dtype=np.float32)
                    }
                
                action = action_dict["robot_actions"]
                epsilon = 1e-4
                action = np.clip(action, env.action_space.low + epsilon, env.action_space.high - epsilon)
                if "BaseMotion3D" in env_name:
                    action[3:] = 0.0

                action = action.astype(np.float32)
                # Execute action in environment
                obs, reward, terminated, truncated, _ = env.step(  # type: ignore # pylint: disable=line-too-long
                    action
                )
                episode_reward += reward
                next_state = env.observation_space.devectorize(obs)
                state = next_state

                # Check for episode end
                if terminated or truncated:
                    ep_terminated = terminated
                    ep_truncated = truncated
                    print(f"Episode ended after {step_idx + 1} steps")
                    print(
                        f"  Reward: {reward}, Total Reward: {episode_reward:.3f}, "
                        f"Terminated: {terminated}, Truncated: {truncated}"
                    )
                    if terminated:
                        successes += 1
                    episode_lengths.append(step_idx + 1)
                    break

            else:
                # Max steps reached without termination
                episode_lengths.append(max_steps)
                print(f"Episode reached max steps ({max_steps})")
            
            # Log episode results (runs for both break and normal completion)
            episode_rewards.append(episode_reward)
            episode_terminated.append(ep_terminated)
            episode_truncated.append(ep_truncated)

            if save_videos:
                if len(overview_images) > 0:
                    overview_video_path = video_dir / "overview.mp4"
                    iio.mimsave(overview_video_path, overview_images, fps=fps)
                if len(base_images) > 0:
                    base_video_path = video_dir / "base.mp4"
                    iio.mimsave(base_video_path, base_images, fps=fps)
                if len(wrist_images) > 0:
                    wrist_video_path = video_dir / "wrist.mp4"
                    iio.mimsave(wrist_video_path, wrist_images, fps=fps)
                if len(images_2d) > 0:
                    image_video_path = video_dir / "image.mp4"
                    iio.mimsave(image_video_path, images_2d, fps=fps)
            
            print(f"Episode {episode_idx + 1}: reward={episode_reward:.3f}, "
                  f"terminated={ep_terminated}, truncated={ep_truncated}")
            policy.close()  # type: ignore
            env.close()  # type: ignore

    finally:
        # Print summary statistics
        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Environment: {env_name}")
        print(f"Episodes completed: {len(episode_rewards)}/{num_episodes}")
        print(f"Successes (terminated): {successes}")
        print(f"Success rate: {successes / max(len(episode_rewards), 1):.2%}")
        
        if episode_rewards:
            print(f"\nReward Statistics:")
            print(f"  Total rewards: {episode_rewards}")
            print(f"  Average reward: {np.mean(episode_rewards):.3f}")
            print(f"  Std reward: {np.std(episode_rewards):.3f}")
            print(f"  Min reward: {np.min(episode_rewards):.3f}")
            print(f"  Max reward: {np.max(episode_rewards):.3f}")
        
        if episode_lengths:
            print(f"\nEpisode Length Statistics:")
            print(f"  Average length: {np.mean(episode_lengths):.1f}")
            print(f"  Min length: {np.min(episode_lengths)}")
            print(f"  Max length: {np.max(episode_lengths)}")
        
        print(f"\nTerminated: {sum(episode_terminated)}, Truncated: {sum(episode_truncated)}")
        print("=" * 50)
        
        # Save logs to JSON file
        logs = {
            "environment": env_name,
            "seed": seed,
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "timestamp": timestamp,
            "episodes_completed": len(episode_rewards),
            "successes": successes,
            "success_rate": successes / max(len(episode_rewards), 1),
            "episode_seeds": episode_seeds,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "episode_terminated": episode_terminated,
            "episode_truncated": episode_truncated,
            "reward_stats": {
                "mean": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
                "std": float(np.std(episode_rewards)) if episode_rewards else 0.0,
                "min": float(np.min(episode_rewards)) if episode_rewards else 0.0,
                "max": float(np.max(episode_rewards)) if episode_rewards else 0.0,
            },
            "length_stats": {
                "mean": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
                "min": int(np.min(episode_lengths)) if episode_lengths else 0,
                "max": int(np.max(episode_lengths)) if episode_lengths else 0,
            },
        }
        
        logs_path = video_parent_dir / "evaluation_logs.json"
        with open(logs_path, "w") as f:
            json.dump(logs, f, indent=2)
        print(f"\nLogs saved to: {logs_path}")


def main() -> None:
    """Main function to run policy inference in prbench."""
    parser = argparse.ArgumentParser(description="Run policy inference in prbench")
    parser.add_argument(
        "--output-dir", default="data/evaluations", help="Directory to save episodes"
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num-episodes", type=int, default=1, help="Number of episodes to run"
    )
    parser.add_argument(
        "--num-cubes", type=int, default=1, help="Number of cubes in environment"
    )
    parser.add_argument(
        "--max-steps", type=int, default=400, help="Maximum steps per episode"
    )
    parser.add_argument(
        "--policy-host",
        default=POLICY_SERVER_HOST,
        help="Policy server hostname",
    )
    parser.add_argument(
        "--policy-port",
        type=int,
        default=POLICY_SERVER_PORT,
        help="Policy server port",
    )
    parser.add_argument(
        "--env-name", type=str, default="Shelf3D-o1-v0", help="Name of the environment"
    )
    parser.add_argument(
        "--show-images",
        action="store_true",
        default=False,
        help="Show images in a window",
    )
    parser.add_argument("--save-videos", action="store_true", default=False, help="Save videos for evaluation")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--use-qpos", action="store_true", default=False, help="Use qpos for the policy")
    parser.add_argument("--use-delta-qpos", action="store_true", default=False, help="Use delta qpos for the policy")
    parser.add_argument("--use-env-state", type=bool, default=True, help="Use env state for the policy")
    args = parser.parse_args()

    run_inference(
        output_dir=Path(args.output_dir),
        seed=args.seed,
        num_episodes=args.num_episodes,
        num_cubes=args.num_cubes,
        max_steps=args.max_steps,
        policy_host=args.policy_host,
        policy_port=args.policy_port,
        env_name=args.env_name,
        render=args.render,
        show_images=args.show_images,
        use_qpos=args.use_qpos,
        use_delta_qpos=args.use_delta_qpos,
        use_env_state=args.use_env_state,
        save_videos=args.save_videos,
    )


if __name__ == "__main__":
    main()
