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
from prpl_utils.utils import sample_seed_from_rng

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
    remove_velocity: bool = False,
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
        remove_velocity: Whether to remove velocity from the policy.
    """
    

    # Episode tracking
    successes = 0
    episode_rewards = []
    episode_lengths = []
    episode_terminated = []
    episode_truncated = []
    episode_seeds = []
    episode_avg_inference_times = []
    executed_action_steps = 8
    
    try:
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        for episode_idx in range(num_episodes):
            this_episode_inference_times = []
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
                video_dir = seed_dir / f"eval_episode_{episode_idx}" 
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
            episode_seed = sample_seed_from_rng(rng)
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
                        if remove_velocity and "TidyBot" in env_name:
                            obs_dict = {
                                "robot_state": env.observation_space.get_object_subvector(obs, "robot")[:11],
                                "env_state": env.observation_space.get_vector_excluding_object(obs, "robot"),
                                "overview_image": overview_image,
                                "base_image": base_image,
                                "wrist_image": wrist_image,
                            }
                        else:
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
                        if remove_velocity and "TidyBot" in env_name:
                            obs_dict = {
                                "robot_state": env.observation_space.get_object_subvector(obs, "robot")[:11],
                                "overview_image": overview_image,
                                "base_image": base_image,
                                "wrist_image": wrist_image,
                            }
                        else:
                            obs_dict = {
                                "robot_state": env.observation_space.get_object_subvector(obs, "robot"),
                                "overview_image": overview_image,
                                "base_image": base_image,
                                "wrist_image": wrist_image,
                            }
                    else:
                        obs_dict = {
                            "robot_state": env.observation_space.get_object_subvector(obs, "robot"),
                            "image": image,
                        }
                
                if "TidyBot" in env_name:
                    if remove_velocity:
                        assert obs_dict["robot_state"].shape == obs[-22:-11].shape
                    else:
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
                        "robot_actions": np.zeros(env.action_space.shape[0], dtype=np.float32),
                        "inference_time": 0.0
                    }
                
                action = action_dict["robot_actions"]
                inference_time = action_dict["inference_time"]
                epsilon = 1e-4
                if "2D" in env_name:
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

                this_episode_inference_times.append(inference_time/executed_action_steps)

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
            episode_avg_inference_times.append(np.sum(this_episode_inference_times)/episode_lengths[-1])
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

        if episode_avg_inference_times:
            print(f"\nAverage Inference Time Statistics (All Episodes):")
            print(f"  Average inference time: {np.mean(episode_avg_inference_times):.3f}")
            print(f"  Std inference time: {np.std(episode_avg_inference_times):.3f}")
            print(f"  Min inference time: {np.min(episode_avg_inference_times):.3f}")
            print(f"  Max inference time: {np.max(episode_avg_inference_times):.3f}")
        
        if episode_rewards:
            print(f"\nReward Statistics (All Episodes):")
            print(f"  Total rewards: {episode_rewards}")
            print(f"  Average reward: {np.mean(episode_rewards):.3f}")
            print(f"  Std reward: {np.std(episode_rewards):.3f}")
            print(f"  Min reward: {np.min(episode_rewards):.3f}")
            print(f"  Max reward: {np.max(episode_rewards):.3f}")
        
        # if episode_lengths:
        #     print(f"\nEpisode Length Statistics (All Episodes):")
        #     print(f"  Average length: {np.mean(episode_lengths):.1f}")
        #     print(f"  Min length: {np.min(episode_lengths)}")
        #     print(f"  Max length: {np.max(episode_lengths)}")
        
        # Calculate stats for successful episodes only
        successful_rewards = [r for r, t in zip(episode_rewards, episode_terminated) if t]
        successful_lengths = [l for l, t in zip(episode_lengths, episode_terminated) if t]
        
        if successful_rewards:
            print(f"\nReward Statistics (Successful Episodes Only):")
            print(f"  Average reward: {np.mean(successful_rewards):.3f}")
            print(f"  Std reward: {np.std(successful_rewards):.3f}")
            print(f"  Min reward: {np.min(successful_rewards):.3f}")
            print(f"  Max reward: {np.max(successful_rewards):.3f}")
        
        # if successful_lengths:
        #     print(f"\nEpisode Length Statistics (Successful Episodes Only):")
        #     print(f"  Average length: {np.mean(successful_lengths):.1f}")
        #     print(f"  Min length: {np.min(successful_lengths)}")
        #     print(f"  Max length: {np.max(successful_lengths)}")
        
        print(f"\nTerminated: {sum(episode_terminated)}, Truncated: {sum(episode_truncated)}")
        print("=" * 50)
        
        # Save logs to JSON file
        logs = {
            "environment": env_name,
            "seed": seed,
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "episodes_completed": len(episode_rewards),
            "successes": successes,
            "success_rate": successes / max(len(episode_rewards), 1),
            "episode_seeds": episode_seeds,
            "episode_rewards": [float(r) for r in episode_rewards],
            "episode_lengths": [int(l) for l in episode_lengths],
            "episode_terminated": [bool(t) for t in episode_terminated],
            "episode_truncated": [bool(t) for t in episode_truncated],
            "episode_avg_inference_times": [float(t) for t in episode_avg_inference_times],
            "inference_time_stats": {
                "mean": float(np.mean(episode_avg_inference_times)) if episode_avg_inference_times else 0.0,
                "std": float(np.std(episode_avg_inference_times)) if episode_avg_inference_times else 0.0,
                "min": float(np.min(episode_avg_inference_times)) if episode_avg_inference_times else 0.0,
                "max": float(np.max(episode_avg_inference_times)) if episode_avg_inference_times else 0.0,
            },
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
            "successful_reward_stats": {
                "mean": float(np.mean(successful_rewards)) if successful_rewards else 0.0,
                "std": float(np.std(successful_rewards)) if successful_rewards else 0.0,
                "min": float(np.min(successful_rewards)) if successful_rewards else 0.0,
                "max": float(np.max(successful_rewards)) if successful_rewards else 0.0,
            },
            "successful_length_stats": {
                "mean": float(np.mean(successful_lengths)) if successful_lengths else 0.0,
                "min": int(np.min(successful_lengths)) if successful_lengths else 0,
                "max": int(np.max(successful_lengths)) if successful_lengths else 0,
            },
        }
        
        logs_path = seed_dir / "evaluation_logs.json"
        with open(logs_path, "w") as f:
            json.dump(logs, f, indent=2)
        print(f"\nLogs saved to: {logs_path}")

def run_summary(
    output_dir: Path
):
    """Aggregate evaluation logs from all seed directories and print summary."""
    # Find all seed directories
    seed_dirs = sorted(output_dir.glob("seed_*"))
    if not seed_dirs:
        print("No seed directories found.")
        return
    
    # Load all evaluation logs
    all_logs = []
    for seed_dir in seed_dirs:
        logs_path = seed_dir / "evaluation_logs.json"
        if logs_path.exists():
            with open(logs_path, "r") as f:
                logs = json.load(f)
                all_logs.append(logs)
                print(f"Loaded: {logs_path}")
        else:
            print(f"Warning: {logs_path} not found")
    
    if not all_logs:
        print("No evaluation logs found.")
        return
    
    # Calculate per-seed statistics first, then aggregate across seeds
    all_success_rates = [log["success_rate"] for log in all_logs]
    per_seed_mean_rewards = []
    per_seed_mean_lengths = []
    per_seed_mean_inference_times = []
    per_seed_mean_successful_rewards = []
    per_seed_mean_successful_lengths = []
    total_successes = 0
    total_episodes = 0
    
    for log in all_logs:
        episode_rewards = log.get("episode_rewards", [])
        episode_lengths = log.get("episode_lengths", [])
        episode_terminated = log.get("episode_terminated", [])
        episode_inference_times = log.get("episode_avg_inference_times", [])
        
        if episode_rewards:
            per_seed_mean_rewards.append(np.mean(episode_rewards))
        if episode_lengths:
            per_seed_mean_lengths.append(np.mean(episode_lengths))
        if episode_inference_times:
            per_seed_mean_inference_times.append(np.mean(episode_inference_times))
        
        # Calculate stats for successful episodes only
        successful_rewards = [r for r, t in zip(episode_rewards, episode_terminated) if t]
        successful_lengths = [l for l, t in zip(episode_lengths, episode_terminated) if t]
        if successful_rewards:
            per_seed_mean_successful_rewards.append(np.mean(successful_rewards))
        if successful_lengths:
            per_seed_mean_successful_lengths.append(np.mean(successful_lengths))
        
        total_successes += log.get("successes", 0)
        total_episodes += log.get("episodes_completed", 0)
    
    # Print aggregated summary
    print("\n" + "=" * 60)
    print("AGGREGATED SUMMARY ACROSS ALL SEEDS")
    print("=" * 60)
    print(f"Number of seeds: {len(all_logs)}")
    print(f"Total episodes: {total_episodes}")
    print(f"Total successes: {total_successes}")
    print(f"Overall success rate: {total_successes / max(total_episodes, 1):.2%}")
    
    print(f"\nSuccess Rate Across Seeds:")
    print(f"  Mean: {np.mean(all_success_rates):.2%}")
    print(f"  Std: {np.std(all_success_rates):.2%}")
    print(f"  Min: {np.min(all_success_rates):.2%}")
    print(f"  Max: {np.max(all_success_rates):.2%}")
    
    if per_seed_mean_rewards:
        print(f"\nReward Statistics (Mean per Seed, Aggregated Across Seeds):")
        print(f"  Mean: {np.mean(per_seed_mean_rewards):.3f}")
        print(f"  Std: {np.std(per_seed_mean_rewards):.3f}")
        print(f"  Min: {np.min(per_seed_mean_rewards):.3f}")
        print(f"  Max: {np.max(per_seed_mean_rewards):.3f}")
    
    if per_seed_mean_lengths:
        print(f"\nEpisode Length Statistics (Mean per Seed, Aggregated Across Seeds):")
        print(f"  Mean: {np.mean(per_seed_mean_lengths):.1f}")
        print(f"  Std: {np.std(per_seed_mean_lengths):.1f}")
        print(f"  Min: {np.min(per_seed_mean_lengths):.1f}")
        print(f"  Max: {np.max(per_seed_mean_lengths):.1f}")
    
    if per_seed_mean_inference_times:
        print(f"\nInference Time Statistics (Mean per Seed, Aggregated Across Seeds):")
        print(f"  Mean: {np.mean(per_seed_mean_inference_times):.6f}")
        print(f"  Std: {np.std(per_seed_mean_inference_times):.6f}")
        print(f"  Min: {np.min(per_seed_mean_inference_times):.6f}")
        print(f"  Max: {np.max(per_seed_mean_inference_times):.6f}")
    
    if per_seed_mean_successful_rewards:
        print(f"\nSuccessful Episode Reward Statistics (Mean per Seed, Aggregated Across Seeds):")
        print(f"  Mean: {np.mean(per_seed_mean_successful_rewards):.3f}")
        print(f"  Std: {np.std(per_seed_mean_successful_rewards):.3f}")
        print(f"  Min: {np.min(per_seed_mean_successful_rewards):.3f}")
        print(f"  Max: {np.max(per_seed_mean_successful_rewards):.3f}")
    
    if per_seed_mean_successful_lengths:
        print(f"\nSuccessful Episode Length Statistics (Mean per Seed, Aggregated Across Seeds):")
        print(f"  Mean: {np.mean(per_seed_mean_successful_lengths):.1f}")
        print(f"  Std: {np.std(per_seed_mean_successful_lengths):.1f}")
        print(f"  Min: {np.min(per_seed_mean_successful_lengths):.1f}")
        print(f"  Max: {np.max(per_seed_mean_successful_lengths):.1f}")
    
    print("=" * 60)
    
    # Save aggregated summary
    summary = {
        "num_seeds": len(all_logs),
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "overall_success_rate": total_successes / max(total_episodes, 1),
        "success_rate_per_seed": [float(r) for r in all_success_rates],
        "per_seed_mean_rewards": [float(r) for r in per_seed_mean_rewards],
        "per_seed_mean_lengths": [float(l) for l in per_seed_mean_lengths],
        "per_seed_mean_inference_times": [float(t) for t in per_seed_mean_inference_times],
        "success_rate_stats": {
            "mean": float(np.mean(all_success_rates)),
            "std": float(np.std(all_success_rates)),
            "min": float(np.min(all_success_rates)),
            "max": float(np.max(all_success_rates)),
        },
        "reward_stats": {
            "mean": float(np.mean(per_seed_mean_rewards)) if per_seed_mean_rewards else 0.0,
            "std": float(np.std(per_seed_mean_rewards)) if per_seed_mean_rewards else 0.0,
            "min": float(np.min(per_seed_mean_rewards)) if per_seed_mean_rewards else 0.0,
            "max": float(np.max(per_seed_mean_rewards)) if per_seed_mean_rewards else 0.0,
        },
        "length_stats": {
            "mean": float(np.mean(per_seed_mean_lengths)) if per_seed_mean_lengths else 0.0,
            "std": float(np.std(per_seed_mean_lengths)) if per_seed_mean_lengths else 0.0,
            "min": float(np.min(per_seed_mean_lengths)) if per_seed_mean_lengths else 0.0,
            "max": float(np.max(per_seed_mean_lengths)) if per_seed_mean_lengths else 0.0,
        },
        "inference_time_stats": {
            "mean": float(np.mean(per_seed_mean_inference_times)) if per_seed_mean_inference_times else 0.0,
            "std": float(np.std(per_seed_mean_inference_times)) if per_seed_mean_inference_times else 0.0,
            "min": float(np.min(per_seed_mean_inference_times)) if per_seed_mean_inference_times else 0.0,
            "max": float(np.max(per_seed_mean_inference_times)) if per_seed_mean_inference_times else 0.0,
        },
        "per_seed_mean_successful_rewards": [float(r) for r in per_seed_mean_successful_rewards],
        "per_seed_mean_successful_lengths": [float(l) for l in per_seed_mean_successful_lengths],
        "successful_reward_stats": {
            "mean": float(np.mean(per_seed_mean_successful_rewards)) if per_seed_mean_successful_rewards else 0.0,
            "std": float(np.std(per_seed_mean_successful_rewards)) if per_seed_mean_successful_rewards else 0.0,
            "min": float(np.min(per_seed_mean_successful_rewards)) if per_seed_mean_successful_rewards else 0.0,
            "max": float(np.max(per_seed_mean_successful_rewards)) if per_seed_mean_successful_rewards else 0.0,
        },
        "successful_length_stats": {
            "mean": float(np.mean(per_seed_mean_successful_lengths)) if per_seed_mean_successful_lengths else 0.0,
            "std": float(np.std(per_seed_mean_successful_lengths)) if per_seed_mean_successful_lengths else 0.0,
            "min": float(np.min(per_seed_mean_successful_lengths)) if per_seed_mean_successful_lengths else 0.0,
            "max": float(np.max(per_seed_mean_successful_lengths)) if per_seed_mean_successful_lengths else 0.0,
        },
    }
    
    summary_path = output_dir / "aggregated_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAggregated summary saved to: {summary_path}")
    

def main() -> None:
    """Main function to run policy inference in prbench."""
    parser = argparse.ArgumentParser(description="Run policy inference in prbench")
    parser.add_argument(
        "--output-dir", default="data/evaluations_final", help="Directory to save episodes"
    )
    parser.add_argument(
        "--seed", type=int, default=301, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num-seeds", type=int, default=5, help="Number of random seeds to run"
    )
    parser.add_argument(
        "--num-episodes", type=int, default=50, help="Number of episodes to run"
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
    parser.add_argument("--render", action="store_true", default=True, help="Render the environment")
    parser.add_argument("--use-qpos", action="store_true", default=False, help="Use qpos for the policy")
    parser.add_argument("--use-delta-qpos", action="store_true", default=False, help="Use delta qpos for the policy")
    parser.add_argument("--use-env-state", action="store_true", default=False, help="Use env state for the policy")
    parser.add_argument("--remove_velocity", action="store_true", default=False, help="remove velocity from the policy")
    parser.add_argument("--summary_only", action="store_true", default=False, help="only run the summary")
    args = parser.parse_args()

    if args.summary_only:
        # Find the most recent matching directory
        output_base = Path(args.output_dir)
        pattern = f"videos_{args.env_name}_*"
        matching_dirs = sorted(output_base.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if not matching_dirs:
            print(f"No directories matching '{pattern}' found in {output_base}")
            return
        video_parent_dir = matching_dirs[0]
        print(f"Using most recent directory: {video_parent_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_parent_dir = Path(args.output_dir) / f"videos_{args.env_name}_{timestamp}"
        video_parent_dir.mkdir(parents=True, exist_ok=True)
        for seed_idx in range(args.num_seeds):
            run_inference(
                output_dir=video_parent_dir,
                seed=args.seed + seed_idx,
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
                remove_velocity=args.remove_velocity,
            )
    
    run_summary(
        output_dir=video_parent_dir
    )


if __name__ == "__main__":
    main()
