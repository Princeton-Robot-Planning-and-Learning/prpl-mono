"""Policy inference script for running remote policies in prbench environments."""

import argparse
import time

import cv2 as cv
import numpy as np
import prbench
import zmq
from episode_storage import EpisodeWriter
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
    output_dir: str = "data/inference",
    seed: int = 123,
    save: bool = True,
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
):
    """Run policy inference in the prbench environment.

    Args:
        output_dir: Directory to save episode data.
        seed: Random seed for reproducibility.
        save: Whether to save the episode data to disk.
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
    """
    

    successes = 0
    try:
        for episode_idx in range(num_episodes):
            # Create the environment
            render_mode = "rgb_array" if render or save else None
            env = prbench.make(
                f"prbench/{env_name}",
                render_mode=render_mode,
            )

            # Create remote policy
            policy = RemotePolicy(host=policy_host, port=policy_port)

            print(f"\n=== Episode {episode_idx + 1}/{num_episodes} ===")

            # Create episode writer if saving is enabled
            writer = EpisodeWriter(output_dir) if save else None

            # Reset the environment
            episode_seed = seed + episode_idx
            (
                obs,
                _,
            ) = env.reset(
                seed=episode_seed
            )  # type: ignore
            assert isinstance(env.observation_space, ObjectCentricBoxSpace)
            state = env.observation_space.devectorize(obs)

            # Target object for this episode (can be detected or specified)
            if "DynObstruction2D" in env_name or "Motion2D" in env_name or "StickButton2D" in env_name:
                target_object_key = "target_agent"
            elif "Shelf3D" in env_name or "Ground3D" in env_name:
                target_object_key = f"cube{num_cubes - 1}"
            elif "Transport3D" in env_name:
                target_object_key = "box0"
            elif "BaseMotion3D" in env_name:
                target_object_key = "target"
            elif "Motion3D" in env_name:
                target_object_key = "target"
            elif "Obstruction3D" in env_name:
                target_object_key = "target_block"
            else:
                raise ValueError(f"Environment {env_name} not supported")

            # Reset the policy
            policy.reset(target_object_key)  # type: ignore

            start_time = time.time()
            for step_idx in range(max_steps):
                # Enforce desired control frequency
                step_end_time = start_time + step_idx * POLICY_CONTROL_PERIOD
                while time.time() < step_end_time:
                    time.sleep(0.0001)

                # Get robot state
                robot = state.get_object_from_name("robot")

                image = env.unwrapped._object_centric_env.render()
                if show_images:
                    _visualize_image_in_window(image, "overview")

                # Create observation dict for policy
                if use_env_state:
                    if "DynObstruction2D" in env_name:
                        obs_dict = {
                            "robot_state": obs[-24:],
                            "env_state": obs[:-24],
                            "image": image,
                        }
                    elif "Motion2D" in env_name or "StickButton2D" in env_name:
                        obs_dict = {
                            "robot_state": obs[:9],
                            "env_state": obs[9:],
                            "image": image,
                        }
                else:
                    if "DynObstruction2D" in env_name:
                        obs_dict = {
                            "robot_state": obs[-24:],
                            "image": image,
                        }
                    elif "Motion2D" in env_name or "StickButton2D" in env_name:
                        obs_dict = {
                            "robot_state": obs[:9],
                            "image": image,
                        }
                
                
                # Get action from policy
                action_dict = policy.step(obs_dict)

                
                if action_dict is None:
                    action_dict = {
                        "robot_actions": np.zeros(5, dtype=np.float32)
                    }
                
                action = action_dict["robot_actions"]
                if "DynObstruction2D" in env_name:
                    action_min = np.array([-0.0499, -0.0499, -0.065, -0.10, -0.02], dtype=np.float32)
                    action_max = np.array([0.0499, 0.0499, 0.065, 0.10, 0.02], dtype=np.float32)
                elif "Motion2D" in env_name or "StickButton2D" in env_name:
                    action_min = np.array([-0.05, -0.05, -0.196, -0.10, 0.000], dtype=np.float32)
                    action_max = np.array([0.05, 0.05, 0.196, 0.10, 1.000], dtype=np.float32)
                else:
                    raise ValueError(f"Environment {env_name} not supported")
                action = np.clip(action, action_min, action_max)
                print('action', action)

                # Record observation and action before stepping
                if writer is not None:
                    writer.step(obs_dict, action_dict, target_object_key)

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
                        f"  Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}"  # pylint: disable=line-too-long
                    )
                    successes += 1
                    break

            else:
                print(f"Episode reached max steps ({max_steps})")

            print(f"Successes: {successes}")
            print(f"Success rate: {successes / num_episodes}")
            policy.close()  # type: ignore
            env.close()  # type: ignore
            # Save episode data to disk
            if writer is not None and len(writer) > 0:
                writer.flush_async()
                writer.wait_for_flush()
                print(f"Episode saved with {len(writer)} steps")

    finally:
        print(f"Successes: {successes}")
        print(f"Success rate: {successes / num_episodes}")


def main() -> None:
    """Main function to run policy inference in prbench."""
    parser = argparse.ArgumentParser(description="Run policy inference in prbench")
    parser.add_argument(
        "--output-dir", default="data/inference", help="Directory to save episodes"
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
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--use-qpos", action="store_true", default=False, help="Use qpos for the policy")
    parser.add_argument("--use-delta-qpos", action="store_true", default=False, help="Use delta qpos for the policy")
    parser.add_argument("--use-env-state", type=bool, default=True, help="Use env state for the policy")
    args = parser.parse_args()

    run_inference(
        output_dir=args.output_dir,
        seed=args.seed,
        save=args.save,
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
    )


if __name__ == "__main__":
    main()
