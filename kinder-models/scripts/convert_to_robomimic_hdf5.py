"""Convert KinDER demos to RoboMimic HDF5 format."""

import argparse
from pathlib import Path

import cv2 as cv
import h5py  # type: ignore
import numpy as np
from episode_storage import EpisodeReader
from scipy.spatial.transform import Rotation  # type: ignore

from kinder_models.policy_constants import POLICY_IMAGE_HEIGHT, POLICY_IMAGE_WIDTH


def main(
    input_dir: str, output_path: str, args  # pylint: disable=redefined-outer-name
) -> None:
    """Convert KinDER demos to RoboMimic HDF5 format."""
    # Get list of episode dirs
    episode_dirs = sorted(
        [child for child in Path(input_dir).iterdir() if child.is_dir()]
    )

    # Convert to robomimic HDF5 format
    with h5py.File(output_path, "w") as f:
        data_group = f.create_group("data")

        # Iterate through episodes
        for episode_idx in range(
            args.start_episode, args.start_episode + args.max_episodes
        ):
            if episode_idx >= len(episode_dirs):
                break
            episode_dir = episode_dirs[episode_idx]
            reader = EpisodeReader(episode_dir)

            # Extract observations
            observations: dict[str, list[np.ndarray]] = {}
            for i in range(len(reader.observations)):
                obs = reader.observations[i]
                for k, v in obs.items():
                    if v.ndim == 3:
                        # Resize image
                        if args.high_resolution:
                            v = cv.resize(v, (224, 224))  # pylint: disable=no-member
                        else:
                            v = cv.resize(  # pylint: disable=no-member
                                v, (POLICY_IMAGE_WIDTH, POLICY_IMAGE_HEIGHT)
                            )

                    # Append extracted observation
                    if k not in observations:
                        observations[k] = []
                    if args.obs_discrete_gripper:
                        if k == "gripper_pos" and v[0] > 0.01:
                            v[0] = 1.0
                    observations[k].append(v)

            if "arm_qpos" in reader.actions[0].keys():
                actions = [
                    np.concatenate(
                        (
                            action["base_pose"],
                            action["arm_qpos"],
                            action["gripper_pos"],
                        )
                    )
                    for action in reader.actions
                ]
            else:
                actions = [
                    np.concatenate(
                        (
                            action["base_pose"],
                            action["arm_pos"],
                            Rotation.from_quat(
                                action["arm_quat"]
                            ).as_rotvec(),  # Convert quat to axis-angle
                            action["gripper_pos"],
                        )
                    )
                    for action in reader.actions
                ]

            # Write to HDF5
            episode_key = f"demo_{episode_idx}"
            episode_group = data_group.create_group(episode_key)
            for k, v in observations.items():
                episode_group.create_dataset(f"obs/{k}", data=np.array(v))
            # print('actions', actions)
            episode_group.create_dataset("actions", data=np.array(actions))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/demos")
    parser.add_argument("--output-path", default="data/demos.hdf5")
    parser.add_argument("--language", type=bool, default=False)
    parser.add_argument("--predicate", type=bool, default=False)
    parser.add_argument("--quaternion", type=bool, default=False)
    parser.add_argument("--follow_obs", type=bool, default=False)
    parser.add_argument("--high_resolution", type=bool, default=False)
    parser.add_argument("--discrete_gripper", type=bool, default=False)
    parser.add_argument("--obs_discrete_gripper", type=bool, default=False)
    parser.add_argument("--max_episodes", type=int, default=1000000)
    parser.add_argument("--start_episode", type=int, default=0)
    parser.add_argument("--navigation_only", type=bool, default=False)
    args = parser.parse_args()
    main(args.input_dir, args.output_path, args=args)
