"""Convert PRBench demos to RoboMimic HDF5 format."""

import argparse
from pathlib import Path

import cv2 as cv
import h5py  # type: ignore
import numpy as np
from constants import POLICY_IMAGE_HEIGHT, POLICY_IMAGE_WIDTH
from episode_storage import EpisodeReader
from scipy.spatial.transform import Rotation  # type: ignore


def main(
    input_dir: str, output_path: str, args  # pylint: disable=redefined-outer-name
) -> None:
    """Convert PRBench demos to RoboMimic HDF5 format."""
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

            if args.navigation_only:
                max_nav_steps = 0
                for t in range(2, len(reader.observations)):
                    if np.allclose(
                        reader.observations[t]["base_pose"],
                        reader.observations[t + 1]["base_pose"],
                        atol=0.003,
                    ):
                        max_nav_steps = t
                        break
                print("max_nav_steps", max_nav_steps)

            # Extract observations
            observations: dict[str, list[np.ndarray]] = {}
            for i in range(len(reader.observations)):
                obs = reader.observations[i]
                if args.navigation_only:
                    if i > max_nav_steps:
                        break
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
                    observations[k].append(v)

            # Extract actions
            if args.discrete_gripper:
                actions = []
                gripper_pos = 0.0
                for i in range(len(reader.actions)):
                    if args.navigation_only:
                        if i > max_nav_steps:
                            break
                    if i == len(reader.actions) - 1:
                        gripper_pos = 0.0
                    else:
                        # import pdb; pdb.set_trace()
                        if reader.actions[i]["gripper_pos"] == 1.0:
                            gripper_pos = 1.0
                    action = reader.actions[i]
                    actions.append(
                        np.concatenate(
                            (
                                action["base_pose"],
                                action["arm_pos"],
                                Rotation.from_quat(
                                    action["arm_quat"]
                                ).as_rotvec(),  # Convert quat to axis-angle
                                np.array([gripper_pos]),
                            )
                        )
                    )
            elif args.follow_obs:
                actions = []
                gripper_pos = 0.0
                for i in range(len(reader.actions)):
                    if i == len(reader.actions) - 1:
                        actions.append(
                            np.concatenate(
                                (
                                    reader.actions[i]["base_pose"],
                                    reader.actions[i]["arm_pos"],
                                    reader.actions[i]["arm_quat"],
                                    reader.actions[i]["gripper_pos"],
                                )
                            )
                        )
                    else:
                        # import pdb; pdb.set_trace()
                        if reader.actions[i]["gripper_pos"] == 1.0:
                            gripper_pos = 1.0
                        actions.append(
                            np.concatenate(
                                (
                                    observations["base_pose"][i + 1],
                                    observations["arm_pos"][i + 1],
                                    observations["arm_quat"][i + 1],
                                    np.array([gripper_pos]),
                                )
                            )
                        )
            elif args.quaternion:
                actions = [
                    np.concatenate(
                        (
                            action["base_pose"],
                            action["arm_pos"],
                            action["arm_quat"],  # Convert quat to axis-angle
                            action["gripper_pos"],
                        )
                    )
                    for action in reader.actions
                ]
            else:  # Convert quat to axis-angle
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

            if args.predicate:
                predicates = []
                state = "moving"
                for t in range(len(reader.observations)):
                    if state == "moving":
                        if t >= 2 and np.allclose(
                            reader.observations[t]["base_pose"],
                            reader.observations[t + 1]["base_pose"],
                            atol=0.003,
                        ):
                            predicates.append("Grasp the target object.")
                            state = "reach_moving_target"
                        else:
                            predicates.append("Navigate to the target object.")

                    elif state == "reach_moving_target":
                        if reader.observations[t]["gripper_pos"] > 0.1 and (
                            reader.observations[t + 1]["gripper_pos"]
                            - reader.observations[t]["gripper_pos"]
                            < 0.01
                        ):
                            predicates.append("Place the target object.")
                            state = "moving_object"
                        else:
                            predicates.append("Grasp the target object.")
                    elif state == "moving_object":
                        predicates.append("Place the target object.")

            # Write to HDF5
            episode_key = f"demo_{episode_idx}"
            episode_group = data_group.create_group(episode_key)
            for k, v in observations.items():
                episode_group.create_dataset(f"obs/{k}", data=np.array(v))
            # print('actions', actions)
            episode_group.create_dataset("actions", data=np.array(actions))
            if args.language:
                if len(reader.target_object_key) > 0:
                    target_object_key = reader.target_object_key[0]
                    target_object_key = target_object_key.split("_")[0]
                    print("target_object_key", target_object_key)
                    if args.navigation_only:
                        episode_group.create_dataset(
                            "language", data=f"Navigate to the {target_object_key}"
                        )
                    else:
                        episode_group.create_dataset(
                            "language",
                            data=f"Pick the {target_object_key} and place it on the shelf.",  # pylint: disable=line-too-long
                        )
                else:
                    episode_group.create_dataset(
                        "language",
                        data="Pick the target object and place it on the shelf.",
                    )
            if args.predicate:
                assert len(predicates) == len(reader.actions)
                assert len(predicates) == len(reader.observations)
                # print('predicates', predicates)
                episode_group.create_dataset("predicates", data=predicates)


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
    parser.add_argument("--max_episodes", type=int, default=1000000)
    parser.add_argument("--start_episode", type=int, default=0)
    parser.add_argument("--navigation_only", type=bool, default=False)
    args = parser.parse_args()
    main(args.input_dir, args.output_path, args=args)
