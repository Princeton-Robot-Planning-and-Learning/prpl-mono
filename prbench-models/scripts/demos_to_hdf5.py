#!/usr/bin/env python
"""Convert expert pickle data to HDF5 format for RoboMimic-style training.

This script will create an HDF5 file with the following structure:
  data/
    demo_0/
      observation   - state observations (N, state_dim)
      action        - actions (N, action_dim)
      image         - images (N, H, W, C) [optional]
    demo_1/
      ...

Usage:
  # For expert data (with images):

  # For teleoperated demonstrations (with rendered images):
  python scripts/demos_to_hdf5.py \
      --teleop_data_dir ../prbench/demos/Motion2D-p0 \
      --output_path datasets/demos.hdf5 \
      --render_images

  # For teleoperated demonstrations (state-only, no images):
  python scripts/demos_to_hdf5.py \
      --teleop_data_dir third-party/prbench/demos/Motion2D-p0 \
      --output_path datasets/demos.hdf5
"""

import argparse
from pathlib import Path

import h5py  # type: ignore
import numpy as np
import cv2 as cv

from prbench_imitation_learning.dataset import (
    iter_teleop_episodes,
)


def convert(
    teleop_data_dir: Path | None = None,
    output_path: Path | None = None,
    render_images: bool = False,
    use_dynamic2d: bool = False,
    use_pushpull2d: bool = False,
) -> None:
    """Convert expert or teleoperated data to HDF5 format.

    Memory-efficient implementation: processes one episode at a time using a generator,
    writes directly to HDF5, then discards the episode data before loading the next.

    Args:
        teleop_data_dir: Path to teleoperated demo directory
        output_path: Output HDF5 file path
        render_images: If True, render images for teleoperated demos
        use_dynamic2d: If True, use dynamic2d environment
        use_pushpull2d: If True, use pushpull2d environment
    """
    if teleop_data_dir is None:
        raise ValueError("teleop_data_dir must be provided")

    assert output_path is not None
    has_images = render_images

    # Create HDF5 file and write incrementally
    with h5py.File(output_path, "w") as f:
        data_group = f.create_group("data")

        total_frames = 0
        total_episodes = 0
        metadata_written = False

        # Iterate over episodes one at a time (memory-efficient)
        for ep_idx, ep_frames, metadata in iter_teleop_episodes(
            teleop_data_dir, render_images=render_images
        ):
            # Write metadata once (from first episode)
            if not metadata_written:
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float)):
                        data_group.attrs[key] = value
                metadata_written = True

            # Create episode group
            episode_key = f"demo_{ep_idx}"
            episode_group = data_group.create_group(episode_key)

            # Process frames for this episode
            env_states = []
            robot_states = []
            actions = []
            images = []

            for fr in ep_frames:
                if use_pushpull2d:
                    robot_observation = np.array(fr["observation.state"][:24], dtype=np.float32)
                    env_observations = np.array(fr["observation.state"][24:], dtype=np.float32)
                elif use_dynamic2d:
                    robot_observation = np.array(fr["observation.state"][-24:], dtype=np.float32)
                    env_observations = np.array(fr["observation.state"][:-24], dtype=np.float32)
                else:
                    robot_observation = np.array(fr["observation.state"][:9], dtype=np.float32)
                    env_observations = np.array(fr["observation.state"][9:], dtype=np.float32)
                action = np.array(fr["action"], dtype=np.float32)
                env_states.append(env_observations)
                robot_states.append(robot_observation)
                actions.append(action)

                # Add image if present
                if has_images and "observation.image" in fr:
                    image = fr["observation.image"]
                    if isinstance(image, np.ndarray):
                        image = cv.resize(image, (224, 224))
                        images.append(image)

            # Write datasets for this episode
            episode_group.create_dataset(
                "obs/robot_state", data=np.array(robot_states, dtype=np.float32)
            )
            episode_group.create_dataset(
                "obs/env_state", data=np.array(env_states, dtype=np.float32)
            )
            episode_group.create_dataset(
                "actions", data=np.array(actions, dtype=np.float32)
            )

            # Write images if present
            if images:
                episode_group.create_dataset(
                    "obs/image", data=np.array(images, dtype=np.uint8)
                )

            # Store episode length as attribute
            episode_group.attrs["num_frames"] = len(ep_frames)
            total_frames += len(ep_frames)
            total_episodes += 1

            # Clear episode data to free memory immediately
            del env_states, robot_states, actions, images, ep_frames

        # Store total counts as attributes
        data_group.attrs["total_episodes"] = total_episodes
        data_group.attrs["total_frames"] = total_frames

    print("\nConversion complete!")
    print(f"Output file: {output_path}")
    print(f"Total episodes: {total_episodes}")
    print(f"Total frames: {total_frames}")
    print("\nHDF5 structure:")
    print("  data/")
    print("    demo_0/")
    print("      obs/robot_state  (N, robot_state_dim)")
    print("      obs/env_state    (N, env_state_dim)")
    print("      actions          (N, action_dim)")
    if has_images:
        print("      obs/image        (N, H, W, C)")
    print("    demo_1/")
    print("      ...")


def main() -> None:
    """Main function to convert expert demos to HDF5 format."""
    parser = argparse.ArgumentParser(
        description="Convert expert pickle or teleoperated demos to HDF5 format"
    )
    parser.add_argument(
        "--teleop_data_dir",
        type=str,
        default=None,
        help="Directory containing teleoperated demonstrations (state-only)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output HDF5 file path (e.g., datasets/demos.hdf5)",
    )
    parser.add_argument(
        "--render_images",
        action="store_true",
        help="For teleoperated demos: render images by "
        "replaying in environment (requires prbench)",
    )
    parser.add_argument(
        "--use_dynamic2d",
        action="store_true",
        help="Use dynamic2d environment",
    )
    parser.add_argument(
        "--use_pushpull2d",
        action="store_true",
        help="Use dynamicpushpull2d environment",
    )
    args = parser.parse_args()

    # Validate inputs
    if args.teleop_data_dir is None:
        parser.error("--teleop_data_dir must be provided")

    teleop_dir = Path(args.teleop_data_dir) if args.teleop_data_dir else None
    out_path = Path(args.output_path)

    # Create parent directory if needed
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"Warning: Output file already exists: {out_path}")
        print("Overwriting...")

    convert(
        teleop_data_dir=teleop_dir,
        output_path=out_path,
        render_images=args.render_images,
        use_dynamic2d=args.use_dynamic2d,
        use_pushpull2d=args.use_pushpull2d,
    )


if __name__ == "__main__":
    main()
