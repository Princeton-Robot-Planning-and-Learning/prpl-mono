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
      --teleop_data_dir ../kinder/demos/Motion2D-p0 \
      --output_path datasets/demos.hdf5 \
      --render_images

  # For teleoperated demonstrations (state-only, no images):
  python scripts/demos_to_hdf5.py \
      --teleop_data_dir third-party/kinder/demos/Motion2D-p0 \
      --output_path datasets/demos.hdf5
"""

import argparse
from pathlib import Path

import cv2 as cv
import h5py  # type: ignore
import imageio as iio
import numpy as np
from kinder_imitation_learning.dataset import (  # type: ignore
    iter_teleop_episodes,
)


def convert(
    teleop_data_dir: Path | None = None,
    output_path: Path | None = None,
    render_images: bool = False,
    use_dynamic2d: bool = False,
    use_kinematic3d: bool = False,
    use_dynamics3d: bool = False,
    use_pushpull2d: bool = False,
    save_videos: bool = False,
    use_velocity_state: bool = False,
) -> None:
    """Convert expert or teleoperated data to HDF5 format.

    Memory-efficient implementation: processes one episode at a time using a generator,
    writes directly to HDF5, then discards the episode data before loading the next.

    Args:
        teleop_data_dir: Path to teleoperated demo directory
        output_path: Output HDF5 file path
        render_images: If True, render images for teleoperated demos
        use_dynamic2d: If True, use dynamic2d environment
        use_kinematic3d: If True, use kinematic3d environment
        use_dynamics3d: If True, use dynamics3d environment
        use_pushpull2d: If True, use pushpull2d environment
        save_videos: If True, save videos for teleoperated demos
        use_velocity_state: If True, use dynamics3d velocity state
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
            teleop_data_dir,
            render_images=render_images,
            use_kinematic3d=use_kinematic3d,
            use_dynamics3d=use_dynamics3d,
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
            overview_images = []
            wrist_images = []
            base_images = []

            for fr in ep_frames:
                if use_velocity_state and use_dynamics3d:
                    robot_observation = np.array(
                        fr["observation.robot_state"][:11], dtype=np.float32
                    )
                    env_observations = np.array(
                        fr["observation.env_state"], dtype=np.float32
                    )
                else:
                    robot_observation = np.array(
                        fr["observation.robot_state"], dtype=np.float32
                    )
                    env_observations = np.array(
                        fr["observation.env_state"], dtype=np.float32
                    )
                if use_dynamics3d:
                    if use_velocity_state:
                        assert (
                            robot_observation.shape
                            == np.array(
                                fr["observation.state"][-22:-11], dtype=np.float32
                            ).shape
                        )
                    else:
                        assert (
                            robot_observation.shape
                            == np.array(
                                fr["observation.state"][-22:], dtype=np.float32
                            ).shape
                        )
                    assert (
                        env_observations.shape
                        == np.array(
                            fr["observation.state"][:-22], dtype=np.float32
                        ).shape
                    )
                elif use_kinematic3d:
                    assert (
                        robot_observation.shape
                        == np.array(
                            fr["observation.state"][:19], dtype=np.float32
                        ).shape
                    )
                    assert (
                        env_observations.shape
                        == np.array(
                            fr["observation.state"][19:], dtype=np.float32
                        ).shape
                    )
                elif use_pushpull2d:
                    assert (
                        robot_observation.shape
                        == np.array(
                            fr["observation.state"][:24], dtype=np.float32
                        ).shape
                    )
                    assert (
                        env_observations.shape
                        == np.array(
                            fr["observation.state"][24:], dtype=np.float32
                        ).shape
                    )
                elif use_dynamic2d:
                    assert (
                        robot_observation.shape
                        == np.array(
                            fr["observation.state"][-24:], dtype=np.float32
                        ).shape
                    )
                    assert (
                        env_observations.shape
                        == np.array(
                            fr["observation.state"][:-24], dtype=np.float32
                        ).shape
                    )
                else:
                    assert (
                        robot_observation.shape
                        == np.array(fr["observation.state"][:9], dtype=np.float32).shape
                    )
                    assert (
                        env_observations.shape
                        == np.array(fr["observation.state"][9:], dtype=np.float32).shape
                    )
                action = np.array(fr["action"], dtype=np.float32)
                env_states.append(env_observations)
                robot_states.append(robot_observation)
                actions.append(action)

                resize_constant = 224
                # Add image if present
                if use_kinematic3d or use_dynamics3d:
                    overview_image = fr["observation.overview_image"]
                    if isinstance(overview_image, np.ndarray):
                        overview_image = cv.resize(  # pylint: disable=no-member
                            overview_image, (resize_constant, resize_constant)
                        )
                        overview_images.append(overview_image)
                    wrist_image = fr["observation.wrist_image"]
                    if isinstance(wrist_image, np.ndarray):
                        wrist_image = cv.resize(  # pylint: disable=no-member
                            wrist_image, (resize_constant, resize_constant)
                        )
                        wrist_images.append(wrist_image)
                    base_image = fr["observation.base_image"]
                    if isinstance(base_image, np.ndarray):
                        base_image = cv.resize(  # pylint: disable=no-member
                            base_image, (resize_constant, resize_constant)
                        )
                        base_images.append(base_image)
                elif has_images and "observation.image" in fr:
                    image = fr["observation.image"]
                    if isinstance(image, np.ndarray):
                        image = cv.resize(  # pylint: disable=no-member
                            image, (resize_constant, resize_constant)
                        )
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
            if use_kinematic3d or use_dynamics3d:
                episode_group.create_dataset(
                    "obs/overview_image", data=np.array(overview_images, dtype=np.uint8)
                )
                episode_group.create_dataset(
                    "obs/wrist_image", data=np.array(wrist_images, dtype=np.uint8)
                )
                episode_group.create_dataset(
                    "obs/base_image", data=np.array(base_images, dtype=np.uint8)
                )
                if save_videos and overview_images:
                    # Create video output directory next to the HDF5 file
                    video_dir = (
                        output_path.parent
                        / f"videos_{teleop_data_dir.name}"
                        / f"demo_{ep_idx}"
                    )
                    video_dir.mkdir(parents=True, exist_ok=True)

                    # Save videos for each camera view
                    fps = 30
                    overview_video_path = video_dir / "overview.mp4"
                    wrist_video_path = video_dir / "wrist.mp4"
                    base_video_path = video_dir / "base.mp4"

                    iio.mimsave(overview_video_path, overview_images, fps=fps)  # type: ignore # pylint: disable=line-too-long
                    iio.mimsave(wrist_video_path, wrist_images, fps=fps)  # type: ignore
                    iio.mimsave(base_video_path, base_images, fps=fps)  # type: ignore
                    print(f"  Saved videos for episode {ep_idx}")
            elif images:
                episode_group.create_dataset(
                    "obs/image", data=np.array(images, dtype=np.uint8)
                )
                if save_videos and images:
                    # Create video output directory next to the HDF5 file
                    video_dir = (
                        output_path.parent
                        / f"videos_{teleop_data_dir.name}"
                        / f"demo_{ep_idx}"
                    )
                    video_dir.mkdir(parents=True, exist_ok=True)

                    # Save videos for each camera view
                    fps = 30
                    image_video_path = video_dir / "image.mp4"
                    iio.mimsave(image_video_path, images, fps=fps)  # type: ignore
                    print(f"  Saved video for episode {ep_idx}")

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
        "replaying in environment (requires kinder)",
    )
    parser.add_argument(
        "--use_dynamic2d",
        action="store_true",
        help="Use dynamic2d environment",
    )
    parser.add_argument(
        "--use_kinematic3d",
        action="store_true",
        help="Use kinematic3d environment",
    )
    parser.add_argument(
        "--use_dynamics3d",
        action="store_true",
        help="Use dynamics3d environment",
    )
    parser.add_argument(
        "--use_velocity_state",
        action="store_true",
        help="Use dynamics3d velocity state",
    )
    parser.add_argument(
        "--use_pushpull2d",
        action="store_true",
        help="Use dynamicpushpull2d environment",
    )
    parser.add_argument(
        "--save_videos",
        action="store_true",
        help="Save videos for teleoperated demos",
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
        use_kinematic3d=args.use_kinematic3d,
        use_dynamics3d=args.use_dynamics3d,
        use_pushpull2d=args.use_pushpull2d,
        save_videos=args.save_videos,
        use_velocity_state=args.use_velocity_state,
    )


if __name__ == "__main__":
    main()
