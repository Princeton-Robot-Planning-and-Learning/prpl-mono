#!/usr/bin/env python
"""Convert expert pickle data to a LeRobot v3.0 dataset (file-based Parquet) using
LeRobot's dataset API, mirroring PushT's structure so it works with
train_lerobot_direct.py locally (no Hub required).

This script will:
- Create `meta/info.json` with features + defaults
- Write `data/chunk-000/file-000.parquet` with frames (images embedded)
- Write `meta/tasks.parquet` (index = task name, column = task_index)
- Write `meta/episodes/chunk-000/file-000.parquet` with episode ranges and data file refs
- Write `meta/stats.json`

Usage:
  # For expert data (with images):
  python scripts/convert_expert_to_lerobot_v3.py \
      --expert_data_dir expert_data/motion2d_p0_20251008_105219 \
      --output_dir datasets/motion2d_lerobot_v3 \
      --repo_id motion2d_expert \
      --fps 10

  # For teleoperated demonstrations (with rendered images):
  python scripts/convert_expert_to_lerobot_v3.py \
      --teleop_data_dir ../prbench/demos/Motion2D-p0 \
      --output_dir datasets/motion2d_teleop_v3 \
      --repo_id motion2d_teleop \
      --fps 10 \
      --render_images

  # For teleoperated demonstrations (state-only, no images):
  python scripts/convert_expert_to_lerobot_v3.py \
      --teleop_data_dir third-party/prbench/demos/Motion2D-p0 \
      --output_dir datasets/motion2d_teleop_v3 \
      --repo_id motion2d_teleop \
      --fps 10
"""

import argparse
from pathlib import Path

import numpy as np

# Import LeRobot APIs
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from prbench_imitation_learning.dataset import (
    build_features,
    group_by_episode,
    infer_shapes,
    load_expert_pickle,
    load_teleop_demonstrations,
)


def convert(
    expert_data_dir: Path = None,  # type: ignore
    teleop_data_dir: Path = None,  # type: ignore
    output_dir: Path = None,  # type: ignore
    repo_id: str = None,  # type: ignore
    fps: int = 10,
    render_images: bool = False,
) -> None:
    """Convert expert or teleoperated data to LeRobot format.

    Args:
        expert_data_dir: Path to expert data directory (with images)
        teleop_data_dir: Path to teleoperated demo directory
        output_dir: Output directory for LeRobot dataset
        repo_id: Repository ID for the dataset
        fps: Frames per second
        render_images: If True, render images for teleoperated demos
    """
    # Load data based on input type
    if expert_data_dir is not None:
        metadata, frames = load_expert_pickle(expert_data_dir)
        has_images = True
    elif teleop_data_dir is not None:  # type: ignore
        metadata, frames = load_teleop_demonstrations(
            teleop_data_dir, render_images=render_images
        )
        has_images = render_images
    else:
        raise ValueError("Either expert_data_dir or teleop_data_dir must be provided")

    # Infer shapes
    state_dim, action_dim, img_shape = infer_shapes(frames)

    # Build features dict
    features = build_features(state_dim, action_dim, img_shape)

    # Create dataset structure using LeRobot API (ensures perfect v3.0 compliance)
    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=output_dir,
        robot_type=metadata.get("env_name", metadata.get("env_type", "geom2d")),
        use_videos=False,
    )

    # Map frames by episode
    episodes = group_by_episode(frames)

    # Derive a task name
    env_name = str(metadata.get("env_name", "motion2d")).lower()
    task_name = env_name.replace("/", "_")
    if not task_name:
        task_name = "geom2d_task"

    # Write episodes
    total_frames = 0
    for _, ep_frames in episodes.items():
        # For each frame in the episode, add to buffer
        for fr in ep_frames:
            obs_state = np.array(fr["observation.state"], dtype=np.float32)
            action = np.array(fr["action"], dtype=np.float32)

            frame = {
                # special field required (not in features)
                "task": task_name,
                # features
                "observation.state": obs_state,
                "action": action,
                # Do NOT include 'timestamp' here; LeRobot will infer it automatically
            }

            # Add image if present
            if has_images and "observation.image" in fr:
                image = fr["observation.image"]
                # image can be np array (H,W,C) uint8; pass PIL or numpy
                if isinstance(image, np.ndarray):
                    img_val = (
                        image  # LeRobot accepts np ndarray; will be embedded later
                    )
                else:
                    img_val = image
                frame["observation.images.cam0"] = img_val

            ds.add_frame(frame)

        # save episode (writes data parquet, updates meta, tasks, stats, episodes)
        ds.save_episode()
        total_frames += len(ep_frames)

    # Write a minimal README on the Hub card structure (optional locally)
    # Not needed for local training.

    print("\nConversion complete!")
    print(f"Output root: {output_dir}")
    print(f"Total episodes: {len(episodes)}")
    print(f"Total frames: {total_frames}")
    print("Structure:")
    print(f"  - {output_dir}/meta/info.json")
    print(f"  - {output_dir}/meta/tasks.parquet")
    print(
        f"  - {output_dir}/meta/episodes/chunk-000/file-000.parquet (and possibly more)"
    )
    print(f"  - {output_dir}/data/chunk-000/file-000.parquet (and possibly more)")


def main() -> None:
    """Main function to convert expert demos to LeRobot v3.0 file-based dataset."""
    parser = argparse.ArgumentParser(
        description="Convert expert pickle or teleoperated demos "
        "to LeRobot v3.0 file-based dataset"
    )
    parser.add_argument(
        "--expert_data_dir",
        type=str,
        default=None,
        help="Directory containing expert dataset.pkl (with images)",
    )
    parser.add_argument(
        "--teleop_data_dir",
        type=str,
        default=None,
        help="Directory containing teleoperated demonstrations (state-only)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output dataset root directory"
    )
    parser.add_argument(
        "--repo_id", type=str, default="motion2d_expert", help="Local dataset repo_id"
    )
    parser.add_argument(
        "--fps", type=int, default=10, help="Frames per second for timestamps"
    )
    parser.add_argument(
        "--render_images",
        action="store_true",
        help="For teleoperated demos: render images by "
        "replaying in environment (requires prbench)",
    )
    args = parser.parse_args()

    # Validate inputs
    if args.expert_data_dir is None and args.teleop_data_dir is None:
        parser.error("Either --expert_data_dir or --teleop_data_dir must be provided")

    if args.expert_data_dir is not None and args.teleop_data_dir is not None:
        parser.error("Cannot specify both --expert_data_dir and --teleop_data_dir")

    if args.render_images and args.expert_data_dir is not None:
        print(
            "Warning: --render_images has no effect for "
            "expert data (images already included)"
        )

    expert_dir = Path(args.expert_data_dir) if args.expert_data_dir else None
    teleop_dir = Path(args.teleop_data_dir) if args.teleop_data_dir else None
    out_dir = Path(args.output_dir)

    if out_dir.exists():
        # Avoid accidental overwrite of existing datasets
        raise FileExistsError(f"Output directory already exists: {out_dir}")

    convert(
        expert_data_dir=expert_dir,  # type: ignore
        teleop_data_dir=teleop_dir,  # type: ignore
        output_dir=out_dir,
        repo_id=args.repo_id,
        fps=args.fps,
        render_images=args.render_images,
    )

    print("\nTo train locally with this dataset, run:")
    print(
        " ".join(
            [
                "python scripts/train_lerobot_direct.py",
                f"--dataset.repo_id={args.repo_id}",
                f"--dataset.root={args.output_dir}",
                "--policy.type=diffusion",
                "--policy.repo_id=yixuanh/motion2d_policy",
                "--output_dir=outputs/expert_training",
                "--steps=50000",
                "--eval_freq=10000",
                "--save_freq=10000",
                "--policy.device=cuda",
                "--policy.push_to_hub=false",
            ]
        )
    )


if __name__ == "__main__":
    main()
