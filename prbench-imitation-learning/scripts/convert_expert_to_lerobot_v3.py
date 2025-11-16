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
      --teleop_data_dir third-party/prbench/demos/Motion2D-p0 \
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
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Import LeRobot APIs
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import combine_feature_dicts, hw_to_dataset_features
from PIL import Image as PILImage


def load_expert_pickle(
    expert_data_dir: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    import pickle

    pkl_path = expert_data_dir / "dataset.pkl"
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Expected keys based on prior usage
    metadata = data.get("metadata", {})
    episodes_or_frames = data.get("episodes")
    if episodes_or_frames is None:
        # fallback
        episodes_or_frames = data.get("frames")
    if episodes_or_frames is None:
        raise ValueError("dataset.pkl missing 'episodes' or 'frames' key")

    return metadata, episodes_or_frames


def load_teleop_demonstrations(
    teleop_data_dir: Path,
    render_images: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load teleoperated demonstrations from individual episode pickle files.

    Expected structure:
    teleop_data_dir/
        0/
            <timestamp>.p
        1/
            <timestamp>.p
        ...

    Each pickle file contains:
        - env_id: str
        - seed: int
        - observations: List[np.ndarray]  # state vectors
        - actions: List[np.ndarray]
        - rewards: List[float]
        - terminated: bool
        - truncated: bool

    Args:
        teleop_data_dir: Path to directory with demonstrations
        render_images: If True, replay episodes in environment to generate images

    Returns:
        metadata: Dict with env info
        frames: List of frame dicts with keys:
            - observation.state: np.ndarray
            - action: np.ndarray
            - observation.image: np.ndarray (if render_images=True)
            - episode_index: int
            - frame_index: int
    """
    import pickle
    import sys

    # Find all episode directories (numeric subdirectories)
    episode_dirs = sorted(
        [d for d in teleop_data_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )

    if not episode_dirs:
        raise ValueError(f"No episode directories found in {teleop_data_dir}")

    frames = []
    env_id = None
    env = None

    # Setup environment if rendering images
    if render_images:
        # Add prbench src to path if not already there
        prbench_root = teleop_data_dir.parent.parent.parent
        prbench_src = prbench_root / "src"
        if str(prbench_src) not in sys.path:
            sys.path.insert(0, str(prbench_src))

        try:
            import gymnasium as gym
            import prbench

            # Register all prbench environments
            prbench.register_all_environments()
        except ImportError as e:
            raise ImportError(
                f"Failed to import prbench/gymnasium for rendering: {e}\n"
                f"Tried adding {prbench_src} to path. Make sure prbench is installed."
            ) from e

    for ep_idx, ep_dir in enumerate(episode_dirs):
        # Find the pickle file in this episode directory
        pickle_files = list(ep_dir.glob("*.p"))
        if not pickle_files:
            print(f"Warning: No pickle file found in {ep_dir}, skipping")
            continue

        pkl_path = pickle_files[0]
        with open(pkl_path, "rb") as f:
            ep_data = pickle.load(f)

        if env_id is None:
            env_id = ep_data.get("env_id", "Motion2D-p0")

        observations = ep_data["observations"]
        actions = ep_data["actions"]
        seed = ep_data.get("seed", 0)

        # Replay episode to generate images if requested
        episode_images = None
        if render_images:
            if env is None:
                import gymnasium as gym

                env = gym.make(env_id, render_mode="rgb_array")

            # Reset with the same seed
            env.reset(seed=seed)
            rendered = env.render()
            # Convert RGBA to RGB if needed
            if rendered.shape[-1] == 4:
                rendered = rendered[:, :, :3]
            episode_images = [rendered]

            # Execute actions to get images
            for action in actions:
                env.step(action)
                rendered = env.render()
                # Convert RGBA to RGB if needed
                if rendered.shape[-1] == 4:
                    rendered = rendered[:, :, :3]
                episode_images.append(rendered)

        # Create frames (note: len(actions) == len(observations) - 1 typically)
        for frame_idx, (obs, act) in enumerate(zip(observations[:-1], actions)):
            frame = {
                "observation.state": obs,
                "action": act,
                "episode_index": ep_idx,
                "frame_index": frame_idx,
            }

            # Add image if rendered
            if episode_images is not None and frame_idx < len(episode_images):
                frame["observation.image"] = episode_images[frame_idx]

            frames.append(frame)

        if (ep_idx + 1) % 10 == 0:
            print(f"Loaded {ep_idx + 1}/{len(episode_dirs)} episodes...")

    if env is not None:
        env.close()

    metadata = {
        "env_name": env_id or "Motion2D",
        "env_type": "geom2d",
        "data_type": "teleoperated",
    }

    return metadata, frames


def to_pil(img: np.ndarray) -> PILImage:
    if isinstance(img, PILImage):
        return img
    if img.dtype != np.uint8:
        # clip + convert
        arr = np.clip(img, 0, 255).astype(np.uint8)
    else:
        arr = img
    return PILImage.fromarray(arr)


def infer_shapes(frames: List[Dict[str, Any]]) -> Tuple[int, int, Any]:
    """Infer state_dim, action_dim, and img_shape (or None if no images)."""
    # Assume frames contain np arrays
    for fr in frames:
        if "observation.state" in fr and "action" in fr:
            state_dim = int(np.array(fr["observation.state"]).shape[0])
            action_dim = int(np.array(fr["action"]).shape[0])

            # Check if images are present
            if "observation.image" in fr:
                img_shape = tuple(np.array(fr["observation.image"]).shape)
                return state_dim, action_dim, img_shape  # (H, W, C)
            else:
                return state_dim, action_dim, None  # No images
    raise ValueError("Could not infer shapes from frames; expected keys missing.")


def build_features(
    state_dim: int, action_dim: int, img_shape: Any = None
) -> Dict[str, Dict]:
    """Build features dict for LeRobot dataset.

    Args:
        state_dim: Dimension of state vector
        action_dim: Dimension of action vector
        img_shape: Image shape (H, W, C) or None if no images
    """
    # Build observation features (state + optional image)
    obs_hw = {f"s{i}": float for i in range(state_dim)}

    # Add a single camera if images are present
    if img_shape is not None:
        obs_hw.update({"cam0": img_shape})
        obs_feats = hw_to_dataset_features(
            obs_hw, prefix="observation", use_video=False
        )
    else:
        obs_feats = hw_to_dataset_features(obs_hw, prefix="observation")

    # Build action features
    act_hw = {f"a{i}": float for i in range(action_dim)}
    act_feats = hw_to_dataset_features(act_hw, prefix="action")

    features = combine_feature_dicts(obs_feats, act_feats)
    return features


def group_by_episode(frames: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    buckets: Dict[int, List[Dict[str, Any]]] = {}
    for fr in frames:
        ep_idx = int(fr.get("episode_index", 0))
        buckets.setdefault(ep_idx, []).append(fr)
    # sort frames within episode by frame_index if present
    for ep_idx in buckets:
        buckets[ep_idx].sort(key=lambda x: int(x.get("frame_index", 0)))
    return dict(sorted(buckets.items(), key=lambda kv: kv[0]))


def convert(
    expert_data_dir: Path = None,
    teleop_data_dir: Path = None,
    output_dir: Path = None,
    repo_id: str = None,
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
    elif teleop_data_dir is not None:
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
    for ep_idx, ep_frames in episodes.items():
        # For each frame in the episode, add to buffer
        for i, fr in enumerate(ep_frames):
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


def main():
    parser = argparse.ArgumentParser(
        description="Convert expert pickle or teleoperated demos to LeRobot v3.0 file-based dataset"
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
        help="For teleoperated demos: render images by replaying in environment (requires prbench)",
    )
    args = parser.parse_args()

    # Validate inputs
    if args.expert_data_dir is None and args.teleop_data_dir is None:
        parser.error("Either --expert_data_dir or --teleop_data_dir must be provided")

    if args.expert_data_dir is not None and args.teleop_data_dir is not None:
        parser.error("Cannot specify both --expert_data_dir and --teleop_data_dir")

    if args.render_images and args.expert_data_dir is not None:
        print(
            "Warning: --render_images has no effect for expert data (images already included)"
        )

    expert_dir = Path(args.expert_data_dir) if args.expert_data_dir else None
    teleop_dir = Path(args.teleop_data_dir) if args.teleop_data_dir else None
    out_dir = Path(args.output_dir)

    if out_dir.exists():
        # Avoid accidental overwrite of existing datasets
        raise FileExistsError(f"Output directory already exists: {out_dir}")

    convert(
        expert_data_dir=expert_dir,
        teleop_data_dir=teleop_dir,
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
