#!/usr/bin/env python
"""Dataset utilities for PRBench imitation learning."""

import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
import prbench

# Import LeRobot APIs
from lerobot.datasets.utils import combine_feature_dicts, hw_to_dataset_features
from PIL import Image


def load_expert_pickle(
    expert_data_dir: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load expert data from pickle file."""

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
                env = gym.make(env_id, render_mode="rgb_array")

            # Reset with the same seed
            env.reset(seed=seed)
            rendered = env.render()  # type: ignore
            # Convert RGBA to RGB if needed
            if rendered.shape[-1] == 4:  # type: ignore
                rendered = rendered[:, :, :3]  # type: ignore
            episode_images = [rendered]

            # Execute actions to get images
            for action in actions:
                env.step(action)
                rendered = env.render()
                # Convert RGBA to RGB if needed
                if rendered.shape[-1] == 4:  # type: ignore
                    rendered = rendered[:, :, :3]  # type: ignore
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
        env.close()  # type: ignore

    metadata = {
        "env_name": env_id or "Motion2D",
        "env_type": "geom2d",
        "data_type": "teleoperated",
    }

    return metadata, frames


def to_pil(img: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL image."""
    if isinstance(img, Image.Image):  # type: ignore
        return img  # type: ignore
    if img.dtype != np.uint8:
        # clip + convert
        arr = np.clip(img, 0, 255).astype(np.uint8)
    else:
        arr = img
    return Image.fromarray(arr)


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
    """Group frames by episode."""
    buckets: Dict[int, List[Dict[str, Any]]] = {}
    for fr in frames:
        ep_idx = int(fr.get("episode_index", 0))
        buckets.setdefault(ep_idx, []).append(fr)
    # sort frames within episode by frame_index if present
    for ep_idx in buckets:  # pylint: disable=consider-using-dict-items
        buckets[ep_idx].sort(key=lambda x: int(x.get("frame_index", 0)))
    return dict(sorted(buckets.items(), key=lambda kv: kv[0]))
