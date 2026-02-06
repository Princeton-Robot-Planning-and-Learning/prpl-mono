"""Episode storage for KinDER demos."""

import pickle
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2 as cv
import numpy as np

from kinder_models.policy_constants import POLICY_CONTROL_FREQ


def write_frames_to_mp4(frames: list[np.ndarray], mp4_path: Path) -> None:
    """Write frames to MP4 video."""
    height, width, _ = frames[0].shape
    fourcc = cv.VideoWriter_fourcc(*"mp4v")  # type: ignore # pylint: disable=no-member
    out = cv.VideoWriter(  # pylint: disable=no-member
        str(mp4_path),
        fourcc,
        POLICY_CONTROL_FREQ,
        (width, height),  # pylint: disable=no-member
    )
    for frame in frames:
        bgr_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)  # pylint: disable=no-member
        out.write(bgr_frame)
    out.release()


def read_frames_from_mp4(mp4_path: Path) -> list[np.ndarray]:
    """Read frames from MP4 video."""
    cap = cv.VideoCapture(str(mp4_path))  # pylint: disable=no-member
    frames: list[np.ndarray] = []
    while True:
        ret, bgr_frame = cap.read()
        if not ret:
            break
        frames.append(
            cv.cvtColor(bgr_frame, cv.COLOR_BGR2RGB)  # pylint: disable=no-member
        )
    cap.release()
    return frames


class EpisodeWriter:
    """Writer for KinDER demos."""

    def __init__(self, output_dir: str):
        """Initialize EpisodeWriter."""
        self.output_dir = Path(output_dir)
        self.episode_dir = self.output_dir / datetime.now().strftime("%Y%m%dT%H%M%S%f")
        assert not self.episode_dir.exists()

        # Episode data
        self.timestamps: list[float] = []
        self.observations: list[dict] = []
        self.actions: list[dict] = []
        self.target_object_key: list[str] = []

        # Write to disk in separate thread to avoid blocking main thread
        self.flush_thread = None

    def step(
        self,
        obs: dict[str, np.ndarray],
        action: dict[str, np.ndarray],
        target_object_key: str = "cube1",
    ):
        """Step the EpisodeWriter."""
        self.timestamps.append(time.time())
        self.observations.append(obs)
        self.actions.append(action)
        self.target_object_key.append(target_object_key)

    def __len__(self) -> int:
        """Get the length of the EpisodeWriter."""
        return len(self.observations)

    def _flush(self) -> None:
        """Flush the EpisodeWriter."""
        assert len(self) > 0

        # Create episode dir
        self.episode_dir.mkdir(parents=True)

        # Extract image observations
        frames_dict: dict[str, list[np.ndarray]] = {}
        for obs in self.observations:
            for k, v in obs.items():
                if v.ndim == 3:
                    if k not in frames_dict:
                        frames_dict[k] = []
                    frames_dict[k].append(v)
                    obs[k] = None

        # Write images as MP4 videos
        for k, frames in frames_dict.items():
            mp4_path = self.episode_dir / f"{k}.mp4"
            write_frames_to_mp4(frames, mp4_path)

        # Write rest of episode data
        with open(
            self.episode_dir / "data.pkl", "wb"
        ) as f:  # Note: Not secure. Only unpickle data you trust.
            pickle.dump(
                {
                    "timestamps": self.timestamps,
                    "observations": self.observations,
                    "actions": self.actions,
                    "target_object_key": self.target_object_key,
                },
                f,
            )
        num_episodes = len(
            [child for child in self.output_dir.iterdir() if child.is_dir()]
        )
        print(f"Saved episode to {self.episode_dir} ({num_episodes} total)")

    def flush_async(self) -> None:
        """Flush the EpisodeWriter asynchronously."""
        print("Saving successful episode to disk...")
        # Note: Disk writes may cause latency spikes in low-level controllers
        self.flush_thread = threading.Thread(target=self._flush, daemon=True)  # type: ignore # pylint: disable=line-too-long
        self.flush_thread.start()  # type: ignore

    def wait_for_flush(self) -> None:
        """Wait for the EpisodeWriter to flush."""
        if self.flush_thread is not None:
            self.flush_thread.join()  # type: ignore
            self.flush_thread = None


class EpisodeReader:
    """Reader for KinDER demos."""

    def __init__(self, episode_dir: Path):
        """Initialize EpisodeReader."""
        self.episode_dir = episode_dir

        # Load data
        # isort: off
        with open(
            episode_dir / "data.pkl", "rb"
        ) as f:  # Note: Not secure. Only unpickle data you trust.
            data = pickle.load(f)
        # isort: on
        self.timestamps = data["timestamps"]
        self.observations = data["observations"]
        self.actions = data["actions"]
        self.target_object_key = data["target_object_key"]
        assert len(self.timestamps) > 0
        assert len(self.timestamps) == len(self.observations) == len(self.actions)

        # Restore image observations from MP4 videos
        frames_dict = {}
        for step_idx, obs in enumerate(self.observations):
            for k, v in obs.items():
                if v is None:  # Images are stored as MP4 videos
                    # Load images from MP4 file
                    if k not in frames_dict:
                        mp4_path = episode_dir / f"{k}.mp4"  # type: ignore
                        frames_dict[k] = read_frames_from_mp4(mp4_path)

                    # Restore image for current step
                    obs[k] = frames_dict[k][step_idx]  # np.uint8

    def __len__(self) -> int:
        """Get the length of the EpisodeReader."""
        return len(self.observations)
