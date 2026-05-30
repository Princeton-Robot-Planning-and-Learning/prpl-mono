"""The Renderer interface and backend-agnostic capture/video helpers.

A renderer turns a ``KinematicTree`` plus a configuration into images. Because
rendering only needs each geometry-bearing node's world-frame pose (from the
tree's forward kinematics) and its shapes, every backend consumes the same
inputs: a grasped object (re-parented in the tree) renders with no special
handling. PyBullet and Blender backends both conform to :class:`Renderer`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import imageio.v2 as imageio
import numpy as np

from prpl_kinematics.tree.kinematic_tree import Configuration, KinematicTree
from prpl_kinematics.tree.state import KinematicState

# A light, soft purple used as the default scene background by both renderers.
DEFAULT_BACKGROUND_COLOR = (0.91, 0.87, 0.96)


@dataclass(frozen=True)
class CameraParams:
    """Synthetic-camera settings for image capture, shared across backends."""

    target: tuple[float, float, float] = (0.0, 0.0, 0.5)
    distance: float = 1.5
    yaw: float = 45.0
    pitch: float = -30.0
    width: int = 480
    height: int = 360
    fov: float = 60.0
    near: float = 0.1
    far: float = 100.0


class Renderer(ABC):
    """Renders a KinematicTree's nodes from their forward-kinematics poses."""

    @abstractmethod
    def load(self, tree: KinematicTree) -> None:
        """Prepare the backend to render ``tree``'s geometry."""

    @abstractmethod
    def render(self, config: Configuration) -> None:
        """Set the scene to ``config`` (each node at its FK world pose)."""

    @abstractmethod
    def capture_image(self, camera: CameraParams = CameraParams()) -> np.ndarray:
        """Capture an RGB image ``(H, W, 3)`` ``uint8`` of the current scene."""

    def render_frames(
        self,
        configs: Iterable[Configuration],
        camera: CameraParams = CameraParams(),
    ) -> list[np.ndarray]:
        """Render and capture one frame per configuration.

        Backends with expensive per-frame setup (e.g. an out-of-process Blender)
        override this to render the whole sequence in one batch.
        """
        frames = []
        for config in configs:
            self.render(config)
            frames.append(self.capture_image(camera))
        return frames


def render_configurations(
    renderer: Renderer,
    configs: Iterable[Configuration],
    camera: CameraParams = CameraParams(),
) -> list[np.ndarray]:
    """Render each configuration with ``renderer`` and capture a frame."""
    return renderer.render_frames(configs, camera)


def _structure_key(state: KinematicState) -> tuple[tuple[str, str], ...]:
    """Each node's parent, identifying a state's tree structure (its grasps)."""
    return tuple(sorted((child, parent) for child, (parent, _) in state.edges.items()))


def render_states(
    renderer: Renderer,
    states: Iterable[KinematicState],
    tree: KinematicTree,
    camera: CameraParams = CameraParams(),
) -> list[np.ndarray]:
    """Render a plan of states, restoring each state's structure before capture.

    A plan from a manipulation primitive is a sequence of :class:`KinematicState`
    whose tree structure changes at a grasp (an object is re-parented onto the
    gripper). Consecutive states that share a structure are rendered together in
    one :meth:`Renderer.render_frames` call (one Blender launch for the Blender
    backend); the structure is restored on ``tree`` only when it changes.
    """
    frames: list[np.ndarray] = []
    group: list[KinematicState] = []
    key: tuple[tuple[str, str], ...] | None = None
    for state in states:
        state_key = _structure_key(state)
        if group and state_key != key:
            group[0].apply(tree)
            frames += renderer.render_frames(
                [s.as_configuration() for s in group], camera
            )
            group = []
        group.append(state)
        key = state_key
    if group:
        group[0].apply(tree)
        frames += renderer.render_frames([s.as_configuration() for s in group], camera)
    return frames


def save_video(frames: Sequence[np.ndarray], path: str, fps: int = 20) -> None:
    """Write captured frames to a video file."""
    imageio.mimsave(path, list(frames), fps=fps)
