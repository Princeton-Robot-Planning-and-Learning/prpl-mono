"""PyBullet-based visualization: capture frames and save videos.

A lightweight viewer that drives PyBullet's own articulation to render a
``KinematicTree`` configuration, so motion can be reviewed as video. The
correctness of the tree's kinematics is established separately by
forward-kinematics tests, not by this viewer.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pybullet as p

from prpl_kinematics.tree.kinematic_tree import Configuration


@dataclass(frozen=True)
class CameraParams:
    """Synthetic-camera settings for image capture."""

    target: tuple[float, float, float] = (0.0, 0.0, 0.5)
    distance: float = 1.5
    yaw: float = 45.0
    pitch: float = -30.0
    width: int = 480
    height: int = 360
    fov: float = 60.0
    near: float = 0.1
    far: float = 100.0


def load_urdf_for_rendering(
    physics_client_id: int, path: Path | str, fixed_base: bool = True
) -> tuple[int, dict[str, int]]:
    """Load a URDF into PyBullet for rendering.

    Returns the body id and a map from joint name to PyBullet joint index.
    """
    body = p.loadURDF(
        str(path), useFixedBase=fixed_base, physicsClientId=physics_client_id
    )
    joint_index: dict[str, int] = {}
    for i in range(p.getNumJoints(body, physicsClientId=physics_client_id)):
        info = p.getJointInfo(body, i, physicsClientId=physics_client_id)
        joint_index[info[1].decode()] = i
    return body, joint_index


def apply_configuration(
    physics_client_id: int,
    body: int,
    joint_index: dict[str, int],
    config: Configuration,
) -> None:
    """Reset the PyBullet joint states for the 1-DOF joints named in ``config``."""
    for name, values in config.items():
        if name in joint_index and len(values) == 1:
            p.resetJointState(
                body, joint_index[name], values[0], physicsClientId=physics_client_id
            )


def capture_image(
    physics_client_id: int, camera: CameraParams = CameraParams()
) -> np.ndarray:
    """Capture an RGB image (H, W, 3) ``uint8`` from a synthetic camera."""
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=camera.target,
        distance=camera.distance,
        yaw=camera.yaw,
        pitch=camera.pitch,
        roll=0.0,
        upAxisIndex=2,
    )
    projection = p.computeProjectionMatrixFOV(
        fov=camera.fov,
        aspect=camera.width / camera.height,
        nearVal=camera.near,
        farVal=camera.far,
    )
    _, _, rgba, _, _ = p.getCameraImage(
        camera.width,
        camera.height,
        viewMatrix=view,
        projectionMatrix=projection,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=physics_client_id,
    )
    image = np.reshape(
        np.asarray(rgba, dtype=np.uint8), (camera.height, camera.width, 4)
    )
    return image[:, :, :3]


def render_configurations(
    physics_client_id: int,
    body: int,
    joint_index: dict[str, int],
    configs: Iterable[Configuration],
    camera: CameraParams = CameraParams(),
) -> list[np.ndarray]:
    """Apply each configuration in turn and capture a frame."""
    frames = []
    for config in configs:
        apply_configuration(physics_client_id, body, joint_index, config)
        frames.append(capture_image(physics_client_id, camera))
    return frames


def save_video(frames: Sequence[np.ndarray], path: Path | str, fps: int = 20) -> None:
    """Write captured frames to a video file."""
    imageio.mimsave(str(path), list(frames), fps=fps)
