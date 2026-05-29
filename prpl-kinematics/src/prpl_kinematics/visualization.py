"""PyBullet-based visualization: a shape-soup renderer, capture, and video.

``PyBulletRenderer`` creates one PyBullet visual body per node shape and, each
frame, positions every body from the tree's forward kinematics. Nothing uses
PyBullet's articulation, so a grasped object (re-parented in the tree) renders
correctly with no special handling.

Meshes are fed to PyBullet's file importer. Formats it reads natively
(``.obj``/``.stl``/``.dae``) are passed straight through; others (e.g. ``.glb``)
are converted to ``.obj`` once via trimesh and cached on disk.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import imageio.v2 as imageio
import numpy as np
import pybullet as p
import trimesh

from prpl_kinematics.geometry.shapes import BoxShape, CylinderShape, MeshShape, Shape
from prpl_kinematics.geometry.transforms import pose_to_pybullet
from prpl_kinematics.tree.kinematic_tree import Configuration, KinematicTree

_NATIVE_MESH_FORMATS = frozenset({".obj", ".stl", ".dae"})
_MESH_CACHE_DIR = os.path.join(tempfile.gettempdir(), "prpl_kinematics_mesh_cache")


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


def to_pybullet_mesh(filename: str) -> str:
    """Return a path to a PyBullet-loadable mesh for ``filename``.

    Native formats are returned unchanged; others are converted to ``.obj`` via
    trimesh and cached on disk by source path and modification time, so the
    conversion is paid at most once per mesh.
    """
    extension = os.path.splitext(filename)[1].lower()
    if extension in _NATIVE_MESH_FORMATS:
        return filename
    os.makedirs(_MESH_CACHE_DIR, exist_ok=True)
    key = f"{os.path.abspath(filename)}:{os.path.getmtime(filename)}"
    cached = os.path.join(
        _MESH_CACHE_DIR, hashlib.md5(key.encode()).hexdigest() + ".obj"
    )
    if not os.path.exists(cached):
        mesh: Any = trimesh.load(filename, force="mesh")
        mesh.export(cached)
    return cached


def _create_visual_shape(physics_client_id: int, shape: Shape) -> int:
    position, orientation = pose_to_pybullet(shape.origin)
    common = {
        "visualFramePosition": position,
        "visualFrameOrientation": orientation,
        "physicsClientId": physics_client_id,
    }
    if isinstance(shape, MeshShape):
        return int(
            p.createVisualShape(
                p.GEOM_MESH,
                fileName=to_pybullet_mesh(shape.filename),
                meshScale=list(shape.scale),
                **common,
            )
        )
    if isinstance(shape, BoxShape):
        half_extents = [shape.size[0] / 2, shape.size[1] / 2, shape.size[2] / 2]
        return int(p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, **common))
    if isinstance(shape, CylinderShape):
        return int(
            p.createVisualShape(
                p.GEOM_CYLINDER, radius=shape.radius, length=shape.length, **common
            )
        )
    return int(p.createVisualShape(p.GEOM_SPHERE, radius=shape.radius, **common))


class PyBulletRenderer:
    """Renders a KinematicTree by positioning per-shape visual bodies via FK."""

    def __init__(self, physics_client_id: int) -> None:
        self._physics_client_id = physics_client_id
        self._tree: KinematicTree | None = None
        self._bodies: list[tuple[int, str]] = []

    @property
    def physics_client_id(self) -> int:
        """The PyBullet client this renderer draws into."""
        return self._physics_client_id

    def load(self, tree: KinematicTree) -> None:
        """Create one visual body per node shape (positioned later by FK)."""
        self._tree = tree
        for name, node in tree.nodes.items():
            for shape in node.visuals:
                visual = _create_visual_shape(self._physics_client_id, shape)
                body = int(
                    p.createMultiBody(
                        baseMass=0,
                        baseVisualShapeIndex=visual,
                        physicsClientId=self._physics_client_id,
                    )
                )
                self._bodies.append((body, name))

    def render(self, config: Configuration) -> None:
        """Move every visual body to its node's world pose under ``config``."""
        assert self._tree is not None, "call load() before render()"
        for body, name in self._bodies:
            position, orientation = pose_to_pybullet(
                self._tree.forward_kinematics(name, config)
            )
            p.resetBasePositionAndOrientation(
                body, position, orientation, physicsClientId=self._physics_client_id
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
    renderer: PyBulletRenderer,
    configs: Iterable[Configuration],
    camera: CameraParams = CameraParams(),
) -> list[np.ndarray]:
    """Render each configuration and capture a frame."""
    frames = []
    for config in configs:
        renderer.render(config)
        frames.append(capture_image(renderer.physics_client_id, camera))
    return frames


def save_video(frames: Sequence[np.ndarray], path: str, fps: int = 20) -> None:
    """Write captured frames to a video file."""
    imageio.mimsave(path, list(frames), fps=fps)
