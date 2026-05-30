"""Blender rendering backend: high-fidelity images and videos via headless Blender.

``bpy`` is not installable for every Python, so this backend drives a headless
``blender --background`` process instead: it serializes the scene (each node's
shapes and per-frame forward-kinematics world poses, plus the camera) to JSON,
lets :mod:`prpl_kinematics.visualization._blender_script` build the scene once and
render every frame, then reads the PNGs back. Because it consumes the same
FK-driven inputs as the PyBullet backend, a grasped object (re-parented in the
tree) renders with no special handling.

Blender is found via ``$PRPL_BLENDER``, then ``blender`` on the ``PATH``, then the
default macOS app bundle. The Blender process is only spawned when frames are
rendered, so importing this module never requires Blender to be installed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterable

import imageio.v2 as imageio
import numpy as np
from scipy.spatial.transform import Rotation
from spatialmath import SE3

from prpl_kinematics.geometry.shapes import (
    BoxShape,
    CylinderShape,
    MeshShape,
    Shape,
    SphereShape,
)
from prpl_kinematics.meshes import to_pybullet_mesh
from prpl_kinematics.tree.kinematic_tree import Configuration, KinematicTree
from prpl_kinematics.visualization.interface import CameraParams, Renderer

_SCRIPT = os.path.join(os.path.dirname(__file__), "_blender_script.py")
_MACOS_BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
# Formats Blender imports natively; anything else is converted to .obj on the way.
_BLENDER_NATIVE_FORMATS = frozenset(
    {".obj", ".stl", ".dae", ".glb", ".gltf", ".ply", ".fbx"}
)


def _find_blender() -> str:
    """Locate a Blender executable, or raise if none is available."""
    override = os.environ.get("PRPL_BLENDER")
    if override:
        return override
    found = shutil.which("blender")
    if found:
        return found
    if os.path.exists(_MACOS_BLENDER):
        return _MACOS_BLENDER
    raise FileNotFoundError(
        "Blender executable not found; install Blender or set $PRPL_BLENDER."
    )


def _pose_list(pose: SE3) -> list[float]:
    quaternion = Rotation.from_matrix(pose.R).as_quat()  # xyzw
    translation = pose.t
    return [float(v) for v in translation] + [float(v) for v in quaternion]


def _shape_spec(index: int, node: str, shape: Shape) -> dict:
    spec: dict = {"id": index, "node": node, "origin": _pose_list(shape.origin)}
    if shape.color is not None:
        spec["color"] = list(shape.color)
    if isinstance(shape, MeshShape):
        extension = os.path.splitext(shape.filename)[1].lower()
        path = (
            shape.filename
            if extension in _BLENDER_NATIVE_FORMATS
            else to_pybullet_mesh(shape.filename)
        )
        spec.update(kind="mesh", file=path, scale=list(shape.scale))
    elif isinstance(shape, BoxShape):
        spec.update(kind="box", size=list(shape.size))
    elif isinstance(shape, CylinderShape):
        spec.update(kind="cylinder", radius=shape.radius, length=shape.length)
    elif isinstance(shape, SphereShape):
        spec.update(kind="sphere", radius=shape.radius)
    return spec


class BlenderRenderer(Renderer):
    """Renders a KinematicTree through a headless Blender process (Cycles)."""

    def __init__(
        self,
        blender_executable: str | None = None,
        samples: int = 64,
        ground_plane: bool = True,
    ) -> None:
        self._blender = blender_executable or _find_blender()
        self._samples = samples
        self._ground_plane = ground_plane
        self._tree: KinematicTree | None = None
        self._shapes: list[dict] = []
        self._nodes: list[str] = []  # nodes carrying visual shapes
        self._current: Configuration | None = None

    @property
    def blender_executable(self) -> str:
        """Path to the Blender executable this renderer launches."""
        return self._blender

    def load(self, tree: KinematicTree) -> None:
        """Collect every node's visual shapes into render specs."""
        self._tree = tree
        self._shapes = []
        nodes: list[str] = []
        for name, node in tree.nodes.items():
            if node.visuals:
                nodes.append(name)
            for shape in node.visuals:
                self._shapes.append(_shape_spec(len(self._shapes), name, shape))
        self._nodes = nodes

    def render(self, config: Configuration) -> None:
        """Record ``config`` as the scene to capture next."""
        self._current = dict(config)

    def capture_image(self, camera: CameraParams = CameraParams()) -> np.ndarray:
        """Render the most recently set configuration to an RGB image."""
        assert self._current is not None, "call render() before capture_image()"
        return self.render_frames([self._current], camera)[0]

    def render_frames(
        self,
        configs: Iterable[Configuration],
        camera: CameraParams = CameraParams(),
    ) -> list[np.ndarray]:
        """Render all configurations in one Blender launch (scene built once)."""
        assert self._tree is not None, "call load() before rendering"
        config_list = list(configs)
        if not config_list:
            return []
        frames = [
            {
                "poses": {
                    name: _pose_list(self._tree.forward_kinematics(name, config))
                    for name in self._nodes
                }
            }
            for config in config_list
        ]
        with tempfile.TemporaryDirectory() as work_dir:
            job = {
                "samples": self._samples,
                "ground_plane": self._ground_plane,
                "camera": {
                    "target": list(camera.target),
                    "distance": camera.distance,
                    "yaw": camera.yaw,
                    "pitch": camera.pitch,
                    "width": camera.width,
                    "height": camera.height,
                    "fov": camera.fov,
                },
                "output_dir": work_dir,
                "shapes": self._shapes,
                "frames": frames,
            }
            job_path = os.path.join(work_dir, "job.json")
            with open(job_path, "w", encoding="utf-8") as handle:
                json.dump(job, handle)
            self._run_blender(job_path)
            images: list[np.ndarray] = []
            for index in range(len(frames)):
                frame_path = os.path.join(work_dir, f"frame_{index:04d}.png")
                if not os.path.exists(frame_path):
                    raise RuntimeError(
                        f"Blender did not render frame {index}; "
                        f"rendered {len(images)} of {len(frames)}."
                    )
                images.append(np.asarray(imageio.imread(frame_path))[:, :, :3])
            return images

    def _run_blender(self, job_path: str) -> None:
        result = subprocess.run(
            [self._blender, "--background", "--python", _SCRIPT, "--", job_path],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Blender render failed:\n"
                + result.stderr[-2000:]
                + result.stdout[-2000:]
            )
