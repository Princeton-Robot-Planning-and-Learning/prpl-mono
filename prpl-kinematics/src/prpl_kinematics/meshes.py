"""Prepare mesh files for PyBullet's file importer.

PyBullet's file-based mesh loader (used by both the renderer and the collision
checker) reads ``.obj``/``.stl``/``.dae`` natively and has no vertex-count limit,
unlike the programmatic ``createVisualShape(vertices=...)`` path. Other formats
(e.g. ``.glb``) are converted to ``.obj`` once via trimesh and cached on disk.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from typing import Any

import trimesh

_NATIVE_MESH_FORMATS = frozenset({".obj", ".stl", ".dae"})
_MESH_CACHE_DIR = os.path.join(tempfile.gettempdir(), "prpl_kinematics_mesh_cache")


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
