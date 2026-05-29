"""Prepare mesh files for PyBullet's file importer.

PyBullet's mesh loaders have no vertex-count limit (unlike the programmatic
``createVisualShape(vertices=...)`` path), but they accept different formats. The
visual loader reads ``.obj``/``.stl``/``.dae`` natively, while
``createCollisionShape`` only reads ``.obj``/``.stl``. Anything outside the
relevant set (e.g. ``.glb`` for either, or ``.dae`` for collision) is converted
to ``.obj`` once via trimesh and cached on disk.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from typing import Any

import trimesh

_VISUAL_NATIVE_FORMATS = frozenset({".obj", ".stl", ".dae"})
_COLLISION_NATIVE_FORMATS = frozenset({".obj", ".stl"})
_MESH_CACHE_DIR = os.path.join(tempfile.gettempdir(), "prpl_kinematics_mesh_cache")


def to_pybullet_mesh(filename: str, collision: bool = False) -> str:
    """Return a path to a PyBullet-loadable mesh for ``filename``.

    Formats the target loader reads natively are returned unchanged; others are
    converted to ``.obj`` via trimesh and cached on disk by source path and
    modification time, so the conversion is paid at most once per mesh. Set
    ``collision`` for ``createCollisionShape``, which does not read ``.dae``.
    """
    native = _COLLISION_NATIVE_FORMATS if collision else _VISUAL_NATIVE_FORMATS
    extension = os.path.splitext(filename)[1].lower()
    if extension in native:
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
