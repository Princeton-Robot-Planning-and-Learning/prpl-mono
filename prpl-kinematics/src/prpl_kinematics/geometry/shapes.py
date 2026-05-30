"""Geometry shapes attached to KinematicTree nodes.

Each shape carries an ``origin`` (the link-frame to shape-frame transform), its
parameters, and an optional ``color`` (RGBA in ``[0, 1]``). Backends turn these
into PyBullet collision/visual shapes; the tree itself never interprets them. A
node may hold several visual and several collision shapes. When ``color`` is
``None`` the backend's default applies (and a mesh keeps any material it ships
with); setting it tints the shape in both the PyBullet and Blender renderers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from spatialmath import SE3

Color = tuple[float, float, float, float]


@dataclass(frozen=True)
class MeshShape:
    """A triangle mesh loaded from ``filename`` (any trimesh-readable format)."""

    filename: str
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    origin: SE3 = field(default_factory=SE3)
    color: Color | None = None


@dataclass(frozen=True)
class BoxShape:
    """An axis-aligned box specified by its full-extent ``size``."""

    size: tuple[float, float, float]
    origin: SE3 = field(default_factory=SE3)
    color: Color | None = None


@dataclass(frozen=True)
class CylinderShape:
    """A cylinder of the given ``radius`` and ``length`` along its local +z."""

    radius: float
    length: float
    origin: SE3 = field(default_factory=SE3)
    color: Color | None = None


@dataclass(frozen=True)
class SphereShape:
    """A sphere of the given ``radius``."""

    radius: float
    origin: SE3 = field(default_factory=SE3)
    color: Color | None = None


Shape = MeshShape | BoxShape | CylinderShape | SphereShape
