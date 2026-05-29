"""Package-wide helpers."""

from __future__ import annotations

from pathlib import Path


def get_assets_path() -> Path:
    """The directory holding bundled robot assets (URDFs and meshes)."""
    return Path(__file__).resolve().parent / "assets"
