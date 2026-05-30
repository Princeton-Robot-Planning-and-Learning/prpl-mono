"""Rendering: a backend-agnostic Renderer over a KinematicTree's FK poses.

``PyBulletRenderer`` is a fast shape-soup preview; ``BlenderRenderer`` produces
high-fidelity images and videos through a headless Blender process. Both consume
the same FK-driven inputs and conform to :class:`Renderer`, so grasped objects
render with no special handling and the same plans target either backend.
"""

from prpl_kinematics.visualization.blender_renderer import BlenderRenderer
from prpl_kinematics.visualization.interface import (
    CameraParams,
    Renderer,
    render_configurations,
    render_states,
    save_video,
)
from prpl_kinematics.visualization.pybullet_renderer import (
    PyBulletRenderer,
    capture_image,
)

__all__ = [
    "BlenderRenderer",
    "CameraParams",
    "PyBulletRenderer",
    "Renderer",
    "capture_image",
    "render_configurations",
    "render_states",
    "save_video",
]
