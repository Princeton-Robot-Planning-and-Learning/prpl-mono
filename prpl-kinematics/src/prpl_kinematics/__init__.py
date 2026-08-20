"""prpl_kinematics: kinematics-only robot modeling, IK, motion planning, and
manipulation primitives built on a general KinematicTree.

The package is engine-agnostic: a ``KinematicTree`` (transforms via spatialmath)
is the single source of truth for forward kinematics, and physics engines such
as PyBullet are used only as pluggable collision/render backends.
"""

from importlib.metadata import version as _version

__version__ = _version("prpl-kinematics")
