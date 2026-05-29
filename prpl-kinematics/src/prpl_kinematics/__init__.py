"""prpl_kinematics: kinematics-only robot modeling, IK, motion planning, and
manipulation primitives built on a general KinematicTree.

The package is engine-agnostic: a ``KinematicTree`` (transforms via spatialmath)
is the single source of truth for forward kinematics, and physics engines such
as PyBullet are used only as pluggable collision/render backends.
"""

__version__ = "0.0.1"
