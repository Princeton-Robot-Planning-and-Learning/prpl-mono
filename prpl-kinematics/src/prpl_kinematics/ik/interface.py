"""The inverse-kinematics interface.

An ``InverseKinematics`` maps a target end-effector pose plus a seed
configuration to a configuration reaching that pose, or ``None``. It is a
``Protocol`` so that any solver -- a generic numerical one, an IKFast-backed
analytic one, or a bespoke per-robot solver -- can be used interchangeably
(e.g. by :func:`~prpl_kinematics.ik.follow.follow_end_effector_path`) without a
shared base class.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from spatialmath import SE3

from prpl_kinematics.tree.kinematic_tree import Configuration


@runtime_checkable
class InverseKinematics(Protocol):
    """A solver for configurations whose end-effector reaches a target pose."""

    def solve(self, target_pose: SE3, seed: Configuration) -> Configuration | None:
        """A configuration reaching ``target_pose``, or ``None`` if none is found.

        ``seed`` provides both the values of any joints outside the solver's
        group and a preference among multiple solutions (the returned one is the
        closest reachable branch).
        """
