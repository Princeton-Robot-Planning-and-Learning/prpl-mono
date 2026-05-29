"""The motion-planner interface.

A ``MotionPlanner`` finds a collision-free path of configurations from a start
to a goal, or returns ``None``. It is a ``Protocol`` so a sampling planner
(``BiRRTPlanner``) and an OMPL-backed one (``OMPLPlanner``) are interchangeable.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from prpl_kinematics.tree.kinematic_tree import Configuration


@runtime_checkable
class MotionPlanner(Protocol):
    """Plans a collision-free path between two configurations."""

    def plan(
        self, start: Configuration, goal: Configuration
    ) -> list[Configuration] | None:
        """A collision-free path from ``start`` to ``goal``, or ``None``.

        Only the planner's configuration-space coordinates vary along the path;
        other joints stay at their ``start`` values.
        """
        raise NotImplementedError
