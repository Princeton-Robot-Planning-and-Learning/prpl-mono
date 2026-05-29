"""Follow an end-effector path by chaining warm-started IK solves.

Each target pose is solved seeded from the previous solution, so a redundant arm tracks
the path on one continuous IK branch -- avoiding the elbow jitter that comes from
solving each waypoint independently.
"""

from __future__ import annotations

from collections.abc import Sequence

from spatialmath import SE3

from prpl_kinematics.ik.numerical import NumericalIK
from prpl_kinematics.tree.kinematic_tree import Configuration


def follow_end_effector_path(
    ik: NumericalIK, poses: Sequence[SE3], seed: Configuration
) -> list[Configuration] | None:
    """Configurations tracking ``poses`` in order, or ``None`` if any fails.

    Each solve is seeded from the previous solution (the first from ``seed``).
    """
    configs: list[Configuration] = []
    current_seed = seed
    for pose in poses:
        solution = ik.solve(pose, current_seed)
        if solution is None:
            return None
        configs.append(solution)
        current_seed = solution
    return configs
