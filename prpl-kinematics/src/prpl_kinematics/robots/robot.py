"""A robot as composition over a KinematicTree.

A ``Robot`` bundles what algorithms need to act on a specific robot: its tree,
its actuated joints grouped by name (``"arm"``, ``"gripper"``, ...), an
end-effector frame, an injected inverse-kinematics solver, a home configuration,
and the robot's intrinsic allowed-collision pairs. This is composition, not an
inheritance tower: a specific robot is a configured ``Robot`` produced by a
factory (see e.g. :func:`~prpl_kinematics.robots.panda.make_panda`), and
algorithms consume the capabilities they need -- a joint group, the IK solver --
rather than a robot base class. Robots with extra capabilities (a mobile base,
a second arm) simply expose more groups, so they compose instead of forcing a
single inheritance spine.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from prpl_kinematics.ik.interface import InverseKinematics
from prpl_kinematics.planning.joint_space import JointSpace
from prpl_kinematics.tree.kinematic_tree import Configuration, KinematicTree


@dataclass(frozen=True)
class Robot:
    """A specific robot: named joint groups, an end effector, and an IK solver.

    ``allowed_collision_pairs`` are the robot's intrinsic rest-overlapping link
    pairs, computed from the robot alone; a caller building a scene checker
    passes them to ``ignore`` so that real robot-vs-environment collisions are
    never masked.
    """

    name: str
    tree: KinematicTree
    groups: Mapping[str, JointSpace]
    ee_frame: str
    ik: InverseKinematics
    home: Configuration
    allowed_collision_pairs: frozenset[frozenset[str]]
