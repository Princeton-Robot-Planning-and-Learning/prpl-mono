"""A robot as composition over a KinematicTree.

A ``Robot`` bundles what algorithms need to act on a specific robot: its tree,
its actuated joints grouped by name (``"arm"``, ``"gripper"``, ``"base"``, ...),
its end effectors (each a ``Manipulator`` pairing an EE frame and an IK solver
with a joint group), a home configuration, and the robot's intrinsic
allowed-collision pairs. This is composition, not an inheritance tower: a
specific robot is a configured ``Robot`` produced by a factory (see e.g.
:func:`~prpl_kinematics.robots.panda.make_panda`), and algorithms consume the
capabilities they need. A bimanual robot simply has two manipulators and a
mobile manipulator an extra base group, so they compose instead of forcing a
single inheritance spine.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from prpl_kinematics.ik.interface import InverseKinematics
from prpl_kinematics.planning.configuration_space import ConfigurationSpace
from prpl_kinematics.tree.kinematic_tree import Configuration, KinematicTree


@dataclass(frozen=True)
class Manipulator:
    """An end effector: a joint group, its EE frame, and its IK solver."""

    group: str
    ee_frame: str
    ik: InverseKinematics


@dataclass(frozen=True)
class Robot:
    """A specific robot: named joint groups, manipulators, home, and its ACM.

    ``allowed_collision_pairs`` are the robot's intrinsic rest-overlapping link
    pairs, computed from the robot alone; a caller building a scene checker
    passes them to ``ignore`` so that real robot-vs-environment collisions are
    never masked.
    """

    name: str
    tree: KinematicTree
    groups: Mapping[str, ConfigurationSpace]
    manipulators: Mapping[str, Manipulator]
    home: Configuration
    allowed_collision_pairs: frozenset[frozenset[str]]
