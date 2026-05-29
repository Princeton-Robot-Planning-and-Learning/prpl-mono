"""KinematicState: a restorable snapshot of a tree's configuration.

A state captures the joint values for every actuated joint of a tree, so a
configuration can be saved and later reapplied for forward kinematics.
"""

from __future__ import annotations

from dataclasses import dataclass

from prpl_kinematics.tree.joints import JointValues
from prpl_kinematics.tree.kinematic_tree import Configuration, KinematicTree


@dataclass(frozen=True)
class KinematicState:
    """An immutable snapshot of all actuated joint values."""

    joint_values: dict[str, tuple[float, ...]]

    @classmethod
    def from_tree(cls, tree: KinematicTree, config: Configuration) -> KinematicState:
        """Snapshot ``config`` over the actuated joints of ``tree``."""
        values: dict[str, tuple[float, ...]] = {}
        for name in tree.actuated_joint_names():
            num_dof = tree.joint(name).num_dof
            values[name] = tuple(config.get(name, [0.0] * num_dof))
        return cls(values)

    def as_configuration(self) -> dict[str, JointValues]:
        """Return the snapshot as a configuration mapping for forward kinematics."""
        return {name: list(vals) for name, vals in self.joint_values.items()}
