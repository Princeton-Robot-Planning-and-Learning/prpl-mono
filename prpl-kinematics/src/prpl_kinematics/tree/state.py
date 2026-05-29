"""KinematicState: a restorable snapshot of a tree's configuration and structure.

A state captures the joint values of every actuated joint *and* the tree's edges
(each node's parent and joint). Because a grasp is just an edge -- ``attach``
re-parents an object onto a gripper -- snapshotting the edges captures grasps for
free: two states differ in an object's incoming edge across a pick or place.
:meth:`KinematicState.apply` restores both the structure and the configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from prpl_kinematics.tree.joints import Joint, JointValues
from prpl_kinematics.tree.kinematic_tree import (
    Configuration,
    Edge,
    KinematicTree,
)


@dataclass(frozen=True)
class KinematicState:
    """An immutable snapshot of actuated joint values and tree structure."""

    joint_values: dict[str, tuple[float, ...]]
    edges: dict[str, tuple[str, Joint]] = field(default_factory=dict)

    @classmethod
    def from_tree(cls, tree: KinematicTree, config: Configuration) -> KinematicState:
        """Snapshot ``config`` over the actuated joints and ``tree``'s edges."""
        values: dict[str, tuple[float, ...]] = {}
        for name in tree.actuated_joint_names():
            num_dof = tree.joint(name).num_dof
            values[name] = tuple(config.get(name, [0.0] * num_dof))
        edges = {child: (edge.parent, edge.joint) for child, edge in tree.edges.items()}
        return cls(values, edges)

    def as_configuration(self) -> dict[str, JointValues]:
        """Return the snapshot as a configuration mapping for forward kinematics."""
        return {name: list(vals) for name, vals in self.joint_values.items()}

    def apply(self, tree: KinematicTree) -> dict[str, JointValues]:
        """Restore this state's edges onto ``tree`` and return its configuration.

        Re-parents nodes to match the snapshot (restoring any grasp), then hands back
        the configuration for forward kinematics.
        """
        for child, (parent, joint) in self.edges.items():
            tree.set_edge(Edge(parent=parent, child=child, joint=joint))
        return self.as_configuration()
