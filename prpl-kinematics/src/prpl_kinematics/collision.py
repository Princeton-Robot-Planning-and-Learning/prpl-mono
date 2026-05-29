"""PyBullet-based collision checking via shape-soup.

One PyBullet collision body per node shape; each query positions every body from
the tree's forward kinematics, then checks the non-ignored body pairs. Because
bodies are positioned by FK, a grasped object (re-parented in the tree) collides
with the environment for free.

Pairs that are in contact by construction are ignored: shapes on the same node,
and tree-adjacent (parent-child) node pairs. Some links' collision geometry also
overlaps at rest without being adjacent; such pairs can be supplied up front or
discovered with :meth:`PyBulletCollisionChecker.pairs_in_collision` and passed to
:meth:`PyBulletCollisionChecker.ignore`.
"""

from __future__ import annotations

from collections.abc import Iterable

import pybullet as p

from prpl_kinematics.geometry.shapes import BoxShape, CylinderShape, MeshShape, Shape
from prpl_kinematics.geometry.transforms import pose_to_pybullet
from prpl_kinematics.meshes import to_pybullet_mesh
from prpl_kinematics.tree.kinematic_tree import Configuration, KinematicTree


def _create_collision_shape(physics_client_id: int, shape: Shape) -> int:
    position, orientation = pose_to_pybullet(shape.origin)
    common = {
        "collisionFramePosition": position,
        "collisionFrameOrientation": orientation,
        "physicsClientId": physics_client_id,
    }
    if isinstance(shape, MeshShape):
        return int(
            p.createCollisionShape(
                p.GEOM_MESH,
                fileName=to_pybullet_mesh(shape.filename),
                meshScale=list(shape.scale),
                **common,
            )
        )
    if isinstance(shape, BoxShape):
        half_extents = [shape.size[0] / 2, shape.size[1] / 2, shape.size[2] / 2]
        return int(
            p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, **common)
        )
    if isinstance(shape, CylinderShape):
        return int(
            p.createCollisionShape(
                p.GEOM_CYLINDER, radius=shape.radius, height=shape.length, **common
            )
        )
    return int(p.createCollisionShape(p.GEOM_SPHERE, radius=shape.radius, **common))


class PyBulletCollisionChecker:
    """Shape-soup collision checker for a KinematicTree."""

    def __init__(
        self,
        physics_client_id: int,
        ignored_pairs: Iterable[Iterable[str]] = (),
    ) -> None:
        self._physics_client_id = physics_client_id
        self._tree: KinematicTree | None = None
        self._bodies: list[tuple[int, str]] = []
        self._ignored: set[frozenset[str]] = {frozenset(pair) for pair in ignored_pairs}

    def load(self, tree: KinematicTree) -> None:
        """Create one collision body per node shape and ignore adjacent pairs."""
        self._tree = tree
        for name, node in tree.nodes.items():
            for shape in node.collisions:
                collision = _create_collision_shape(self._physics_client_id, shape)
                body = int(
                    p.createMultiBody(
                        baseMass=0,
                        baseCollisionShapeIndex=collision,
                        physicsClientId=self._physics_client_id,
                    )
                )
                self._bodies.append((body, name))
        for child in tree.nodes:
            edges = tree.path_from_root(child)
            if edges:
                self._ignored.add(frozenset({edges[-1].parent, child}))

    def ignore(self, pairs: Iterable[Iterable[str]]) -> None:
        """Additionally ignore collisions between the given node-name pairs."""
        for pair in pairs:
            self._ignored.add(frozenset(pair))

    def in_collision(self, config: Configuration, max_distance: float = 0.0) -> bool:
        """Whether any non-ignored shape pair is within ``max_distance``."""
        self._position(config)
        for index, (body_a, node_a) in enumerate(self._bodies):
            for body_b, node_b in self._bodies[index + 1 :]:
                if node_a == node_b or frozenset({node_a, node_b}) in self._ignored:
                    continue
                if self._closest_points(body_a, body_b, max_distance):
                    return True
        return False

    def pairs_in_collision(
        self, config: Configuration, max_distance: float = 0.0
    ) -> set[frozenset[str]]:
        """All overlapping node-name pairs, ignoring only same-node shapes.

        Useful for discovering rest-overlapping pairs to pass to :meth:`ignore`.
        """
        self._position(config)
        found: set[frozenset[str]] = set()
        for index, (body_a, node_a) in enumerate(self._bodies):
            for body_b, node_b in self._bodies[index + 1 :]:
                if node_a == node_b:
                    continue
                if self._closest_points(body_a, body_b, max_distance):
                    found.add(frozenset({node_a, node_b}))
        return found

    def _position(self, config: Configuration) -> None:
        assert self._tree is not None, "call load() before querying"
        for body, name in self._bodies:
            position, orientation = pose_to_pybullet(
                self._tree.forward_kinematics(name, config)
            )
            p.resetBasePositionAndOrientation(
                body, position, orientation, physicsClientId=self._physics_client_id
            )

    def _closest_points(self, body_a: int, body_b: int, max_distance: float) -> bool:
        points = p.getClosestPoints(
            body_a,
            body_b,
            distance=max_distance,
            physicsClientId=self._physics_client_id,
        )
        return len(points) > 0
