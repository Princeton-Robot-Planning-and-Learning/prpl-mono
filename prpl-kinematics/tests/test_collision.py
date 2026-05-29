"""Unit tests for PyBullet shape-soup collision checking."""

from spatialmath import SE3

from prpl_kinematics.collision import PyBulletCollisionChecker
from prpl_kinematics.geometry.shapes import BoxShape
from prpl_kinematics.loading import load_urdf
from prpl_kinematics.tree.joints import FixedJoint, PrismaticJoint
from prpl_kinematics.tree.kinematic_tree import Edge, KinematicTree, Node
from prpl_kinematics.utils import get_assets_path


def _box(name: str, size: float = 0.2) -> Node:
    return Node(name, collisions=[BoxShape(size=(size, size, size))])


def _two_box_tree() -> KinematicTree:
    """World -[prismatic x]-> a(box); world -[fixed]-> b(box at origin)."""
    tree = KinematicTree()
    tree.add_node(_box("a"))
    tree.add_node(_box("b"))
    tree.add_edge(
        Edge(
            "world", "a", PrismaticJoint(name="slide", axis=(1, 0, 0), lower=0, upper=5)
        )
    )
    tree.add_edge(Edge("world", "b", FixedJoint(name="fix_b", origin=SE3())))
    return tree


def test_boxes_collide_when_overlapping(physics_client_id):
    """Two boxes collide when overlapping and not when separated."""
    checker = PyBulletCollisionChecker(physics_client_id)
    checker.load(_two_box_tree())
    assert checker.in_collision({"slide": [0.0]})
    assert not checker.in_collision({"slide": [1.0]})


def test_adjacent_links_ignored(physics_client_id):
    """Overlapping parent-child links are ignored, but still reported raw."""
    tree = KinematicTree()
    tree.add_node(_box("l1"))
    tree.add_node(_box("l2"))
    tree.add_edge(Edge("world", "l1", FixedJoint(name="f1", origin=SE3())))
    tree.add_edge(Edge("l1", "l2", FixedJoint(name="f2", origin=SE3(0.05, 0, 0))))
    checker = PyBulletCollisionChecker(physics_client_id)
    checker.load(tree)
    assert not checker.in_collision({})
    assert frozenset({"l1", "l2"}) in checker.pairs_in_collision({})


def test_explicitly_ignored_pair(physics_client_id):
    """A non-adjacent overlapping pair can be explicitly allowed."""
    checker = PyBulletCollisionChecker(physics_client_id, ignored_pairs=[("a", "b")])
    checker.load(_two_box_tree())
    assert not checker.in_collision({"slide": [0.0]})


def test_attached_object_collides_with_environment(physics_client_id):
    """A grasped object (attached in the tree) collides with the environment."""
    tree = _two_box_tree()
    tree.add_node(Node("held", collisions=[BoxShape(size=(0.1, 0.1, 0.1))]))
    tree.attach("held", "a", SE3(-1, 0, 0))  # held trails 1m behind link a
    checker = PyBulletCollisionChecker(physics_client_id)
    checker.load(tree)
    # a at x=1 (clear of b), so the held object at x=0 is what overlaps b.
    assert checker.in_collision({"slide": [1.0]})
    assert not checker.in_collision({"slide": [3.0]})


def test_panda_home_collision_free_with_allowed_pairs(physics_client_id):
    """Panda's rest pose is collision-free once rest-overlapping pairs are allowed."""
    path = str(get_assets_path() / "urdf" / "panda_arm_hand.urdf")
    checker = PyBulletCollisionChecker(physics_client_id)
    checker.load(load_urdf(path))
    resting = checker.pairs_in_collision({})
    assert resting  # mesh collision shapes detect real rest overlaps
    checker.ignore(resting)
    assert not checker.in_collision({})
