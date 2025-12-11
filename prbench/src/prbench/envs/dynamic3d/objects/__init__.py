"""Object definitions for TidyBot environments."""

from prbench.envs.dynamic3d.objects.base import (
    MujocoFixture,
    MujocoObject,
    get_fixture_class,
    get_object_class,
    register_fixture,
    register_object,
)
from prbench.envs.dynamic3d.objects.fixtures import Cupboard, Table
from prbench.envs.dynamic3d.objects.primitive_objects import Cube, Cuboid

# Import robocasa_objects to trigger auto-registration of RoboCasa object classes
from prbench.envs.dynamic3d.objects import robocasa_objects

__all__ = [
    "MujocoObject",
    "MujocoFixture",
    "register_object",
    "register_fixture",
    "get_object_class",
    "get_fixture_class",
    "Cuboid",
    "Cube",
    "Table",
    "Cupboard",
    "robocasa_objects",
]
