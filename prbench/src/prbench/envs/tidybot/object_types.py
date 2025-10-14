"""Define object types for the TidyBot environment."""

from relational_structs import Type

MujocoObjectTypeFeatures: dict[Type, list[str]] = {}

MujocoObjectType = Type("mujoco_object")
MujocoObjectTypeFeatures[MujocoObjectType] = [
    "x",
    "y",
    "z",
    "qw",
    "qx",
    "qy",
    "qz",
]
