"""Utils for tidybot environments."""

from prbench.envs.dynamic3d.object_types import MujocoObjectType, MujocoRobotObjectType
from relational_structs import (
    Object,
    ObjectCentricState,
)
from spatialmath import SE2, UnitQuaternion
from tomsgeoms2d.structs import Geom2D


def get_overhead_object_se2_pose(state: ObjectCentricState, obj: Object) -> SE2:
    """Get the top-down SE2 pose for an object in a dynamic3D state."""
    assert obj.is_instance(MujocoObjectType)
    x = state.get(obj, "x")
    y = state.get(obj, "y")
    q = UnitQuaternion(
        s=state.get(obj, "qw"),
        v=(
            state.get(obj, "qx"),
            state.get(obj, "qy"),
            state.get(obj, "qz"),
        ),
    )
    rpy = q.rpy()
    yaw = rpy[2]
    return SE2(x, y, yaw)


def get_overhead_robot_se2_pose(state: ObjectCentricState, obj: Object) -> SE2:
    """Get the top-down SE2 pose for an object in a dynamic3D state."""
    assert obj.is_instance(MujocoRobotObjectType)
    x = state.get(obj, "pos_base_x")
    y = state.get(obj, "pos_base_y")
    yaw = state.get(obj, "pos_base_rot")
    return SE2(x, y, yaw)


def get_overhead_geom2ds(state: ObjectCentricState) -> dict[str, Geom2D]:
    """Get a mapping from object name to Geom2D from an overhead perspective."""
