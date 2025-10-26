"""Utils for tidybot environments."""

import numpy as np
from matplotlib import pyplot as plt
from prbench.envs.dynamic3d.object_types import (
    MujocoObjectType,
    MujocoRobotObjectType,
)
from relational_structs import (
    Object,
    ObjectCentricState,
)
from spatialmath import SE2, UnitQuaternion
from tomsgeoms2d.structs import Geom2D, Rectangle


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


def get_bounding_box(
    state: ObjectCentricState, obj: Object
) -> tuple[float, float, float]:
    """Returns (x extent, y extent, z extent) for the given object.

    We may want to later add something to the state that allows these values to be
    extracted automatically.
    """
    if obj.is_instance(MujocoRobotObjectType):
        # NOTE: hardcoded for now.
        return (0.5, 0.5, 1.0)
    if obj.is_instance(MujocoObjectType):
        return (
            state.get(obj, "bb_x"),
            state.get(obj, "bb_y"),
            state.get(obj, "bb_z"),
        )
    raise NotImplementedError


def get_overhead_geom2ds(state: ObjectCentricState) -> dict[str, Geom2D]:
    """Get a mapping from object name to Geom2D from an overhead perspective."""
    geoms: dict[str, Geom2D] = {}
    for obj in state:
        if obj.is_instance(MujocoRobotObjectType):
            pose = get_overhead_robot_se2_pose(state, obj)
        elif obj.is_instance(MujocoObjectType):
            pose = get_overhead_object_se2_pose(state, obj)
        else:
            raise NotImplementedError
        width, height, _ = get_bounding_box(state, obj)
        geom = Rectangle.from_center(
            pose.x, pose.y, width, height, rotation_about_center=pose.theta()
        )
        geoms[obj.name] = geom
    return geoms


def plot_overhead_scene(
    state: ObjectCentricState,
    min_x: float = -2.5,
    max_x: float = 2.5,
    min_y: float = -2.5,
    max_y: float = 2.5,
    fontsize: int = 6,
) -> tuple[plt.Figure, plt.Axes]:
    """Create a matplotlib figure with a top-down scene rendering."""

    fig, ax = plt.subplots()

    fontdict = {
        "fontsize": fontsize,
        "color": "black",
        "ha": "center",
        "va": "center",
        "fontweight": "medium",
        "bbox": {"facecolor": "white", "alpha": 0.25, "edgecolor": "none", "pad": 2},
    }

    geoms = get_overhead_geom2ds(state)
    for obj_name, geom in geoms.items():
        geom.plot(ax, facecolor="white", edgecolor="black")
        assert isinstance(geom, Rectangle)
        x, y = geom.center
        dx = geom.width / 1.5 * np.cos(geom.theta)
        dy = geom.height / 1.5 * np.sin(geom.theta)
        arrow_width = max(max_x - min_x, max_y - min_y) / 250.0
        ax.arrow(x, y, dx, dy, color="gray", width=arrow_width)
        ax.text(x, y, obj_name, fontdict=fontdict)

    ax.set_xlim((min_x, max_x))
    ax.set_ylim((min_y, max_y))

    return fig, ax
