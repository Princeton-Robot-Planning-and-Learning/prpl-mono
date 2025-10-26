"""Utils for tidybot environments."""

from typing import Iterable

import numpy as np
from matplotlib import pyplot as plt
from prbench.envs.dynamic3d.object_types import (
    MujocoObjectType,
    MujocoRobotObjectType,
)
from prpl_utils.motion_planning import BiRRT
from prpl_utils.utils import get_signed_angle_distance, wrap_angle
from relational_structs import (
    Object,
    ObjectCentricState,
)
from spatialmath import SE2, UnitQuaternion
from tomsgeoms2d.structs import Geom2D, Rectangle
from tomsgeoms2d.utils import geom2ds_intersect


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


def run_base_motion_planning(
    state: ObjectCentricState,
    target_base_pose: SE2,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    seed: int,
    extend_xy_magnitude: float = 0.025,
    extend_rot_magnitude: float = np.pi / 8,
    num_attempts: int = 10,
    num_iters: int = 100,
    smooth_amt: int = 50,
) -> list[SE2] | None:
    """Run motion planning for the robot base."""

    rng = np.random.default_rng(seed)

    # Construct geoms.
    geoms = get_overhead_geom2ds(state)
    (robot,) = state.get_objects(MujocoRobotObjectType)
    robot_width, robot_height, _ = get_bounding_box(state, robot)
    obstacles = state.get_objects(MujocoObjectType)
    obstacle_geoms = {geoms[o.name] for o in obstacles}

    # Set up the RRT methods.
    def sample_fn(_: SE2) -> SE2:
        """Sample a robot pose."""
        x = rng.uniform(*x_bounds)
        y = rng.uniform(*y_bounds)
        theta = rng.uniform(-np.pi, np.pi)
        return SE2(x, y, theta)

    def extend_fn(pt1: SE2, pt2: SE2) -> Iterable[SE2]:
        """Interpolate between the two poses."""
        # Make sure that we obey the bounds on actions.
        dx = pt2.x - pt1.x
        dy = pt2.y - pt1.y
        dtheta = get_signed_angle_distance(pt2.theta(), pt1.theta())
        x_num_steps = int(abs(dx) / extend_xy_magnitude) + 1
        assert x_num_steps > 0
        y_num_steps = int(abs(dy) / extend_xy_magnitude) + 1
        assert y_num_steps > 0
        theta_num_steps = int(abs(dtheta) / extend_rot_magnitude) + 1
        assert theta_num_steps > 0
        num_steps = max(x_num_steps, y_num_steps, theta_num_steps)
        x = pt1.x
        y = pt1.y
        theta = pt1.theta()
        yield SE2(x, y, theta)
        for _ in range(num_steps):
            x += dx / num_steps
            y += dy / num_steps
            theta = wrap_angle(theta + dtheta / num_steps)
            yield SE2(x, y, theta)

    def collision_fn(pt: SE2) -> bool:
        """Check for collisions if the robot were at this pose."""
        # Get the new robot geom.
        new_state = state.copy()
        new_state.set(robot, "pos_base_x", pt.x)
        new_state.set(robot, "pos_base_y", pt.y)
        new_state.set(robot, "pos_base_rot", pt.theta())
        pose = get_overhead_robot_se2_pose(new_state, robot)
        robot_geom = Rectangle.from_center(
            pose.x,
            pose.y,
            robot_width,
            robot_height,
            rotation_about_center=pose.theta(),
        )
        for obstacle_geom in obstacle_geoms:
            if geom2ds_intersect(robot_geom, obstacle_geom):
                return True
        return False

    def distance_fn(pt1: SE2, pt2: SE2) -> float:
        """Return a distance between the two points."""
        dx = pt2.x - pt1.x
        dy = pt2.y - pt1.y
        dtheta = get_signed_angle_distance(pt2.theta(), pt1.theta())
        return np.sqrt(dx**2 + dy**2) + abs(dtheta)

    birrt = BiRRT(
        sample_fn,
        extend_fn,
        collision_fn,
        distance_fn,
        rng,
        num_attempts,
        num_iters,
        smooth_amt,
    )

    initial_pose = get_overhead_robot_se2_pose(state, robot)
    return birrt.query(initial_pose, target_base_pose)
