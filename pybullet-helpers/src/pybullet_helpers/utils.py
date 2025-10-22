"""Utility functions."""

from pathlib import Path

import numpy as np
import pybullet as p


def get_root_path() -> Path:
    """Get the path to the root directory of this package."""
    return Path(__file__).parent


def get_assets_path() -> Path:
    """Return the absolute path to the assets directory."""
    return get_root_path() / "assets"


def get_third_party_path() -> Path:
    """Return the absolute path to the third party directory."""
    return get_root_path() / "third_party"


def create_pybullet_block(
    color: tuple[float, float, float, float],
    half_extents: tuple[float, float, float],
    physics_client_id: int,
    mass: float = 0,
    friction: float | None = None,
    has_peg: bool = False,
) -> int:
    """A generic utility for creating a new block.

    Returns the PyBullet ID of the newly created block.
    """
    # The poses here are not important because they are overwritten by
    # the state values when a task is reset.
    position = (0, 0, 0)
    orientation = (1, 0, 0, 0)

    # Create the collision shape.
    collision_id = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=half_extents, physicsClientId=physics_client_id
    )

    # Create the visual_shape.
    visual_id = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=color,
        physicsClientId=physics_client_id,
    )

    if has_peg:
        # Create a peg on top of the block for stacking tasks.
        peg_height = 0.05
        peg_radius = 0.01

        peg_visual_id = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=peg_radius,
            length=peg_height,
            rgbaColor=(1, 0, 0, 1),
            physicsClientId=physics_client_id,
        )
        peg_collision_id = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=peg_radius,
            height=peg_height,
            physicsClientId=physics_client_id,
        )

    # Create the body for the block and peg if applicable.
    if has_peg:
        block_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=position,
            baseOrientation=orientation,
            linkMasses=[0],
            linkCollisionShapeIndices=[peg_collision_id],
            linkVisualShapeIndices=[peg_visual_id],
            linkPositions=[(0, 0, half_extents[2] + peg_height / 2)],
            linkOrientations=[(1, 0, 0, 0)],
            linkInertialFramePositions=[(0, 0, 0)],
            linkInertialFrameOrientations=[(1, 0, 0, 0)],
            linkParentIndices=[0],
            linkJointTypes=[p.JOINT_FIXED],
            linkJointAxis=[(0, 0, 0)],
            physicsClientId=physics_client_id,
        )
    else:
        block_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=position,
            baseOrientation=orientation,
            physicsClientId=physics_client_id,
        )

    if friction:
        p.changeDynamics(
            block_id,
            linkIndex=-1,  # -1 for the base
            lateralFriction=friction,
            physicsClientId=physics_client_id,
        )

    return block_id


def create_pybullet_cylinder(
    color: tuple[float, float, float, float],
    radius: float,
    length: float,
    physics_client_id: int,
    mass: float = 0,
    friction: float | None = None,
) -> int:
    """A generic utility for creating a cylinder.

    Returns the PyBullet ID of the newly created cylinder.
    """
    # The poses here are not important because they are overwritten by
    # the state values when a task is reset.
    position = (0, 0, 0)
    orientation = (1, 0, 0, 0)

    # Create the collision shape.
    collision_id = p.createCollisionShape(
        p.GEOM_CYLINDER, radius=radius, height=length, physicsClientId=physics_client_id
    )

    # Create the visual_shape.
    visual_id = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=radius,
        length=length,
        rgbaColor=color,
        physicsClientId=physics_client_id,
    )

    # Create the body.
    cylinder_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
        basePosition=position,
        baseOrientation=orientation,
        physicsClientId=physics_client_id,
    )

    if friction:
        p.changeDynamics(
            cylinder_id,
            linkIndex=-1,  # -1 for the base
            lateralFriction=friction,
            physicsClientId=physics_client_id,
        )

    return cylinder_id


def get_triangle_vertices(
    triangle_type: str,
    side_lengths: tuple[float, float],
) -> list[list[float]]:
    """Get the vertices of a triangle given its type and side lengths."""

    vertices = []
    if triangle_type == "equilateral":
        side = side_lengths[0]
        height = (np.sqrt(3) / 2) * side
        v0 = [-side / 2, -height / 3, 0]
        v1 = [side / 2, -height / 3, 0]
        v2 = [0, 2 * height / 3, 0]
        vertices = [v0, v1, v2]
    elif triangle_type == "isosceles":
        base = side_lengths[0]
        height = side_lengths[1]
        v0 = [-base / 2, 0, 0]
        v1 = [base / 2, 0, 0]
        v2 = [0, height, 0]
        vertices = [v0, v1, v2]
    elif triangle_type == "right":
        base = side_lengths[0]
        height = side_lengths[1]
        v0 = [0, 0, 0]
        v1 = [base, 0, 0]
        v2 = [0, height, 0]
        vertices = [v0, v1, v2]
    else:
        raise ValueError(f"Unknown triangle type: {type}")

    return vertices


def create_pybullet_triangle(
    color: tuple[float, float, float, float],
    triangle_type: str,
    side_lengths: tuple[float, float],
    depth: float,
    physics_client_id: int,
    mass: float = 0,
    friction: float | None = None,
    has_peg: bool = False,
) -> int:
    """A generic utility for creating a triangle.

    Returns the PyBullet ID of the newly created triangle.
    """

    # The poses here are not important because they are overwritten by
    # the state values when a task is reset.
    position = (0, 0, 0)
    orientation = (1, 0, 0, 0)

    vertices = get_triangle_vertices(triangle_type, side_lengths)

    # Extrude the 2D triangle to create a 3D mesh
    mesh_vertices = []
    for v in vertices:
        mesh_vertices.append([v[0], v[1], -depth / 2])
    for v in vertices:
        mesh_vertices.append([v[0], v[1], depth / 2])

    # Define the indices for the triangular and rectangular faces
    indices = [
        # Front face
        0,
        1,
        2,
        # Back face
        3,
        5,
        4,
        # Side faces
        0,
        3,
        4,
        0,
        4,
        1,
        1,
        4,
        5,
        1,
        5,
        2,
        2,
        5,
        3,
        2,
        3,
        0,
    ]

    # Create the collision shape.
    collision_id = p.createCollisionShape(
        p.GEOM_MESH,
        vertices=mesh_vertices,
        indices=indices,
        physicsClientId=physics_client_id,
    )
    # Create the visual_shape.
    visual_id = p.createVisualShape(
        p.GEOM_MESH,
        vertices=mesh_vertices,
        indices=indices,
        rgbaColor=color,
        physicsClientId=physics_client_id,
    )

    if has_peg:
        # Create a peg on top of the block for stacking tasks.
        peg_height = 0.05
        peg_radius = 0.01

        peg_visual_id = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=peg_radius,
            length=peg_height,
            rgbaColor=(1, 0, 0, 1),
            physicsClientId=physics_client_id,
        )
        peg_collision_id = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=peg_radius,
            height=peg_height,
            physicsClientId=physics_client_id,
        )

    triangle_mean = np.mean(np.array(vertices), axis=0)

    # Create the body for the triangle and peg if applicable.
    if has_peg:
        triangle_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=position,
            baseOrientation=orientation,
            linkMasses=[0],
            linkCollisionShapeIndices=[peg_collision_id],
            linkVisualShapeIndices=[peg_visual_id],
            linkPositions=[
                (triangle_mean[0], triangle_mean[1], depth / 2 + peg_height / 2)
            ],
            linkOrientations=[(1, 0, 0, 0)],
            linkInertialFramePositions=[(triangle_mean[0], triangle_mean[1], 0)],
            linkInertialFrameOrientations=[(1, 0, 0, 0)],
            linkParentIndices=[0],
            linkJointTypes=[p.JOINT_FIXED],
            linkJointAxis=[(triangle_mean[0], triangle_mean[1], 0)],
            physicsClientId=physics_client_id,
        )
    else:
        triangle_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=position,
            baseOrientation=orientation,
            physicsClientId=physics_client_id,
        )

    if friction:
        p.changeDynamics(
            triangle_id,
            linkIndex=-1,  # -1 for the base
            lateralFriction=friction,
            physicsClientId=physics_client_id,
        )

    return triangle_id


def get_closest_points_with_optional_links(
    body1: int,
    body2: int,
    physics_client_id: int,
    link1: int | None = None,
    link2: int | None = None,
    distance_threshold: float = 1e-6,
    perform_collision_detection: bool = True,
) -> list[tuple]:
    """Wrapper around getClosestPoints, which doesn't seem to work with
    optional link setting."""
    if perform_collision_detection:
        p.performCollisionDetection(physicsClientId=physics_client_id)
    if link1 is not None and link2 is not None:
        closest_points = p.getClosestPoints(
            bodyA=body1,
            bodyB=body2,
            linkIndexA=link1,
            linkIndexB=link2,
            distance=distance_threshold,
            physicsClientId=physics_client_id,
        )
    elif link1 is not None:
        closest_points = p.getClosestPoints(
            bodyA=body1,
            bodyB=body2,
            linkIndexA=link1,
            distance=distance_threshold,
            physicsClientId=physics_client_id,
        )
    elif link2 is not None:
        closest_points = p.getClosestPoints(
            bodyA=body1,
            bodyB=body2,
            linkIndexB=link2,
            distance=distance_threshold,
            physicsClientId=physics_client_id,
        )
    else:
        closest_points = p.getClosestPoints(
            bodyA=body1,
            bodyB=body2,
            distance=distance_threshold,
            physicsClientId=physics_client_id,
        )
    # PyBullet strangely sometimes returns None, other times returns an empty
    # list in cases where there is no collision. Empty list is more common.
    if closest_points is None:
        return []
    return closest_points
