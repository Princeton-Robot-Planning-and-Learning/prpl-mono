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
) -> int:
    """A generic utility for creating a new block.

    Returns the PyBullet ID of the newly created block.
    """
    # The poses here are not important because they are overwritten by
    # the state values when a task is reset.
    position = (0, 0, 0)
    orientation = (1, 0, 0, 0)

    # Create the collision shape.
    block_collision_id = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=half_extents, physicsClientId=physics_client_id
    )

    # Create the visual_shape.
    block_visual_id = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=color,
        physicsClientId=physics_client_id,
    )

    block_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=block_collision_id,
        baseVisualShapeIndex=block_visual_id,
        basePosition=position,
        baseOrientation=orientation,
        physicsClientId=physics_client_id,
    )

    if friction:
        # Apply friction to the block (base or first link depending on has_peg)
        link_index = -1
        p.changeDynamics(
            block_id,
            linkIndex=link_index,
            lateralFriction=friction,
            physicsClientId=physics_client_id,
        )

    return block_id


def create_pybullet_hollow_box(
    color: tuple[float, float, float, float],
    half_extents: tuple[float, float, float],
    wall_thickness: float,
    physics_client_id: int,
    mass: float = 0,
    friction: float | None = None,
) -> int:
    """A generic utility for creating a hollow box.

    inputs:
        color: RGBA color of the box
        half_extents: half extents of the outer box (x, y, z)
        wall_thickness: thickness of the walls
        physics_client_id: PyBullet physics client ID
        mass: mass of the box
        friction: friction coefficient of the box
    Returns the PyBullet ID of the newly created hollow box.
    """

    # The poses here are not important because they are overwritten by
    # the state values when a task is reset.
    position = (0, 0, 0)
    orientation = (1, 0, 0, 0)

    outer_half_extents = half_extents

    # create base and walls as separate boxes
    base_half_extents = (
        outer_half_extents[0],
        outer_half_extents[1],
        wall_thickness / 2,
    )
    wall1_half_extents = (
        wall_thickness / 2,
        outer_half_extents[1],
        outer_half_extents[2],
    )
    wall2_half_extents = (
        wall_thickness / 2,
        outer_half_extents[1],
        outer_half_extents[2],
    )
    wall3_half_extents = (
        outer_half_extents[0],
        wall_thickness / 2,
        outer_half_extents[2],
    )
    wall4_half_extents = (
        outer_half_extents[0],
        wall_thickness / 2,
        outer_half_extents[2],
    )

    base_collision_id = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=base_half_extents, physicsClientId=physics_client_id
    )
    wall1_collision_id = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=wall1_half_extents, physicsClientId=physics_client_id
    )
    wall2_collision_id = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=wall2_half_extents, physicsClientId=physics_client_id
    )
    wall3_collision_id = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=wall3_half_extents, physicsClientId=physics_client_id
    )
    wall4_collision_id = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=wall4_half_extents, physicsClientId=physics_client_id
    )

    base_visual_id = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=base_half_extents,
        rgbaColor=color,
        physicsClientId=physics_client_id,
    )
    wall1_visual_id = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=wall1_half_extents,
        rgbaColor=color,
        physicsClientId=physics_client_id,
    )
    wall2_visual_id = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=wall2_half_extents,
        rgbaColor=color,
        physicsClientId=physics_client_id,
    )
    wall3_visual_id = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=wall3_half_extents,
        rgbaColor=color,
        physicsClientId=physics_client_id,
    )
    wall4_visual_id = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=wall4_half_extents,
        rgbaColor=color,
        physicsClientId=physics_client_id,
    )

    hollow_box_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=-1,
        basePosition=position,
        baseOrientation=orientation,
        linkMasses=[mass] * 5,
        linkCollisionShapeIndices=[
            base_collision_id,
            wall1_collision_id,
            wall2_collision_id,
            wall3_collision_id,
            wall4_collision_id,
        ],
        linkVisualShapeIndices=[
            base_visual_id,
            wall1_visual_id,
            wall2_visual_id,
            wall3_visual_id,
            wall4_visual_id,
        ],
        linkPositions=[
            (0, 0, -(outer_half_extents[2] - wall_thickness / 2)),  # base
            (-outer_half_extents[0] + wall_thickness / 2, 0, 0),  # wall 1
            (outer_half_extents[0] - wall_thickness / 2, 0, 0),  # wall 2
            (0, -outer_half_extents[1] + wall_thickness / 2, 0),  # wall 3
            (0, outer_half_extents[1] - wall_thickness / 2, 0),  # wall 4
        ],
        linkOrientations=[[0, 0, 0, 1]] * 5,
        linkInertialFramePositions=[[0, 0, 0]] * 5,
        linkInertialFrameOrientations=[[0, 0, 0, 1]] * 5,
        linkParentIndices=[0] * 5,
        linkJointTypes=[p.JOINT_FIXED] * 5,
        linkJointAxis=[[0, 0, 0]] * 5,
        physicsClientId=physics_client_id,
    )

    if friction:
        # Apply friction to the hollow box (base or first link depending on has_peg)
        link_index = 0
        p.changeDynamics(
            hollow_box_id,
            linkIndex=link_index,
            lateralFriction=friction,
            physicsClientId=physics_client_id,
        )

    return hollow_box_id


def create_pybullet_block_with_peg(
    color: tuple[float, float, float, float],
    half_extents: tuple[float, float, float],
    physics_client_id: int,
    mass: float = 0,
    friction: float | None = None,
    peg_height: float = 0.05,
) -> int:
    """A generic utility for creating a new block.

    Returns the PyBullet ID of the newly created block.
    """
    # The poses here are not important because they are overwritten by
    # the state values when a task is reset.
    position = (0, 0, 0)
    orientation = (1, 0, 0, 0)

    # Create the collision shape.
    block_collision_id = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=half_extents, physicsClientId=physics_client_id
    )

    # Create the visual_shape.
    block_visual_id = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=color,
        physicsClientId=physics_client_id,
    )

    # Prepare link data if has_peg
    link_masses = []
    link_collision_ids = []
    link_visual_ids = []
    link_positions = []
    link_orientations = []
    link_inertial_positions = []
    link_inertial_orientations = []
    link_parent_indices = []
    link_joint_types = []
    link_joint_axes = []

    peg_half_extents = (0.01, 0.01, peg_height / 2)

    peg_visual_id = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=peg_half_extents,
        rgbaColor=(1, 0, 0, 1),
        physicsClientId=physics_client_id,
    )
    peg_collision_id = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=peg_half_extents,
        physicsClientId=physics_client_id,
    )

    # Add block as first link
    link_masses.append(mass)
    link_collision_ids.append(block_collision_id)
    link_visual_ids.append(block_visual_id)
    link_positions.append((0.0, 0.0, 0.0))
    link_orientations.append([0, 0, 0, 1])
    link_inertial_positions.append([0, 0, 0])
    link_inertial_orientations.append([0, 0, 0, 1])
    link_parent_indices.append(0)
    link_joint_types.append(p.JOINT_FIXED)
    link_joint_axes.append([0, 0, 0])

    # Add peg as second link
    link_masses.append(0)
    link_collision_ids.append(peg_collision_id)
    link_visual_ids.append(peg_visual_id)
    link_positions.append((0.0, 0.0, half_extents[2] + peg_height / 2))
    link_orientations.append([0, 0, 0, 1])
    link_inertial_positions.append([0, 0, 0])
    link_inertial_orientations.append([0, 0, 0, 1])
    link_parent_indices.append(0)
    link_joint_types.append(p.JOINT_FIXED)
    link_joint_axes.append([0, 0, 0])

    block_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=-1,
        basePosition=position,
        baseOrientation=orientation,
        linkMasses=link_masses,
        linkCollisionShapeIndices=link_collision_ids,
        linkVisualShapeIndices=link_visual_ids,
        linkPositions=link_positions,
        linkOrientations=link_orientations,
        linkInertialFramePositions=link_inertial_positions,
        linkInertialFrameOrientations=link_inertial_orientations,
        linkParentIndices=link_parent_indices,
        linkJointTypes=link_joint_types,
        linkJointAxis=link_joint_axes,
        physicsClientId=physics_client_id,
    )

    if friction:
        # Apply friction to the block (base or first link depending on has_peg)
        link_index = 0
        p.changeDynamics(
            block_id,
            linkIndex=link_index,
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
        raise ValueError(f"Unknown triangle type: {triangle_type}")

    return vertices


def create_pybullet_triangle(
    color: tuple[float, float, float, float],
    triangle_type: str,
    side_lengths: tuple[float, float],
    depth: float,
    physics_client_id: int,
    mass: float = 0,
    friction: float | None = None,
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
    triangle_collision_id = p.createCollisionShape(
        p.GEOM_MESH,
        vertices=mesh_vertices,
        indices=indices,
        physicsClientId=physics_client_id,
    )
    # Create the visual_shape.
    triangle_visual_id = p.createVisualShape(
        p.GEOM_MESH,
        vertices=mesh_vertices,
        indices=indices,
        rgbaColor=color,
        physicsClientId=physics_client_id,
    )

    triangle_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=triangle_collision_id,
        baseVisualShapeIndex=triangle_visual_id,
        basePosition=position,
        baseOrientation=orientation,
        physicsClientId=physics_client_id,
    )

    if friction:
        # Apply friction to the triangle (base or first link depending on has_peg)
        link_index = -1
        p.changeDynamics(
            triangle_id,
            linkIndex=link_index,
            lateralFriction=friction,
            physicsClientId=physics_client_id,
        )

    return triangle_id


def create_pybullet_triangle_with_peg(
    color: tuple[float, float, float, float],
    triangle_type: str,
    side_lengths: tuple[float, float],
    depth: float,
    physics_client_id: int,
    mass: float = 0,
    friction: float | None = None,
    peg_height: float = 0.05,
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
    triangle_collision_id = p.createCollisionShape(
        p.GEOM_MESH,
        vertices=mesh_vertices,
        indices=indices,
        physicsClientId=physics_client_id,
    )
    # Create the visual_shape.
    triangle_visual_id = p.createVisualShape(
        p.GEOM_MESH,
        vertices=mesh_vertices,
        indices=indices,
        rgbaColor=color,
        physicsClientId=physics_client_id,
    )

    triangle_mean = np.mean(np.array(vertices), axis=0, dtype=np.float32)

    # Prepare link data if has_peg
    link_masses = []
    link_collision_ids = []
    link_visual_ids = []
    link_positions = []
    link_orientations = []
    link_inertial_positions = []
    link_inertial_orientations = []
    link_parent_indices = []
    link_joint_types = []
    link_joint_axes = []

    peg_half_extents = (0.01, 0.01, peg_height / 2)

    peg_visual_id = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=peg_half_extents,
        rgbaColor=(1, 0, 0, 1),
        physicsClientId=physics_client_id,
    )
    peg_collision_id = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=peg_half_extents,
        physicsClientId=physics_client_id,
    )

    # Add triangle as first link
    link_masses.append(mass)
    link_collision_ids.append(triangle_collision_id)
    link_visual_ids.append(triangle_visual_id)
    link_positions.append((0.0, 0.0, 0.0))
    link_orientations.append([0, 0, 0, 1])
    link_inertial_positions.append([0, 0, 0])
    link_inertial_orientations.append([0, 0, 0, 1])
    link_parent_indices.append(0)
    link_joint_types.append(p.JOINT_FIXED)
    link_joint_axes.append([0, 0, 0])

    # Add peg as second link
    link_masses.append(0)
    link_collision_ids.append(peg_collision_id)
    link_visual_ids.append(peg_visual_id)
    link_positions.append(
        (triangle_mean[0], triangle_mean[1], depth / 2 + peg_height / 2)
    )
    link_orientations.append([0, 0, 0, 1])
    link_inertial_positions.append([0, 0, 0])
    link_inertial_orientations.append([0, 0, 0, 1])
    link_parent_indices.append(0)
    link_joint_types.append(p.JOINT_FIXED)
    link_joint_axes.append([0, 0, 0])

    triangle_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=-1,
        basePosition=position,
        baseOrientation=orientation,
        linkMasses=link_masses,
        linkCollisionShapeIndices=link_collision_ids,
        linkVisualShapeIndices=link_visual_ids,
        linkPositions=link_positions,
        linkOrientations=link_orientations,
        linkInertialFramePositions=link_inertial_positions,
        linkInertialFrameOrientations=link_inertial_orientations,
        linkParentIndices=link_parent_indices,
        linkJointTypes=link_joint_types,
        linkJointAxis=link_joint_axes,
        physicsClientId=physics_client_id,
    )

    if friction:
        # Apply friction to the triangle (base or first link depending on has_peg)
        link_index = 0
        p.changeDynamics(
            triangle_id,
            linkIndex=link_index,
            lateralFriction=friction,
            physicsClientId=physics_client_id,
        )

    return triangle_id


def create_pybullet_shelf(
    color: tuple[float, float, float, float],
    shelf_width: float,
    shelf_depth: float,
    shelf_height: float,
    spacing: float,
    support_width: float,
    num_layers: int,
    physics_client_id: int,
    shelf_texture_id: int | None = None,
) -> tuple[int, set[int]]:
    """Returns the shelf ID and the link IDs of the individual shelves."""

    collision_shape_ids = []
    visual_shape_ids = []
    base_positions = []
    base_orientations = []
    link_masses = []
    link_parent_indices = []
    link_joint_types = []
    link_joint_axes = []

    # Add each shelf layer to the lists.
    for i in range(num_layers):
        layer_z = i * (spacing + shelf_height)

        col_shape_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[shelf_width / 2, shelf_depth / 2, shelf_height / 2],
            physicsClientId=physics_client_id,
        )
        visual_shape_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[shelf_width / 2, shelf_depth / 2, shelf_height / 2],
            rgbaColor=color,
            physicsClientId=physics_client_id,
        )

        collision_shape_ids.append(col_shape_id)
        visual_shape_ids.append(visual_shape_id)
        base_positions.append([0, 0, layer_z])
        base_orientations.append([0, 0, 0, 1])
        link_masses.append(0)
        link_parent_indices.append(0)
        link_joint_types.append(p.JOINT_FIXED)
        link_joint_axes.append([0, 0, 0])

    shelf_link_ids = set(range(num_layers))

    # Add vertical side supports to the lists.
    support_height = (num_layers - 1) * spacing + (num_layers) * shelf_height
    support_half_height = support_height / 2

    for x_offset in [
        -shelf_width / 2 - support_width / 2,
        shelf_width / 2 + support_width / 2,
    ]:
        support_col_shape_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[support_width / 2, shelf_depth / 2, support_half_height],
            physicsClientId=physics_client_id,
        )
        support_visual_shape_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[support_width / 2, shelf_depth / 2, support_half_height],
            rgbaColor=color,
            physicsClientId=physics_client_id,
        )

        collision_shape_ids.append(support_col_shape_id)
        visual_shape_ids.append(support_visual_shape_id)
        base_positions.append([x_offset, 0, support_half_height - shelf_height / 2])
        base_orientations.append([0, 0, 0, 1])
        link_masses.append(0)
        link_parent_indices.append(0)
        link_joint_types.append(p.JOINT_FIXED)
        link_joint_axes.append([0, 0, 0])

    # Create the multibody with all collision and visual shapes.
    shelf_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=-1,
        basePosition=(0, 0, 0),  # changed externally
        linkMasses=link_masses,
        linkCollisionShapeIndices=collision_shape_ids,
        linkVisualShapeIndices=visual_shape_ids,
        linkPositions=base_positions,
        linkOrientations=base_orientations,
        linkInertialFramePositions=[[0, 0, 0]] * len(collision_shape_ids),
        linkInertialFrameOrientations=[[0, 0, 0, 1]] * len(collision_shape_ids),
        linkParentIndices=link_parent_indices,
        linkJointTypes=link_joint_types,
        linkJointAxis=link_joint_axes,
        physicsClientId=physics_client_id,
    )
    if shelf_texture_id is not None:
        for link_id in range(
            p.getNumJoints(shelf_id, physicsClientId=physics_client_id)
        ):
            p.changeVisualShape(
                shelf_id,
                link_id,
                textureUniqueId=shelf_texture_id,
                physicsClientId=physics_client_id,
            )

    return shelf_id, shelf_link_ids


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
