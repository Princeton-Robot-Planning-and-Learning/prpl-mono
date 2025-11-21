"""Utility functions."""

from pathlib import Path

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

    # Create the body.
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
