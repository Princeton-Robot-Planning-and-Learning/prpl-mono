"""Tests for manipulation.py."""

import numpy as np
import pybullet as p

from pybullet_helpers.geometry import (
    Pose,
    multiply_poses,
    set_pose,
)
from pybullet_helpers.manipulation import (
    generate_surface_placements,
    get_kinematic_plan_to_pick_object,
    get_kinematic_plan_to_place_object,
    get_kinematic_plan_to_retract,
    remap_kinematic_state_plan_to_constant_distance,
)
from pybullet_helpers.math_utils import get_poses_facing_line
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.states import KinematicState
from pybullet_helpers.utils import create_pybullet_block, create_pybullet_cylinder


def test_kinematic_pick_place():
    """Tests for get_kinematic_plan_to_pick_object() and
    get_kinematic_plan_to_place_object()."""

    rng = np.random.default_rng(123)

    # Set up a scene to test manipuation.
    physics_client_id = p.connect(p.DIRECT)

    # Uncomment to debug.
    # from pybullet_helpers.gui import create_gui_connection
    # physics_client_id = create_gui_connection(camera_yaw=0)

    full_plan = []

    # Uncomment to debug.
    # from pybullet_helpers.gui import create_gui_connection
    # physics_client_id = create_gui_connection(camera_yaw=0)

    # Create robot.
    robot = create_pybullet_robot("panda", physics_client_id)

    # Create table with back wall (to test multi-link placement).
    table_position = (-0.5, 0.0, -0.24)
    surface_color = (0.5, 0.5, 0.5, 1.0)
    surface_half_extents = (0.1, 0.3, 0.02)
    wall_color = (0.2, 0.2, 0.2, 1.0)
    wall_half_extents = (0.05, 0.3, 0.2)
    surface_collision_id = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=surface_half_extents,
        physicsClientId=physics_client_id,
    )
    surface_visual_id = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=surface_half_extents,
        rgbaColor=surface_color,
        physicsClientId=physics_client_id,
    )
    surface_base_position = (0, 0, wall_half_extents[2] + surface_half_extents[2])
    wall_collision_id = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=wall_half_extents,
        physicsClientId=physics_client_id,
    )
    wall_visual_id = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=wall_half_extents,
        rgbaColor=wall_color,
        physicsClientId=physics_client_id,
    )
    wall_base_position = (0, 0, 0)
    table_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=-1,
        basePosition=table_position,
        linkMasses=[0, 0],
        linkCollisionShapeIndices=[surface_collision_id, wall_collision_id],
        linkVisualShapeIndices=[surface_visual_id, wall_visual_id],
        linkPositions=[surface_base_position, wall_base_position],
        linkOrientations=[[0, 0, 0, 1]] * 2,
        linkInertialFramePositions=[[0, 0, 0]] * 2,
        linkInertialFrameOrientations=[[0, 0, 0, 1]] * 2,
        linkParentIndices=[0] * 2,
        linkJointTypes=[p.JOINT_FIXED] * 2,
        linkJointAxis=[[0, 0, 0]] * 2,
        physicsClientId=physics_client_id,
    )

    # Create object with handle.
    object_position = (-0.5, 0.0, 0.05)
    handle_size = 0.05
    handle_color = (0.9, 0.6, 0.3, 1.0)
    attachment_size = 0.1
    attachment_color = (0.3, 0.6, 0.9, 1.0)
    handle_collision_id = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[handle_size / 2, handle_size / 2, handle_size / 2],
        physicsClientId=physics_client_id,
    )
    handle_visual_id = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[handle_size / 2, handle_size / 2, handle_size / 2],
        rgbaColor=handle_color,
        physicsClientId=physics_client_id,
    )
    handle_base_position = ((handle_size + attachment_size) / 2, 0, 0)
    attachment_collision_id = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[attachment_size / 2, attachment_size / 2, attachment_size / 2],
        physicsClientId=physics_client_id,
    )
    attachment_visual_id = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[attachment_size / 2, attachment_size / 2, attachment_size / 2],
        rgbaColor=attachment_color,
        physicsClientId=physics_client_id,
    )
    attachment_base_position = (0, 0, 0)
    object_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=-1,
        basePosition=object_position,
        linkMasses=[0, 0],
        linkCollisionShapeIndices=[handle_collision_id, attachment_collision_id],
        linkVisualShapeIndices=[handle_visual_id, attachment_visual_id],
        linkPositions=[handle_base_position, attachment_base_position],
        linkOrientations=[[0, 0, 0, 1]] * 2,
        linkInertialFramePositions=[[0, 0, 0]] * 2,
        linkInertialFrameOrientations=[[0, 0, 0, 1]] * 2,
        linkParentIndices=[0] * 2,
        linkJointTypes=[p.JOINT_FIXED] * 2,
        linkJointAxis=[[0, 0, 0]] * 2,
        physicsClientId=physics_client_id,
    )

    # Extract the initial state.
    initial_state = KinematicState.from_pybullet(robot, {object_id, table_id})

    # Set up a grasp generator.
    def _grasp_generator():
        while True:
            # Uncomment to make this test slower but more realistic.
            # angle_offset = rng.uniform(-np.pi, np.pi)
            angle_offset = 0
            relative_pose = get_poses_facing_line(
                axis=(0.0, 0.0, 1.0),
                point_on_line=(0.0, 0.0, 0.0),
                radius=0.0125,
                num_points=1,
                angle_offset=angle_offset,
            )[0]
            rot = Pose.from_rpy((0, 0, 0), (0.0, 0.0, np.pi / 2))
            yield multiply_poses(relative_pose, rot)

    # Get a plan.
    plan = get_kinematic_plan_to_pick_object(
        initial_state,
        robot,
        object_id,
        table_id,
        collision_ids={table_id, object_id},
        grasp_generator=_grasp_generator(),
        object_link_id=0,  # handle
        surface_link_id=0,  # table surface
    )

    assert plan is not None
    full_plan.extend(remap_kinematic_state_plan_to_constant_distance(plan, robot))

    # Advance to the end of the plan.
    initial_state = plan[-1]
    initial_state.set_pybullet(robot)

    # Get a plan.
    obj_half_extents = (handle_size, handle_size, handle_size)
    placement_generator = generate_surface_placements(
        table_id, obj_half_extents, rng, physics_client_id, surface_link_id=0
    )

    plan = get_kinematic_plan_to_place_object(
        initial_state,
        robot,
        object_id,
        table_id,
        collision_ids={table_id},
        placement_generator=placement_generator,
        object_link_id=1,  # attachment
        surface_link_id=0,  # table surface
        retract_after=False,
    )
    full_plan.extend(remap_kinematic_state_plan_to_constant_distance(plan, robot))

    # Advance to the end of the plan.
    initial_state = plan[-1]
    initial_state.set_pybullet(robot)

    # Get a plan.
    plan = get_kinematic_plan_to_retract(
        initial_state,
        robot,
        collision_ids={table_id, object_id},
        translation_magnitude=0.1,
    )
    assert plan is not None
    full_plan.extend(remap_kinematic_state_plan_to_constant_distance(plan, robot))

    # Uncomment to debug.
    # import time
    # for state in full_plan:
    #     state.set_pybullet(robot)
    #     time.sleep(0.1)

    p.disconnect(physics_client_id)


def test_generate_surface_placements():
    """Tests for generate_surface_placements()."""
    rng = np.random.default_rng(123)

    # Set up a scene to test manipuation.
    physics_client_id = p.connect(p.DIRECT)

    # Uncomment to debug.
    # from pybullet_helpers.gui import create_gui_connection
    # physics_client_id = create_gui_connection(camera_yaw=0)

    # Create table.
    table_pose = Pose(position=(-0.5, 0.0, -0.2))
    table_rgba = (0.5, 0.5, 0.5, 1.0)
    table_half_extents = (0.1, 0.3, 0.2)
    table_id = create_pybullet_block(
        table_rgba,
        half_extents=table_half_extents,
        physics_client_id=physics_client_id,
    )
    set_pose(table_id, table_pose, physics_client_id)

    # Create object.
    object_pose = Pose(position=(-0.5, 0.0, 0.05))
    object_rgba = (0.9, 0.6, 0.3, 1.0)
    object_radius = 0.025
    object_length = 0.1
    object_id = create_pybullet_cylinder(
        object_rgba,
        object_radius,
        object_length,
        physics_client_id=physics_client_id,
    )
    set_pose(object_id, object_pose, physics_client_id)
    object_half_extents_at_placement = [object_radius, object_radius, object_length / 2]

    placement_generator = generate_surface_placements(
        table_id, object_half_extents_at_placement, rng, physics_client_id
    )

    for _ in range(100):
        relative_placement = next(placement_generator)
        world_to_object_placement = multiply_poses(table_pose, relative_placement)
        set_pose(object_id, world_to_object_placement, physics_client_id)
        contact_points = p.getClosestPoints(
            object_id,
            table_id,
            distance=1e-6,
            physicsClientId=physics_client_id,
        )
        assert len(contact_points) > 0
