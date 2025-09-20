"""Tests for export.py."""

import tempfile

import numpy as np
import pybullet as p

from pybullet_helpers.export import create_urdf_from_body_id
from pybullet_helpers.geometry import (
    Pose,
    get_half_extents_from_aabb,
    get_pose,
    set_pose,
)
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.utils import create_pybullet_block, create_pybullet_cylinder


def test_block_write_urdf_from_body_id(physics_client_id):
    """Tests for write_urdf_from_body_id() with a block."""
    block_id = create_pybullet_block(
        color=(0.1, 0.5, 0.9, 1.0),
        half_extents=(1.0, 2.0, 0.5),
        physics_client_id=physics_client_id,
        mass=1.0,
        friction=0.25,
    )
    set_pose(block_id, Pose.identity(), physics_client_id)
    original_pose = get_pose(block_id, physics_client_id)
    original_half_extents = get_half_extents_from_aabb(block_id, physics_client_id)
    original_mass = p.getDynamicsInfo(block_id, -1, physicsClientId=physics_client_id)[
        0
    ]

    urdf = create_urdf_from_body_id(block_id, physics_client_id)

    urdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".urdf").name
    with open(urdf_file, mode="w", encoding="utf-8") as f:
        f.write(urdf)

    p.removeBody(block_id, physicsClientId=physics_client_id)

    # Recreate and compare.
    recreated_block_id = p.loadURDF(
        urdf_file, (0, 0, 0), (0, 0, 0, 1), physicsClientId=physics_client_id
    )
    recovered_pose = get_pose(recreated_block_id, physics_client_id)
    recovered_half_extents = get_half_extents_from_aabb(
        recreated_block_id, physics_client_id
    )
    recovered_mass = p.getDynamicsInfo(
        recreated_block_id, -1, physicsClientId=physics_client_id
    )[0]

    assert original_pose.allclose(recovered_pose, atol=1e-6)
    # NOTE: bounding box may be slightly enlarged in loading URDF, probably
    # because of some conservative thing that pybullet is doing.
    assert np.allclose(original_half_extents, recovered_half_extents, atol=1e-2)
    assert np.isclose(original_mass, recovered_mass)


def test_cylinder_write_urdf_from_body_id(physics_client_id):
    """Tests for write_urdf_from_body_id() with a cylinder."""
    cylinder_id = create_pybullet_cylinder(
        color=(0.1, 0.5, 0.9, 1.0),
        radius=0.2,
        length=0.8,
        physics_client_id=physics_client_id,
        mass=1.0,
        friction=0.25,
    )
    set_pose(cylinder_id, Pose.identity(), physics_client_id)
    original_pose = get_pose(cylinder_id, physics_client_id)
    original_half_extents = get_half_extents_from_aabb(cylinder_id, physics_client_id)
    original_mass = p.getDynamicsInfo(
        cylinder_id, -1, physicsClientId=physics_client_id
    )[0]

    urdf = create_urdf_from_body_id(cylinder_id, physics_client_id)
    urdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".urdf").name
    with open(urdf_file, mode="w", encoding="utf-8") as f:
        f.write(urdf)

    p.removeBody(cylinder_id, physicsClientId=physics_client_id)

    # Recreate and compare.
    recreated_cylinder_id = p.loadURDF(
        urdf_file, (0, 0, 0), (0, 0, 0, 1), physicsClientId=physics_client_id
    )
    recovered_pose = get_pose(recreated_cylinder_id, physics_client_id)
    recovered_half_extents = get_half_extents_from_aabb(
        recreated_cylinder_id, physics_client_id
    )
    recovered_mass = p.getDynamicsInfo(
        recreated_cylinder_id, -1, physicsClientId=physics_client_id
    )[0]

    assert original_pose.allclose(recovered_pose, atol=1e-6)
    # NOTE: bounding box may be slightly enlarged in loading URDF, probably
    # because of some conservative thing that pybullet is doing.
    assert np.allclose(original_half_extents, recovered_half_extents, atol=1e-2)
    assert np.isclose(original_mass, recovered_mass)


def test_capsule_write_urdf_from_body_id(physics_client_id):
    """Tests for write_urdf_from_body_id() with a capsule."""
    radius = 0.25
    length = 0.5
    position = (0, 0, 0)
    orientation = (0, 0, 0, 1)
    color = (0.5, 0.2, 0.9, 1.0)
    mass = 1.0

    # Create the collision shape.
    collision_id = p.createCollisionShape(
        p.GEOM_CAPSULE, radius=radius, height=length, physicsClientId=physics_client_id
    )

    # Create the visual_shape.
    visual_id = p.createVisualShape(
        p.GEOM_CAPSULE,
        radius=radius,
        length=length,
        rgbaColor=color,
        physicsClientId=physics_client_id,
    )

    # Create the body.
    capsule_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
        basePosition=position,
        baseOrientation=orientation,
        physicsClientId=physics_client_id,
    )

    original_pose = get_pose(capsule_id, physics_client_id)
    original_mass = p.getDynamicsInfo(
        capsule_id, -1, physicsClientId=physics_client_id
    )[0]

    urdf = create_urdf_from_body_id(capsule_id, physics_client_id)
    urdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".urdf").name
    with open(urdf_file, mode="w", encoding="utf-8") as f:
        f.write(urdf)

    p.removeBody(capsule_id, physicsClientId=physics_client_id)

    # Recreate and compare.
    recreated_capsule_id = p.loadURDF(
        urdf_file, (0, 0, 0), (0, 0, 0, 1), physicsClientId=physics_client_id
    )

    recovered_pose = get_pose(recreated_capsule_id, physics_client_id)
    recovered_mass = p.getDynamicsInfo(
        recreated_capsule_id, -1, physicsClientId=physics_client_id
    )[0]

    assert original_pose.allclose(recovered_pose, atol=1e-6)
    assert np.isclose(original_mass, recovered_mass)


def test_fat_capsule_write_urdf_from_body_id(physics_client_id):
    """Tests for write_urdf_from_body_id() with a fat capsule."""
    radius = 0.3
    length = 0.1
    position = (0, 0, 0)
    orientation = (0, 0, 0, 1)
    color = (0.5, 0.2, 0.9, 1.0)
    mass = 1.0

    # Create the collision shape.
    collision_id = p.createCollisionShape(
        p.GEOM_CAPSULE, radius=radius, height=length, physicsClientId=physics_client_id
    )

    # Create the visual_shape.
    visual_id = p.createVisualShape(
        p.GEOM_CAPSULE,
        radius=radius,
        length=length,
        rgbaColor=color,
        physicsClientId=physics_client_id,
    )

    # Create the body.
    capsule_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
        basePosition=position,
        baseOrientation=orientation,
        physicsClientId=physics_client_id,
    )

    original_pose = get_pose(capsule_id, physics_client_id)
    original_mass = p.getDynamicsInfo(
        capsule_id, -1, physicsClientId=physics_client_id
    )[0]

    urdf = create_urdf_from_body_id(capsule_id, physics_client_id)
    urdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".urdf").name
    with open(urdf_file, mode="w", encoding="utf-8") as f:
        f.write(urdf)

    p.removeBody(capsule_id, physicsClientId=physics_client_id)

    # Recreate and compare.
    recreated_capsule_id = p.loadURDF(
        urdf_file, (0, 0, 0), (0, 0, 0, 1), physicsClientId=physics_client_id
    )

    recovered_pose = get_pose(recreated_capsule_id, physics_client_id)
    recovered_mass = p.getDynamicsInfo(
        recreated_capsule_id, -1, physicsClientId=physics_client_id
    )[0]

    assert original_pose.allclose(recovered_pose, atol=1e-6)
    assert np.isclose(original_mass, recovered_mass)


def test_capsule_on_joint_write_urdf_from_body_id_v1(physics_client_id):
    """Tests for write_urdf_from_body_id() with a capsule on a joint.

    Version 1: the offset is in the visual and collision frame.
    """
    radius_sphere = 0.25
    radius_capsule = 0.2
    length_capsule = 0.5
    position_sphere = (0, 0, 0)
    orientation_sphere = (0, 0, 0, 1)
    color_sphere = (0.2, 0.8, 0.4, 1.0)
    color_capsule = (0.5, 0.2, 0.9, 1.0)
    mass_sphere = 1.0
    mass_capsule = 0.5

    # Create the collision and visual shapes for the sphere.
    sphere_collision_id = p.createCollisionShape(
        p.GEOM_SPHERE, radius=radius_sphere, physicsClientId=physics_client_id
    )
    sphere_visual_id = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=radius_sphere,
        rgbaColor=color_sphere,
        physicsClientId=physics_client_id,
    )

    # Define the relative position and orientation of the capsule.
    capsule_offset_position = (radius_sphere + length_capsule / 2, 0, 0)
    capsule_offset_orientation = p.getQuaternionFromEuler((0.0, np.pi / 2, 0.0))

    # Create the collision and visual shapes for the capsule.
    capsule_collision_id = p.createCollisionShape(
        p.GEOM_CAPSULE,
        radius=radius_capsule,
        height=length_capsule,
        collisionFramePosition=capsule_offset_position,
        collisionFrameOrientation=capsule_offset_orientation,
        physicsClientId=physics_client_id,
    )
    capsule_visual_id = p.createVisualShape(
        p.GEOM_CAPSULE,
        radius=radius_capsule,
        length=length_capsule,
        rgbaColor=color_capsule,
        visualFramePosition=capsule_offset_position,
        visualFrameOrientation=capsule_offset_orientation,
        physicsClientId=physics_client_id,
    )

    # Create the multi-body with two links.
    body_id = p.createMultiBody(
        baseMass=mass_sphere,
        baseCollisionShapeIndex=sphere_collision_id,
        baseVisualShapeIndex=sphere_visual_id,
        basePosition=position_sphere,
        baseOrientation=orientation_sphere,
        linkMasses=[mass_capsule],
        linkCollisionShapeIndices=[capsule_collision_id],
        linkVisualShapeIndices=[capsule_visual_id],
        linkPositions=[(0, 0, 0)],
        linkOrientations=[(0, 0, 0, 1)],
        linkInertialFramePositions=[(0, 0, 0)],
        linkInertialFrameOrientations=[(0, 0, 0, 1)],
        linkParentIndices=[0],
        linkJointTypes=[p.JOINT_FIXED],
        linkJointAxis=[(0, 0, 0)],
        physicsClientId=physics_client_id,
    )

    original_pose = get_pose(body_id, physics_client_id)
    original_mass = p.getDynamicsInfo(body_id, -1, physicsClientId=physics_client_id)[0]

    urdf = create_urdf_from_body_id(body_id, physics_client_id)

    urdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".urdf").name
    with open(urdf_file, mode="w", encoding="utf-8") as f:
        f.write(urdf)

    p.removeBody(body_id, physicsClientId=physics_client_id)

    # Recreate and compare.
    recreated_capsule_id = p.loadURDF(
        urdf_file, (0, 0, 0), (0, 0, 0, 1), physicsClientId=physics_client_id
    )

    recovered_pose = get_pose(recreated_capsule_id, physics_client_id)
    recovered_mass = p.getDynamicsInfo(
        recreated_capsule_id, -1, physicsClientId=physics_client_id
    )[0]

    assert original_pose.allclose(recovered_pose, atol=1e-6)
    assert np.isclose(original_mass, recovered_mass)


def test_capsule_on_joint_write_urdf_from_body_id_v2(physics_client_id):
    """Tests for write_urdf_from_body_id() with a capsule on a joint.

    Version 1: the offset is in joint frame.
    """
    radius_sphere = 0.25
    radius_capsule = 0.2
    length_capsule = 0.5
    position_sphere = (0, 0, 0)
    orientation_sphere = (0, 0, 0, 1)
    color_sphere = (0.2, 0.8, 0.4, 1.0)
    color_capsule = (0.5, 0.2, 0.9, 1.0)
    mass_sphere = 1.0
    mass_capsule = 0.5

    # Create the collision and visual shapes for the sphere.
    sphere_collision_id = p.createCollisionShape(
        p.GEOM_SPHERE, radius=radius_sphere, physicsClientId=physics_client_id
    )
    sphere_visual_id = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=radius_sphere,
        rgbaColor=color_sphere,
        physicsClientId=physics_client_id,
    )

    # Define the relative position and orientation of the capsule.
    capsule_offset_position = (radius_sphere + length_capsule / 2, 0, 0)
    capsule_offset_orientation = (0.5, 0.5, 0.5, 0.5)

    # Create the collision and visual shapes for the capsule.
    capsule_collision_id = p.createCollisionShape(
        p.GEOM_CAPSULE,
        radius=radius_capsule,
        height=length_capsule,
        physicsClientId=physics_client_id,
    )
    capsule_visual_id = p.createVisualShape(
        p.GEOM_CAPSULE,
        radius=radius_capsule,
        length=length_capsule,
        rgbaColor=color_capsule,
        physicsClientId=physics_client_id,
    )

    # Create the multi-body with two links.
    body_id = p.createMultiBody(
        baseMass=mass_sphere,
        baseCollisionShapeIndex=sphere_collision_id,
        baseVisualShapeIndex=sphere_visual_id,
        basePosition=position_sphere,
        baseOrientation=orientation_sphere,
        linkMasses=[mass_capsule],
        linkCollisionShapeIndices=[capsule_collision_id],
        linkVisualShapeIndices=[capsule_visual_id],
        linkPositions=[capsule_offset_position],
        linkOrientations=[capsule_offset_orientation],
        linkInertialFramePositions=[(0, 0, 0)],
        linkInertialFrameOrientations=[(0, 0, 0, 1)],
        linkParentIndices=[0],
        linkJointTypes=[p.JOINT_FIXED],
        linkJointAxis=[(0, 0, 0)],
        physicsClientId=physics_client_id,
    )

    original_pose = get_pose(body_id, physics_client_id)
    original_mass = p.getDynamicsInfo(body_id, -1, physicsClientId=physics_client_id)[0]

    urdf = create_urdf_from_body_id(body_id, physics_client_id)

    urdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".urdf").name
    with open(urdf_file, mode="w", encoding="utf-8") as f:
        f.write(urdf)

    p.removeBody(body_id, physicsClientId=physics_client_id)

    # Recreate and compare.
    recreated_capsule_id = p.loadURDF(
        urdf_file, (0, 0, 0), (0, 0, 0, 1), physicsClientId=physics_client_id
    )

    recovered_pose = get_pose(recreated_capsule_id, physics_client_id)
    recovered_mass = p.getDynamicsInfo(
        recreated_capsule_id, -1, physicsClientId=physics_client_id
    )[0]

    assert original_pose.allclose(recovered_pose, atol=1e-6)
    assert np.isclose(original_mass, recovered_mass)


def test_revolute_joint_write_urdf_from_body_id(physics_client_id):
    """Tests for write_urdf_from_body_id() with a revolute joint."""

    original_urdf = """
    <?xml version="1.0"?>
<robot name="one-joint-urdf">
  <link name="base_link">
    <visual>
      <origin xyz="0.0 0.0 0.0" rpy="0.0 -0.0 0.0"/>
      <geometry>
        <cylinder radius="0.1" length="0.2"/>
      </geometry>
      <material name="color_0">
        <color rgba="0.0 0.0 1.0 1.0"/>
     </material>
    </visual>
    <collision>
      <origin xyz="0.0 0.0 0.0" rpy="0.0 -0.0 0.0"/>
      <geometry>
        <cylinder radius="0.1" length="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0.0 0.0 0.0" rpy="0.0 -0.0 0.0"/>
      <mass value="0.0"/>
      <inertia ixx="0.0" ixy="0.0" ixz="0.0" iyy="0.0" iyz="0.0" izz="0.0"/>
    </inertial>
  </link>
  <link name="link1">
    <visual>
      <origin xyz="0.5 0.0 0.0" rpy="0.0 -0.0 0.0"/>
      <geometry>
        <box size="1.0 0.1 0.1"/>
      </geometry>
      <material name="color_1">
        <color rgba="0.0 0.0 1.0 1.0"/>
     </material>
    </visual>
    <collision>
      <origin xyz="1.0 0.0 0.0" rpy="0.0 -0.0 0.0"/>
      <geometry>
        <box size="1.0 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="-0.5 0.0 0.0" rpy="0.0 -0.0 0.0"/>
      <mass value="1.0"/>
      <inertia ixx="0.0" ixy="0.0" ixz="0.0" iyy="0.0" iyz="0.0" izz="0.0"/>
    </inertial>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0.0 0.0 0.0" rpy="0.0 -0.0 0.0"/>
    <axis xyz="0.0 0.0 1.0"/>
    <limit effort="1000.0" velocity="1.0" lower="-3.14" upper="3.14"/>
  </joint>
</robot>
"""
    original_urdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".urdf").name
    with open(original_urdf_file, mode="w", encoding="utf-8") as f:
        f.write(original_urdf)

    original_robot_id = p.loadURDF(
        original_urdf_file, (0, 0, 0), (0, 0, 0, 1), physicsClientId=physics_client_id
    )

    original_pose = get_pose(original_robot_id, physics_client_id)
    original_half_extents = get_half_extents_from_aabb(
        original_robot_id, physics_client_id
    )
    original_mass = p.getDynamicsInfo(
        original_robot_id, -1, physicsClientId=physics_client_id
    )[0]

    urdf = create_urdf_from_body_id(original_robot_id, physics_client_id)
    urdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".urdf").name
    with open(urdf_file, mode="w", encoding="utf-8") as f:
        f.write(urdf)

    p.removeBody(original_robot_id, physicsClientId=physics_client_id)

    # Recreate and compare.
    recreated_robot_id = p.loadURDF(
        urdf_file, (0, 0, 0), (0, 0, 0, 1), physicsClientId=physics_client_id
    )
    recovered_pose = get_pose(recreated_robot_id, physics_client_id)
    recovered_half_extents = get_half_extents_from_aabb(
        recreated_robot_id, physics_client_id
    )
    recovered_mass = p.getDynamicsInfo(
        recreated_robot_id, -1, physicsClientId=physics_client_id
    )[0]

    assert original_pose.allclose(recovered_pose, atol=1e-6)
    # NOTE: bounding box may be slightly enlarged in loading URDF, probably
    # because of some conservative thing that pybullet is doing.
    assert np.allclose(original_half_extents, recovered_half_extents, atol=1e-2)
    assert np.isclose(original_mass, recovered_mass)


def test_two_link_robot_write_urdf_from_body_id(physics_client_id):
    """Tests for write_urdf_from_body_id() with a two-link robot."""
    robot = create_pybullet_robot("two-link", physics_client_id)
    original_robot_id = robot.robot_id
    original_pose = get_pose(original_robot_id, physics_client_id)

    urdf = create_urdf_from_body_id(original_robot_id, physics_client_id)
    urdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".urdf").name
    with open(urdf_file, mode="w", encoding="utf-8") as f:
        f.write(urdf)
    original_pose = get_pose(original_robot_id, physics_client_id)
    original_half_extents = get_half_extents_from_aabb(
        original_robot_id, physics_client_id
    )
    original_mass = p.getDynamicsInfo(
        original_robot_id, -1, physicsClientId=physics_client_id
    )[0]

    p.removeBody(original_robot_id, physicsClientId=physics_client_id)

    # Recreate and compare.
    recreated_robot_id = p.loadURDF(
        urdf_file, (0, 0, 0), (0, 0, 0, 1), physicsClientId=physics_client_id
    )
    recovered_pose = get_pose(recreated_robot_id, physics_client_id)
    recovered_half_extents = get_half_extents_from_aabb(
        recreated_robot_id, physics_client_id
    )
    recovered_mass = p.getDynamicsInfo(
        recreated_robot_id, -1, physicsClientId=physics_client_id
    )[0]

    assert original_pose.allclose(recovered_pose, atol=1e-6)
    # NOTE: bounding box may be slightly enlarged in loading URDF, probably
    # because of some conservative thing that pybullet is doing.
    assert np.allclose(original_half_extents, recovered_half_extents, atol=1e-2)
    assert np.isclose(original_mass, recovered_mass)
