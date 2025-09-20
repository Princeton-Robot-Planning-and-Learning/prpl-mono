"""Code for exporting to URDF.

NOTE: this code is less tested than other code in this repository. It's very
likely that there are bugs.
"""

import os
from dataclasses import dataclass

import pybullet as p

from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.joint import JointInfo, get_joint_info, get_num_joints
from pybullet_helpers.link import get_link_state


@dataclass
class URDFStringContainer:
    """Light container around URDF strings."""

    link_strs: list[tuple[str, int]]
    joint_strs: list[tuple[str, int]]

    def to_string(self, urdf_name: str = "pybullet-extracted") -> str:
        """Get the full URDF string."""
        urdf_lines = [('<?xml version="1.0"?>', 0)]
        urdf_lines.append((f'<robot name="{urdf_name}">', 0))
        urdf_lines.extend(self.link_strs)
        urdf_lines.extend(self.joint_strs)
        urdf_lines.append(("</robot>", 0))
        urdf_content = ""
        for urdf_str, indent in urdf_lines:
            urdf_content += " " * indent + urdf_str + "\n"
        return urdf_content


def _get_visual_shape_data_for_link(
    body_id: int, link_idx: int, physics_client_id: int
) -> list:
    return [
        v
        for v in p.getVisualShapeData(body_id, physicsClientId=physics_client_id)
        if v[1] == link_idx
    ]


def _get_collision_shape_data_for_link(
    body_id: int, link_idx: int, physics_client_id: int
) -> list:
    # I'm not sure why this filtering is needed. Sometimes there are returned
    # collision values with link IDs that don't match the input...
    return [
        v
        for v in p.getCollisionShapeData(
            body_id, link_idx, physicsClientId=physics_client_id
        )
        if v[1] == link_idx
    ]


def _get_urdf_pose_str(
    pos: tuple[float, float, float], orn: tuple[float, float, float, float]
) -> str:
    x, y, z = pos
    pose = Pose(pos, orn)
    roll, pitch, yaw = pose.rpy
    return f'<origin xyz="{x} {y} {z}" ' f'rpy="{roll} {pitch} {yaw}"/>'


def _get_urdf_geometry_str(
    geom_type: int, dims: tuple[float, float, float], filename: str
) -> str:
    if geom_type == p.GEOM_SPHERE:
        geometry_str = f'<sphere radius="{dims[0]}"/>'
    elif geom_type == p.GEOM_BOX:
        # dims are half-extents.
        geometry_str = f'<box size="{dims[0]} {dims[1]} {dims[2]}"/>'
    elif geom_type == p.GEOM_CYLINDER:
        # dims = [length, radius].
        length = dims[0]
        radius = dims[1]
        geometry_str = f'<cylinder radius="{radius}" length="{length}"/>'
    elif geom_type == p.GEOM_MESH:
        # dims for mesh are scaling factors.
        if filename == "unknown_file":
            raise ValueError("Tried to create mesh with unknown file.")
        filepath = os.path.relpath(filename)
        geometry_str = (
            f'<mesh filename="{filepath}" scale="{dims[0]} {dims[1]} {dims[2]}"/>'
        )
    else:
        raise ValueError(f"Unsupported geom type: {geom_type}")
    return geometry_str


def _get_urdf_joint_type(joint_type: int) -> str:
    if joint_type == p.JOINT_REVOLUTE:
        jtype_str = "revolute"
    elif joint_type == p.JOINT_PRISMATIC:
        jtype_str = "prismatic"
    elif joint_type == p.JOINT_FIXED:
        jtype_str = "fixed"
    else:
        raise ValueError(f"Unsupported joint type: {joint_type}")
    return jtype_str


def _add_urdf_lines_for_link(
    body_id: int,
    link_idx: int,
    physics_client_id: int,
    container: URDFStringContainer,
) -> None:
    # Get the link name.
    if link_idx == -1:
        link_name = "base_link"
    else:
        joint_info = get_joint_info(body_id, link_idx, physics_client_id)
        link_name = joint_info.linkName
        assert link_name != "base_link", "Cannot have multiple base links"

    # Get visual data.
    all_visual_data = _get_visual_shape_data_for_link(
        body_id, link_idx, physics_client_id
    )
    assert len(all_visual_data) <= 1
    visual_data = all_visual_data[0] if all_visual_data else None

    # Get collision data.
    all_collision_data = _get_collision_shape_data_for_link(
        body_id, link_idx, physics_client_id
    )
    assert len(all_collision_data) <= 1
    collision_data = all_collision_data[0] if all_collision_data else None

    # Get inertial data.
    inertial_data = p.getDynamicsInfo(
        body_id, link_idx, physicsClientId=physics_client_id
    )

    # Handle capsules separately: split them into three links connected with
    # fixed joints.
    if visual_data and visual_data[2] == p.GEOM_CAPSULE:
        assert collision_data is not None
        _add_urdf_lines_for_capsule(
            link_name,
            visual_data,
            collision_data,
            inertial_data,
            physics_client_id,
            container,
        )

    else:
        urdf_lines = _get_urdf_lines_from_link_data(
            link_name, visual_data, collision_data, inertial_data, physics_client_id
        )

        container.link_strs.extend(urdf_lines)


def _add_urdf_lines_for_capsule(
    link_name: str,
    visual_data: tuple,
    collision_data: tuple,
    inertial_data: tuple,
    physics_client_id: int,
    container: URDFStringContainer,
) -> None:

    # Create cylinder.
    cylinder_link_name = link_name  # use original link to preserve joints
    cylinder_dims = [visual_data[3][0], visual_data[3][1], visual_data[3][2]]

    cylinder_visual_data = (
        visual_data[0],
        visual_data[1],
        p.GEOM_CYLINDER,
        cylinder_dims,  # dims
        visual_data[4],  # filename
        visual_data[5],  # pos
        visual_data[6],  # orn
        visual_data[7],  # color
    )

    cylinder_collision_data = (
        collision_data[0],
        collision_data[1],
        p.GEOM_CYLINDER,
        cylinder_dims,  # dims
        collision_data[4],  # filename
        collision_data[5],  # pos
        collision_data[6],  # orn
    )

    cylinder_inertial_data = (
        inertial_data[0],  # mass
        inertial_data[1],  # lateral friction
        inertial_data[2],  # local inertial diagonal
        inertial_data[3],  # pos
        inertial_data[4],  # orn
    )

    container.link_strs.extend(
        _get_urdf_lines_from_link_data(
            cylinder_link_name,
            cylinder_visual_data,
            cylinder_collision_data,
            cylinder_inertial_data,
            physics_client_id,
        )
    )

    # Create top sphere.
    top_sphere_link_name = f"{link_name}---top-sphere"

    top_sphere_pos = (0, 0, 0)
    top_sphere_orn = (0, 0, 0, 1)

    top_sphere_visual_data = (
        visual_data[0],
        visual_data[1],
        p.GEOM_SPHERE,
        [cylinder_dims[1], 0.0, 0.0],  # dims
        visual_data[4],  # filename
        top_sphere_pos,  # pos
        top_sphere_orn,  # orn
        visual_data[7],  # color
    )

    top_sphere_collision_data = (
        collision_data[0],
        collision_data[1],
        p.GEOM_SPHERE,
        [cylinder_dims[1], 0.0, 0.0],  # dims
        collision_data[4],  # filename
        top_sphere_pos,  # pos
        top_sphere_orn,  # orn
    )

    top_sphere_inertial_data = (
        0.0,  # mass
        0.0,  # lateral friction
        (0, 0, 0),  # local inertial diagonal
        top_sphere_pos,  # pos
        top_sphere_orn,  # orn
    )

    container.link_strs.extend(
        _get_urdf_lines_from_link_data(
            top_sphere_link_name,
            top_sphere_visual_data,
            top_sphere_collision_data,
            top_sphere_inertial_data,
            physics_client_id,
        )
    )

    # Add fixed joint between top sphere and cylinder.
    top_sphere_pose = Pose((0, 0, cylinder_dims[1] / 2))
    visual_frame = Pose(visual_data[5], visual_data[6])
    top_sphere_pose = multiply_poses(visual_frame, top_sphere_pose)

    top_sphere_joint_name = top_sphere_link_name + "-fixed-joint"
    top_sphere_joint_info = JointInfo(
        jointIndex=-1,  # not used
        jointName=top_sphere_joint_name,
        jointType=p.JOINT_FIXED,
        qIndex=-1,  # not used
        uIndex=-1,  # not used
        flags=-1,  # not used
        jointDamping=0.0,
        jointFriction=0.0,
        jointLowerLimit=0.0,
        jointUpperLimit=-1.0,
        jointMaxForce=0.0,
        jointMaxVelocity=0.0,
        linkName=top_sphere_link_name,
        jointAxis=(0, 0, 0),
        parentFramePos=top_sphere_pose.position,
        parentFrameOrn=top_sphere_pose.orientation,
        parentIndex=visual_data[1],
    )
    top_sphere_joint_urdf_lines = _get_joint_urdf_from_data(
        top_sphere_joint_info,
        parent_name=cylinder_link_name,
        parent_inertial_frame=Pose.identity(),
    )

    container.joint_strs.extend(top_sphere_joint_urdf_lines)

    # Create bottom sphere.
    bottom_sphere_link_name = f"{link_name}---bottom-sphere"

    bottom_sphere_pos = (0, 0, 0)
    bottom_sphere_orn = (0, 0, 0, 1)

    bottom_sphere_visual_data = (
        visual_data[0],
        visual_data[1],
        p.GEOM_SPHERE,
        [cylinder_dims[1], 0.0, 0.0],  # dims
        visual_data[4],  # filename
        bottom_sphere_pos,  # pos
        bottom_sphere_orn,  # orn
        visual_data[7],  # color
    )

    bottom_sphere_collision_data = (
        collision_data[0],
        collision_data[1],
        p.GEOM_SPHERE,
        [cylinder_dims[1], 0.0, 0.0],  # dims
        collision_data[4],  # filename
        bottom_sphere_pos,  # pos
        bottom_sphere_orn,  # orn
    )

    bottom_sphere_inertial_data = (
        0.0,  # mass
        0.0,  # lateral friction
        (0, 0, 0),  # local inertial diagonal
        bottom_sphere_pos,  # pos
        bottom_sphere_orn,  # orn
    )

    container.link_strs.extend(
        _get_urdf_lines_from_link_data(
            bottom_sphere_link_name,
            bottom_sphere_visual_data,
            bottom_sphere_collision_data,
            bottom_sphere_inertial_data,
            physics_client_id,
        )
    )

    # Add fixed joint between top sphere and cylinder.
    bottom_sphere_pose = Pose((0, 0, -cylinder_dims[1] / 2))
    visual_frame = Pose(visual_data[5], visual_data[6])
    bottom_sphere_pose = multiply_poses(visual_frame, bottom_sphere_pose)

    bottom_sphere_joint_name = bottom_sphere_link_name + "-fixed-joint"
    bottom_sphere_joint_info = JointInfo(
        jointIndex=-1,  # not used
        jointName=bottom_sphere_joint_name,
        jointType=p.JOINT_FIXED,
        qIndex=-1,  # not used
        uIndex=-1,  # not used
        flags=-1,  # not used
        jointDamping=0.0,
        jointFriction=0.0,
        jointLowerLimit=0.0,
        jointUpperLimit=-1.0,
        jointMaxForce=0.0,
        jointMaxVelocity=0.0,
        linkName=bottom_sphere_link_name,
        jointAxis=(0, 0, 0),
        parentFramePos=bottom_sphere_pose.position,
        parentFrameOrn=bottom_sphere_pose.orientation,
        parentIndex=visual_data[1],
    )
    bottom_sphere_joint_urdf_lines = _get_joint_urdf_from_data(
        bottom_sphere_joint_info,
        parent_name=cylinder_link_name,
        parent_inertial_frame=Pose.identity(),
    )

    container.joint_strs.extend(bottom_sphere_joint_urdf_lines)


def _get_urdf_lines_from_link_data(
    link_name: str,
    visual_data: tuple | None,
    collision_data: tuple | None,
    inertial_data: tuple,
    physics_client_id: int,
) -> list[tuple[str, int]]:
    # Start the URDF for the link.
    urdf_lines = [(f'<link name="{link_name}">', 2)]

    # Add visual tags.
    if visual_data:
        visual_urdf_lines = _get_visual_urdf_lines_for_link(visual_data)
        urdf_lines.extend(visual_urdf_lines)

    # Add collision tags.
    if collision_data:
        collision_urdf_lines = _get_collision_urdf_lines_for_link(
            collision_data, physics_client_id
        )
        urdf_lines.extend(collision_urdf_lines)

    # Add inertial tag.
    inertial_urdf_lines = _get_inertial_urdf_lines_for_link(inertial_data)
    urdf_lines.extend(inertial_urdf_lines)

    # Finish the URDF for the link.
    urdf_lines.append(("</link>", 2))
    return urdf_lines


def _get_visual_urdf_lines_for_link(
    visual_data: tuple,
) -> list[tuple[str, int]]:
    # Visual shape data format:
    # v = (bodyUniqueId, linkIndex, visualGeometryType, dimensions, filename,
    #      localVisualFramePos, localVisualFrameOrn, color, specular).
    geom_type = visual_data[2]
    dims = visual_data[3]
    filename = visual_data[4].decode("UTF-8")
    pos = visual_data[5]
    orn = visual_data[6]
    color = visual_data[7]

    return _get_single_visual_urdf_lines_for_link(
        geom_type, dims, filename, pos, orn, color
    )


def _get_single_visual_urdf_lines_for_link(
    geom_type: int,
    dims: tuple[float, float, float],
    filename: str,
    pos: tuple[float, float, float],
    orn: tuple[float, float, float, float],
    color: tuple[float, float, float, float],
) -> list[tuple[str, int]]:
    urdf_lines = [("<visual>", 4)]

    # Add pose.
    pose_str = _get_urdf_pose_str(pos, orn)
    urdf_lines.append((pose_str, 6))

    # Add geometry.
    geometry_str = _get_urdf_geometry_str(geom_type, dims, filename)
    urdf_lines.append(("<geometry>", 6))
    urdf_lines.append((geometry_str, 8))
    urdf_lines.append(("</geometry>", 6))

    # Add color.
    r, g, b, a = color
    color_name = f"color-{int(255*r)}-{int(255*g)}-{int(255*b)}-{int(255*a)}"
    color_tag = f'<color rgba="{r} {g} {b} {a}"/>'
    urdf_lines.append((f'<material name="{color_name}">', 6))
    urdf_lines.append((color_tag, 8))
    urdf_lines.append(("</material>", 6))

    urdf_lines.append(("</visual>", 4))

    return urdf_lines


def _get_collision_urdf_lines_for_link(
    collision_data: tuple,
    physics_client_id: int,
) -> list[tuple[str, int]]:
    # Collision shape data format:
    # c = (object id, linkIndex, geometryType, dimensions, filename,
    #      localCollisionFramePos, localCollisionFrameOrn).
    geom_type = collision_data[2]
    dims = collision_data[3]
    filename = collision_data[4].decode("UTF-8")
    com_to_pose = Pose(collision_data[5], collision_data[6])
    if collision_data[1] == -1:
        origin_to_pose = com_to_pose
    else:
        link_state = get_link_state(
            collision_data[0], collision_data[1], physics_client_id
        )
        tf = Pose(
            link_state.localInertialFramePosition,
            link_state.localInertialFrameOrientation,
        )
        origin_to_pose = multiply_poses(tf, com_to_pose)
    pos = origin_to_pose.position
    orn = origin_to_pose.orientation

    return _get_single_collision_urdf_lines_for_link(
        collision_data[0],
        collision_data[1],
        physics_client_id,
        geom_type,
        dims,
        filename,
        pos,
        orn,
    )


def _get_single_collision_urdf_lines_for_link(
    body_id: int,
    link_idx: int,
    physics_client_id: int,
    geom_type: int,
    dims: tuple[float, float, float],
    filename: str,
    pos: tuple[float, float, float],
    orn: tuple[float, float, float, float],
) -> list[tuple[str, int]]:
    urdf_lines = [("<collision>", 4)]

    # Add pose.
    pose_str = _get_urdf_pose_str(pos, orn)
    urdf_lines.append((pose_str, 6))

    # Add geometry.
    try:
        geometry_str = _get_urdf_geometry_str(geom_type, dims, filename)
    except ValueError:
        # PyBullet seems to internally replace cylinders with meshes
        # when loading from URDF. In this case the best we can do is
        # steal the geometry from visual shape data.
        matching_link_visuals = _get_visual_shape_data_for_link(
            body_id, link_idx, physics_client_id
        )
        if len(matching_link_visuals) == 1:
            v = matching_link_visuals[0]
            geom_type = v[2]
            dims = v[3]
            if geom_type == p.GEOM_MESH:
                print("WARNING: using sphere instead of unknown mesh.")
                geom_type = p.GEOM_SPHERE
            else:
                print("WARNING: using visual geometry for collisions.")
            geometry_str = _get_urdf_geometry_str(geom_type, dims, filename)
        else:
            raise NotImplementedError

    urdf_lines.append(("<geometry>", 6))
    urdf_lines.append((geometry_str, 8))
    urdf_lines.append(("</geometry>", 6))

    urdf_lines.append(("</collision>", 4))

    return urdf_lines


def _get_inertial_urdf_lines_for_link(
    inertial_data: tuple,
) -> list[tuple[str, int]]:
    mass = inertial_data[0]
    ixx, iyy, izz = inertial_data[2]
    inertial_pos = inertial_data[3]
    inertial_orn = inertial_data[4]

    urdf_lines = [("<inertial>", 4)]
    inertial_pose_str = _get_urdf_pose_str(inertial_pos, inertial_orn)
    mass_str = f'<mass value="{mass}"/>'
    inertia_str = (
        f'<inertia ixx="{ixx}" ixy="0.0" ixz="0.0" '
        f'iyy="{iyy}" iyz="0.0" izz="{izz}"/>'
    )
    urdf_lines.append((inertial_pose_str, 6))
    urdf_lines.append((mass_str, 6))
    urdf_lines.append((inertia_str, 6))
    urdf_lines.append(("</inertial>", 4))

    return urdf_lines


def _add_urdf_lines_for_joint(
    body_id: int,
    joint_idx: int,
    physics_client_id: int,
    container: URDFStringContainer,
) -> None:
    joint_info = get_joint_info(body_id, joint_idx, physics_client_id)
    parent_idx = joint_info.parentIndex

    if parent_idx == -1:
        parent_name = "base_link"
        parent_inertial_frame = Pose.identity()

    else:
        parent_name = get_joint_info(body_id, parent_idx, physics_client_id).linkName
        parent_state = get_link_state(body_id, parent_idx, physics_client_id)
        parent_inertial_frame = Pose(
            parent_state.localInertialFramePosition,
            parent_state.localInertialFrameOrientation,
        )

    urdf_lines = _get_joint_urdf_from_data(
        joint_info, parent_name, parent_inertial_frame
    )
    container.joint_strs.extend(urdf_lines)


def _get_joint_urdf_from_data(
    joint_info: JointInfo, parent_name: str, parent_inertial_frame: Pose
) -> list[tuple[str, int]]:
    joint_name = joint_info.jointName
    joint_type = joint_info.jointType
    jtype_str = _get_urdf_joint_type(joint_type)
    child_name = joint_info.linkName
    parent_frame_pos = joint_info.parentFramePos
    parent_frame_orn = joint_info.parentFrameOrn
    parent_frame_pose = Pose(parent_frame_pos, parent_frame_orn)

    inverted_orn = Pose((0, 0, 0), parent_frame_orn).invert().orientation
    parent_frame_pose = Pose(parent_frame_pos, inverted_orn)

    parent_frame_pose = multiply_poses(parent_inertial_frame, parent_frame_pose)

    pose_str = _get_urdf_pose_str(
        parent_frame_pose.position, parent_frame_pose.orientation
    )

    urdf_lines = [(f'<joint name="{joint_name}" type="{jtype_str}">', 2)]
    urdf_lines.append((f'<parent link="{parent_name}"/>', 4))
    urdf_lines.append((f'<child link="{child_name}"/>', 4))
    urdf_lines.append((pose_str, 4))

    # If not fixed, add the axis and limits.
    if jtype_str in ["revolute", "prismatic"]:
        joint_axis = joint_info.jointAxis
        axis_str = f'<axis xyz="{joint_axis[0]} {joint_axis[1]} {joint_axis[2]}"/>'
        urdf_lines.append((axis_str, 4))

        lower = joint_info.jointLowerLimit
        upper = joint_info.jointUpperLimit
        max_velocity = joint_info.jointMaxVelocity
        max_effort = joint_info.jointMaxForce
        limit_str = (
            f'<limit effort="{max_effort}" '
            f'velocity="{max_velocity}" lower="{lower}" upper="{upper}"/>'
        )
        urdf_lines.append((limit_str, 4))

    urdf_lines.append(("</joint>", 2))
    return urdf_lines


def create_urdf_from_body_id(
    body_id: int, physics_client_id: int, name: str = "pybullet-extracted"
) -> str:
    """Create a URDF string for a body that's loaded into pybullet."""

    # Start building the URDF content.
    container = URDFStringContainer([], [])
    num_joints = get_num_joints(body_id, physics_client_id)

    # Handle each link, including the base link (which is index -1 in PyBullet).
    for link_idx in range(-1, num_joints):
        _add_urdf_lines_for_link(body_id, link_idx, physics_client_id, container)

    # Handle joints.
    for joint_idx in range(num_joints):
        _add_urdf_lines_for_joint(body_id, joint_idx, physics_client_id, container)

    return container.to_string(name)
