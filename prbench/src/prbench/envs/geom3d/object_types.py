"""Object types that are common across different environments."""

from relational_structs import Type

Geom3DEnvTypeFeatures: dict[Type, list[str]] = {}

# The robot, which is a 7DOF arm, has joint positions and grasp features.
# Note that we must store the finger state if we want to have different grasps for
# different sized objects.
Geom3DRobotType = Type("Geom3DRobot")
Geom3DEnvTypeFeatures[Geom3DRobotType] = [
    "pos_base_x",
    "pos_base_y",
    "pos_base_rot",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
    "finger_state",
    "grasp_active",
    "grasp_tf_x",
    "grasp_tf_y",
    "grasp_tf_z",
    "grasp_tf_qx",
    "grasp_tf_qy",
    "grasp_tf_qz",
    "grasp_tf_qw",
]

# Cuboid objects have poses, grasp features, and half extents.
Geom3DCuboidType = Type("Geom3DCuboid")
Geom3DEnvTypeFeatures[Geom3DCuboidType] = [
    "pose_x",
    "pose_y",
    "pose_z",
    "pose_qx",
    "pose_qy",
    "pose_qz",
    "pose_qw",
    "grasp_active",
    "object_type",
    # encoded as an int or small float category just
    # like triangle_type to make things uniform
    "half_extent_x",
    "half_extent_y",
    "half_extent_z",
]

# Triangle objects: parameterize by triangle kind and side lengths (a,b,c)
# plus a thickness/depth along Z. Pose and grasp_active included.
Geom3DTriangleType = Type("Geom3DTriangle")
Geom3DEnvTypeFeatures[Geom3DTriangleType] = [
    "pose_x",
    "pose_y",
    "pose_z",
    "pose_qx",
    "pose_qy",
    "pose_qz",
    "pose_qw",
    "grasp_active",
    # Triangle specification: either equilateral/isosceles/scalene etc.
    # The consumer can interpret these fields; they are numeric features.
    "triangle_type",  # encoded as an int or small float category
    "side_a",
    "side_b",
    "depth",
]

# A point is just a position. For example, it could be a target point to reach.
Geom3DPointType = Type("Geom3DPoint")
Geom3DEnvTypeFeatures[Geom3DPointType] = [
    "x",
    "y",
    "z",
]
