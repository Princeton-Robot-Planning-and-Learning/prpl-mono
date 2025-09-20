"""Object types that are common across different environments."""

from relational_structs import Type

Dynamic2DRobotEnvTypeFeatures: dict[Type, list[str]] = {}

# All bodies have an origin (x, y), a rotation (in radians),
# a velocity (vx, vy), an angular velocity (omega),
# a bit indicating whether the geom is static,
# a bit indicating whether the geom is kinematic,
# a bit indicating whether the geom is dynamic.
# They also have RGB.
Dynamic2DType = Type("dynamic2d")
Dynamic2DRobotEnvTypeFeatures[Dynamic2DType] = [
    "x",
    "y",
    "theta",
    "vx",
    "vy",
    "omega",
    "static",
    "color_r",
    "color_g",
    "color_b",
    "z_order",
]
# Specific types.
# For kinematic and static blocks, they don't have mass and moment.
KinRectangleType = Type("kin_rectangle", parent=Dynamic2DType)
Dynamic2DRobotEnvTypeFeatures[KinRectangleType] = Dynamic2DRobotEnvTypeFeatures[
    Dynamic2DType
] + [
    "width",
    "height",
]
# For dynamic blocks, they have mass and moment of inertia.
DynRectangleType = Type("dyn_rectangle", parent=Dynamic2DType)
Dynamic2DRobotEnvTypeFeatures[DynRectangleType] = Dynamic2DRobotEnvTypeFeatures[
    Dynamic2DType
] + [
    "width",
    "height",
    "mass",
]

# A robot with a circle base, a rectangle gripper_base, and two rectangle grippers.
# The (x, y, theta) are for the center of the robot base circle. The base_radius
# is for that circle. The arm_joint is a distance between the center and the
# gripper_base. The arm_length is the max value of arm_joint. The gripper_gap is
# the distance between the two grippers. The gripper_height and gripper_width are
# for the grippers.
KinRobotType = Type("kin_robot", parent=Dynamic2DType)
Dynamic2DRobotEnvTypeFeatures[KinRobotType] = [
    "x",
    "y",
    "theta",
    "vx",
    "vy",
    "omega",
    "static",
    "base_radius",
    "arm_joint",
    "arm_length",
    "gripper_base_width",
    "gripper_base_height",
    "finger_gap",
    "finger_height",
    "finger_width",
]
