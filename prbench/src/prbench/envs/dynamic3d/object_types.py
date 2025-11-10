"""Define object types for the TidyBot environment."""

from relational_structs import Type

MujocoObjectTypeFeatures: dict[Type, list[str]] = {}

MujocoObjectType = Type("mujoco_object")
MujocoObjectTypeFeatures[MujocoObjectType] = [
    # Position.
    "x",
    "y",
    "z",
    # Orientation (quaternion).
    "qw",
    "qx",
    "qy",
    "qz",
    # Linear velocity.
    "vx",
    "vy",
    "vz",
    # Angular velocity.
    "wx",
    "wy",
    "wz",
    # Bounding box dimensions (full, not half).
    "bb_x",
    "bb_y",
    "bb_z",
]

MujocoFixtureObjectType = Type("mujoco_fixture")
MujocoObjectTypeFeatures[MujocoFixtureObjectType] = [
    # Position.
    "x",
    "y",
    "z",
    # Orientation (quaternion).
    "qw",
    "qx",
    "qy",
    "qz",
]

MujocoTidyBotRobotObjectType = Type("mujoco_tidybot_robot")
MujocoObjectTypeFeatures[MujocoTidyBotRobotObjectType] = [
    "pos_base_x",
    "pos_base_y",
    "pos_base_rot",
    "pos_arm_joint1",
    "pos_arm_joint2",
    "pos_arm_joint3",
    "pos_arm_joint4",
    "pos_arm_joint5",
    "pos_arm_joint6",
    "pos_arm_joint7",
    "pos_gripper",
    "vel_base_x",
    "vel_base_y",
    "vel_base_rot",
    "vel_arm_joint1",
    "vel_arm_joint2",
    "vel_arm_joint3",
    "vel_arm_joint4",
    "vel_arm_joint5",
    "vel_arm_joint6",
    "vel_arm_joint7",
    "vel_gripper",
]

MujocoRBY1ARobotObjectType = Type("mujoco_rby1a_robot")
MujocoObjectTypeFeatures[MujocoRBY1ARobotObjectType] = [
    "pos_base_right",
    "pos_base_left",
    # TODO add more attributes  # pylint: disable=fixme
]
