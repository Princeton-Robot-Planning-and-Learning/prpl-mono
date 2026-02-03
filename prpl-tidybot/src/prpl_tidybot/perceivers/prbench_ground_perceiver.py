"""A perceiver for the PRBench Dynamic 3D Ground environment."""

from prbench.envs.dynamic3d.object_types import (
    MujocoMovableObjectType,
    MujocoObjectTypeFeatures,
    MujocoTidyBotRobotObjectType,
)
from prbench.envs.geom3d.ground3d import Ground3DObjectCentricState
from prbench.envs.geom3d.object_types import (
    Geom3DCuboidType,
    Geom3DEnvTypeFeatures,
    Geom3DRobotType,
)
from prbench.envs.geom3d.transport3d import Transport3DObjectCentricState
from prbench.envs.geom3d.utils import Geom3DObjectCentricState
from relational_structs import Object, ObjectCentricState
from relational_structs.utils import create_state_from_dict

from prpl_tidybot.interfaces.interface import Interface
from prpl_tidybot.perceivers.base_perceiver import Perceiver


class PRBenchGroundPerceiver(Perceiver[ObjectCentricState]):
    """A perceiver for the PRBench Dynamic 3D Ground environment."""

    def __init__(self, interface: Interface) -> None:
        self._interface = interface

    def get_state(self) -> ObjectCentricState:
        state_dict: dict[Object, dict[str, float]] = {}

        # Extract the robot state.
        qpos_base = self._interface.get_map_base_state()
        qpos_arm = self._interface.get_arm_state()
        gripper_state = self._interface.get_gripper_state()

        # Add robot into object-centric state.
        robot = Object("robot", MujocoTidyBotRobotObjectType)

        # Build this super explicitly, even though verbose, to be careful.
        state_dict[robot] = {
            "pos_base_x": qpos_base.x,
            "pos_base_y": qpos_base.y,
            "pos_base_rot": qpos_base.theta(),
            "pos_arm_joint1": qpos_arm[0],
            "pos_arm_joint2": qpos_arm[1],
            "pos_arm_joint3": qpos_arm[2],
            "pos_arm_joint4": qpos_arm[3],
            "pos_arm_joint5": qpos_arm[4],
            "pos_arm_joint6": qpos_arm[5],
            "pos_arm_joint7": qpos_arm[6],
            "pos_gripper": gripper_state,
            # NOTE: velocity not actually used or measured in real.
            "vel_base_x": 0.0,
            "vel_base_y": 0.0,
            "vel_base_rot": 0.0,
            "vel_arm_joint1": 0.0,
            "vel_arm_joint2": 0.0,
            "vel_arm_joint3": 0.0,
            "vel_arm_joint4": 0.0,
            "vel_arm_joint5": 0.0,
            "vel_arm_joint6": 0.0,
            "vel_arm_joint7": 0.0,
            "vel_gripper": 0.0,
        }

        # Placeholder for actual object detection! Coming soon!!!
        cube = Object("cube1", MujocoMovableObjectType)
        state_dict[cube] = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "qw": 1.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
            "wx": 0.0,
            "wy": 0.0,
            "wz": 0.0,
            "bb_x": 0.03,
            "bb_y": 0.03,
            "bb_z": 0.03,
        }

        return create_state_from_dict(state_dict, MujocoObjectTypeFeatures)


class PRBenchGeom3DPerceiver(Perceiver[ObjectCentricState]):
    """A perceiver for the PRBench Geom3D Ground environment."""

    def __init__(self, interface: Interface) -> None:
        self._interface = interface

    def get_state(self) -> ObjectCentricState:
        state_dict: dict[Object, dict[str, float]] = {}

        # Extract the robot state.
        qpos_base = self._interface.get_map_base_state()
        qpos_arm = self._interface.get_arm_state()
        gripper_state = self._interface.get_gripper_state()

        # Add robot into object-centric state.
        robot = Object("robot", Geom3DRobotType)

        # Build this super explicitly, even though verbose, to be careful.
        state_dict[robot] = {
            "pos_base_x": qpos_base.x,
            "pos_base_y": qpos_base.y,
            "pos_base_rot": qpos_base.theta(),
            "joint_1": qpos_arm[0],
            "joint_2": qpos_arm[1],
            "joint_3": qpos_arm[2],
            "joint_4": qpos_arm[3],
            "joint_5": qpos_arm[4],
            "joint_6": qpos_arm[5],
            "joint_7": qpos_arm[6],
            "finger_state": gripper_state,
            # NOTE: velocity not actually used or measured in real.
            "grasp_active": 0.0,
            "grasp_tf_x": 0.0,
            "grasp_tf_y": 0.0,
            "grasp_tf_z": 0.0,
            "grasp_tf_qx": 0.0,
            "grasp_tf_qy": 0.0,
            "grasp_tf_qz": 0.0,
            "grasp_tf_qw": 1.0,
        }

        # Placeholder for actual object detection! Coming soon!!!
        cube = Object("cube0", Geom3DCuboidType)
        state_dict[cube] = {
            "pose_x": 0.0,
            "pose_y": 0.0,
            "pose_z": 0.03,
            "pose_qx": 0.0,
            "pose_qy": 0.0,
            "pose_qz": 0.0,
            "pose_qw": 1.0,
            "grasp_active": 0.0,
            "object_type": 0,
            "half_extent_x": 0.03,
            "half_extent_y": 0.03,
            "half_extent_z": 0.03,
        }

        return create_state_from_dict(
            state_dict, Geom3DEnvTypeFeatures, state_cls=Ground3DObjectCentricState
        )


class PRBenchTransport3DPerceiver(Perceiver[ObjectCentricState]):
    """A perceiver for the PRBench Geom3D Transport3D environment."""

    def __init__(self, interface: Interface) -> None:
        self._interface = interface

    def get_state(self) -> ObjectCentricState:
        state_dict: dict[Object, dict[str, float]] = {}

        # Extract the robot state.
        qpos_base = self._interface.get_map_base_state()
        qpos_arm = self._interface.get_arm_state()
        gripper_state = self._interface.get_gripper_state()

        # Add robot into object-centric state.
        robot = Object("robot", Geom3DRobotType)

        # Build this super explicitly, even though verbose, to be careful.
        state_dict[robot] = {
            "pos_base_x": qpos_base.x,
            "pos_base_y": qpos_base.y,
            "pos_base_rot": qpos_base.theta(),
            "joint_1": qpos_arm[0],
            "joint_2": qpos_arm[1],
            "joint_3": qpos_arm[2],
            "joint_4": qpos_arm[3],
            "joint_5": qpos_arm[4],
            "joint_6": qpos_arm[5],
            "joint_7": qpos_arm[6],
            "finger_state": gripper_state,
            # NOTE: velocity not actually used or measured in real.
            "grasp_active": 0.0,
            "grasp_tf_x": 0.0,
            "grasp_tf_y": 0.0,
            "grasp_tf_z": 0.0,
            "grasp_tf_qx": 0.0,
            "grasp_tf_qy": 0.0,
            "grasp_tf_qz": 0.0,
            "grasp_tf_qw": 1.0,
        }

        # Placeholder for actual object detection! Coming soon!!!
        cube = Object("cube0", Geom3DCuboidType)
        state_dict[cube] = {
            "pose_x": 0.0,
            "pose_y": -0.3,
            "pose_z": 0.03,
            "pose_qx": 0.0,
            "pose_qy": 0.0,
            "pose_qz": 0.0,
            "pose_qw": 1.0,
            "grasp_active": 0.0,
            "object_type": 0,
            "half_extent_x": 0.03,
            "half_extent_y": 0.03,
            "half_extent_z": 0.03,
        }

        # box on the ground
        box = Object("box0", Geom3DCuboidType)
        state_dict[box] = {
            "pose_x": 0.0,
            "pose_y": 0.6,
            "pose_z": 0.1,
            "pose_qx": 0.0,
            "pose_qy": 0.0,
            "pose_qz": 0.0,
            "pose_qw": 1.0,
            "grasp_active": 0.0,
            "object_type": 0,
            "half_extent_x": 0.15,
            "half_extent_y": 0.2,
            "half_extent_z": 0.1,
        }

        # table
        table = Object("table", Geom3DCuboidType)
        state_dict[table] = {
            "pose_x": 0.6,
            "pose_y": 0.0,
            "pose_z": 0.2,
            "pose_qx": 0.0,
            "pose_qy": 0.0,
            "pose_qz": 0.0,
            "pose_qw": 1.0,
            "grasp_active": 0.0,
            "object_type": 0,
            "half_extent_x": 0.2,
            "half_extent_y": 0.4,
            "half_extent_z": 0.2,
        }

        return create_state_from_dict(
            state_dict, Geom3DEnvTypeFeatures, state_cls=Transport3DObjectCentricState
        )
