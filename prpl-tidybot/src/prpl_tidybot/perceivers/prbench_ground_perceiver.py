"""A perceiver for the PRBench Dynamic 3D Ground environment."""

from prbench.envs.dynamic3d.object_types import (
    MujocoObjectType,
    MujocoObjectTypeFeatures,
    MujocoRobotObjectType,
)
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
        qpos_base = self._interface.get_base_state()
        qpos_arm = self._interface.get_arm_state()
        gripper_state = self._interface.get_gripper_state()

        # Add robot into object-centric state.
        robot = Object("robot", MujocoRobotObjectType)

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
        cube = Object("cube1", MujocoObjectType)
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
            "bb_x": 0.1,
            "bb_y": 0.1,
            "bb_z": 0.1,
        }

        return create_state_from_dict(state_dict, MujocoObjectTypeFeatures)
