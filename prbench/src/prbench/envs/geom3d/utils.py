"""Utilities."""

import numpy as np
from pybullet_helpers.geometry import Pose
from pybullet_helpers.joint import JointPositions
from relational_structs import Object, ObjectCentricState

from prbench.envs.geom3d.object_types import Geom3DCuboidType
from prbench.envs.utils import RobotActionSpace


class Geom3DObjectCentricState(ObjectCentricState):
    """A state in the Geom3D environment.

    Inherits from ObjectCentricState but adds some conveninent look ups.
    """

    @property
    def robot(self) -> Object:
        """Assumes there is a unique robot object named "robot"."""
        return self.get_object_from_name("robot")

    @property
    def joint_positions(self) -> JointPositions:
        """The robot joint positions."""
        joint_names = [f"joint_{i}" for i in range(1, 8)]
        return [self.get(self.robot, n) for n in joint_names]

    @property
    def finger_state(self) -> float:
        """The robot finger state."""
        return self.get(self.robot, "finger_state")

    @property
    def grasped_object(self) -> str | None:
        """The name of the currently grasped object, or None if there is none."""
        grasped_objs: list[Object] = []
        for obj in self.get_objects(Geom3DCuboidType):
            if self.get(obj, "grasp_active") > 0.5:
                grasped_objs.append(obj)
        if not grasped_objs:
            return None
        assert len(grasped_objs) == 1, "Multiple objects should not be grasped"
        grasped_obj = grasped_objs[0]
        return grasped_obj.name

    @property
    def grasped_object_transform(self) -> Pose | None:
        """The grasped object transform, or None if there is no grasped object."""
        if self.grasped_object is None:
            return None
        robot = self.robot
        x = self.get(robot, "grasp_tf_x")
        y = self.get(robot, "grasp_tf_y")
        z = self.get(robot, "grasp_tf_z")
        qx = self.get(robot, "grasp_tf_qx")
        qy = self.get(robot, "grasp_tf_qy")
        qz = self.get(robot, "grasp_tf_qz")
        qw = self.get(robot, "grasp_tf_qw")
        grasp_tf = Pose((x, y, z), (qx, qy, qz, qw))
        return grasp_tf


class Geom3DRobotActionSpace(RobotActionSpace):
    """An action space for a 7 DOF robot that can open and close its gripper.

    Actions are bounded relative joint positions and open / close.

    The open / close logic is: <-0.5 is close, >0.5 is open, and otherwise no change.
    """

    def __init__(
        self,
        max_magnitude: float = 0.05,
    ) -> None:
        low = np.array([-max_magnitude] * 7 + [-1])
        high = np.array([max_magnitude] * 7 + [-1])
        super().__init__(low, high)

    def create_markdown_description(self) -> str:
        """Create a human-readable markdown description of this space."""
        return """An action space for a 7 DOF robot that can open and close its gripper.

    Actions are bounded relative joint positions and open / close.

    The open / close logic is: <-0.5 is close, >0.5 is open, and otherwise no change.
"""


def extend_joints_to_include_fingers(joint_positions: JointPositions) -> JointPositions:
    """Add 6 DOF for fingers."""
    assert len(joint_positions) == 7
    finger_joints = [0.0] * 6
    return list(joint_positions) + finger_joints


def remove_fingers_from_extended_joints(
    joint_positions: JointPositions,
) -> JointPositions:
    """Inverse of _extend_joints_to_include_fingers()."""
    assert len(joint_positions) == 13
    return joint_positions[:7]
