"""Parameterized skills for the TidyBot3D ground environment."""

from typing import Any

import numpy as np
import pybullet as p
from bilevel_planning.structs import (
    GroundParameterizedController,
    LiftedParameterizedController,
)
from prbench.envs.dynamic3d.object_types import (
    MujocoFixtureObjectType,
    MujocoMovableObjectType,
    MujocoObjectType,
    MujocoTidyBotRobotObjectType,
)
from prbench.envs.dynamic3d.robots.tidybot_robot_env import (
    TidyBot3DRobotActionSpace,
)
from prpl_utils.utils import get_signed_angle_distance
from pybullet_helpers.geometry import Pose, multiply_poses, set_pose
from pybullet_helpers.inverse_kinematics import inverse_kinematics
from pybullet_helpers.joint import JointPositions, get_jointwise_difference
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    run_motion_planning,
)
from pybullet_helpers.robots import SingleArmPyBulletRobot, create_pybullet_robot
from pybullet_helpers.utils import (
    create_pybullet_block,
    create_pybullet_shelf,
)
from relational_structs import (
    Array,
    ObjectCentricState,
    Variable,
)
from spatialmath import SE2

from prbench_models.dynamic3d.utils import (
    get_overhead_object_se2_pose,
    run_base_motion_planning,
)

# Constants.
MAX_BASE_MOVEMENT_MAGNITUDE = 1e-1
GRIPPER_OPEN_THRESHOLD = 0.01
GRASP_CLOSE_THRESHOLD = 0.6  # for stable grasp
GRIPPER_CLOSED_THRESHOLD = 0.02
WAYPOINT_TOL = 1e-2
MOVE_TO_TARGET_DISTANCE_BOUNDS = (0.3, 0.6)
MOVE_TO_TARGET_ROT_BOUNDS = (-np.pi, np.pi)
WORLD_X_BOUNDS = (-2.5, 2.5)  # we should move these later
WORLD_Y_BOUNDS = (-2.5, 2.5)  # we should move these later


# Utility functions.
def get_target_robot_pose_from_parameters(
    target_object_pose: SE2, target_distance: float, target_rot: float
) -> SE2:
    """Determine the pose for the robot given the state and parameters.

    The robot will be facing the target_object_pose position while being target_distance
    away, and rotated w.r.t. the target_object_pose rotation by target_rot.
    """
    # Absolute angle of the line from the robot to the target.
    ang = target_object_pose.theta() + target_rot

    # Place the robot `target_distance` away from the target along -ang
    tx, ty = target_object_pose.t  # target translation (x, y).
    rx = tx - target_distance * np.cos(ang)
    ry = ty - target_distance * np.sin(ang)

    # Robot faces the target: heading points along +ang (toward the target).
    return SE2(rx, ry, ang)


class MoveToTargetGroundController(
    GroundParameterizedController[ObjectCentricState, Array]
):
    """Controller for motion planning to reach a target.

    The object parameters are:
        robot: The robot itself.
        object: The target object.

    The continuous parameters are:
        target_distance: float
        target_rot: float (radians)

    The controller uses motion planning to move the robot base to reach the target. The
    target base pose is computed as follows: starting with the target object pose, get
    the target _robot_ pose by applying the target distance and target rot from the
    continuous parameters. Note that the robot will always be facing directly towards
    the target object.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._last_state: ObjectCentricState | None = None
        self._current_params: np.ndarray | None = None
        self._current_base_motion_plan: list[SE2] | None = None

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator, rotate: bool = False
    ) -> Any:
        if rotate:
            if self.objects[2].name == "cube1":
                distance = 0.85
            elif self.objects[2].name == "cube2":
                distance = 0.92
            else:
                raise ValueError(f"Unknown target object: {self.objects[2].name}")
            rot = -np.pi / 2
        else:
            distance = 0.5  # for stable grasp
            rot = 0.0
            # distance = rng.uniform(*MOVE_TO_TARGET_DISTANCE_BOUNDS)
            # rot = rng.uniform(*MOVE_TO_TARGET_ROT_BOUNDS)
        return np.array([distance, rot])

    def reset(
        self,
        x: ObjectCentricState,
        params: Any,
        extend_xy_magnitude: float = 0.025,
        extend_rot_magnitude: float = np.pi / 8,
        disable_collision_objects: list[str] | None = None,
    ) -> None:
        self._last_state = x
        assert isinstance(params, np.ndarray)
        self._current_params = params.copy()
        # Derive the target pose for the robot.
        target_distance, target_rot = self._current_params
        target_object = self.objects[1]
        target_object_pose = get_overhead_object_se2_pose(x, target_object)
        target_base_pose = get_target_robot_pose_from_parameters(
            target_object_pose, target_distance, target_rot
        )
        # Run motion planning.
        base_motion_plan = run_base_motion_planning(
            state=x,
            target_base_pose=target_base_pose,
            x_bounds=WORLD_X_BOUNDS,
            y_bounds=WORLD_Y_BOUNDS,
            seed=0,  # use a constant seed to effectively make this "deterministic"
            extend_xy_magnitude=extend_xy_magnitude,
            extend_rot_magnitude=extend_rot_magnitude,
            disable_collision_objects=disable_collision_objects,
        )
        assert base_motion_plan is not None
        self._current_base_motion_plan = base_motion_plan

    def terminated(self) -> bool:
        assert self._current_base_motion_plan is not None
        return self._robot_is_close_to_pose(self._current_base_motion_plan[-1])

    def step(self) -> Array:
        assert self._current_base_motion_plan is not None
        while len(self._current_base_motion_plan) > 1:
            peek_pose = self._current_base_motion_plan[0]
            # Close enough, pop and continue.
            if self._robot_is_close_to_pose(peek_pose):
                self._current_base_motion_plan.pop(0)
            # Not close enough, stop popping.
            break
        robot_pose = self._get_current_robot_pose()
        next_pose = self._current_base_motion_plan[0]
        dx = next_pose.x - robot_pose.x
        dy = next_pose.y - robot_pose.y
        drot = get_signed_angle_distance(next_pose.theta(), robot_pose.theta())
        action = np.zeros(11, dtype=np.float32)
        action[0] = dx
        action[1] = dy
        action[2] = drot
        action[-1] = self._get_current_robot_gripper_pose()
        return action

    def observe(self, x: ObjectCentricState) -> None:
        self._last_state = x

    def _get_current_robot_pose(self) -> SE2:
        assert self._last_state is not None
        state = self._last_state
        robot = self.objects[0]
        return SE2(
            state.get(robot, "pos_base_x"),
            state.get(robot, "pos_base_y"),
            state.get(robot, "pos_base_rot"),
        )

    def _get_current_robot_gripper_pose(self) -> float:
        x = self._last_state
        assert x is not None
        robot_obj = x.get_object_from_name("robot")
        if x.get(robot_obj, "pos_gripper") > 0.2:
            return GRASP_CLOSE_THRESHOLD
        return 0.0

    def _robot_is_close_to_pose(self, pose: SE2, atol: float = WAYPOINT_TOL) -> bool:
        robot_pose = self._get_current_robot_pose()
        return bool(
            np.isclose(robot_pose.x, pose.x, atol=atol)
            and np.isclose(robot_pose.y, pose.y, atol=atol)
            and np.isclose(
                get_signed_angle_distance(robot_pose.theta(), pose.theta()),
                0.0,
                atol=atol,
            )
        )


class PyBulletSim:
    """An interface to PyBullet.

    We should generalize and move this out later.
    """

    def __init__(self, initial_state: ObjectCentricState) -> None:
        """NOTE: for now, this is extremely specific to the Ground environment where
        there is exactly one cube. We will generalize this later."""

        # Hardcode the transform from the base pose to the arm pose.
        # check if this is correct......
        self._base_to_arm_pose = Pose((0.12, 0.0, 0.4))

        # Create the PyBullet simulator.
        # Uncomment for debugging.
        # from pybullet_helpers.gui import create_gui_connection
        # self._physics_client_id = create_gui_connection(
        #     camera_pitch=-90, background_rgb=(1.0, 1.0, 1.0)
        # )  # pylint: disable=line-too-long
        self._physics_client_id = p.connect(p.DIRECT)

        # Create the robot, assuming that it is a kinova gen3.
        self._robot = create_pybullet_robot(
            "kinova-gen3", self._physics_client_id, fixed_base=False
        )

        # Create the cube.
        cube1_obj = initial_state.get_object_from_name("cube1")
        cube_half_extents = (
            initial_state.get(cube1_obj, "bb_x") / 2,
            initial_state.get(cube1_obj, "bb_y") / 2,
            initial_state.get(cube1_obj, "bb_z") / 2,
        )
        self._cube1 = create_pybullet_block(
            color=(1.0, 0.0, 0.0, 1.0),  # doesn't matter,
            half_extents=cube_half_extents,
            physics_client_id=self._physics_client_id,
        )

        if "cupboard_1" in initial_state.get_object_names():
            self._cupboard1_shelf_id, self._cupboard1_surface_ids = (
                create_pybullet_shelf(
                    color=(0.5, 0.5, 0.5, 1.0),
                    shelf_width=0.60198,
                    shelf_depth=0.254,
                    shelf_height=0.0127,
                    spacing=0.254,
                    support_width=0.0127,
                    num_layers=4,
                    physics_client_id=self._physics_client_id,
                )
            )

        # Used for checking if two confs are close.
        self._joint_distance_fn = create_joint_distance_fn(self._robot)

    @property
    def physics_client_id(self) -> int:
        """The physics client ID."""
        return self._physics_client_id

    @property
    def robot(self) -> SingleArmPyBulletRobot:
        """The robot pybullet."""
        return self._robot

    def get_robot_joints(self) -> JointPositions:
        """Get the current robot joints from the simulator."""
        return self._robot.get_joint_positions()

    def set_state(self, x: ObjectCentricState) -> None:
        """Update the internal state of the simulator from an object-centric state."""
        # Update the robot state.
        robot_obj = x.get_object_from_name("robot")
        # Update the arm base.
        base_pose = Pose.from_rpy(
            (x.get(robot_obj, "pos_base_x"), x.get(robot_obj, "pos_base_y"), 0.0),
            (0, 0, x.get(robot_obj, "pos_base_rot")),
        )
        arm_pose = multiply_poses(base_pose, self._base_to_arm_pose)
        self._robot.set_base(arm_pose)
        # Update the arm conf.
        arm_conf = [
            x.get(robot_obj, "pos_arm_joint1"),
            x.get(robot_obj, "pos_arm_joint2"),
            x.get(robot_obj, "pos_arm_joint3"),
            x.get(robot_obj, "pos_arm_joint4"),
            x.get(robot_obj, "pos_arm_joint5"),
            x.get(robot_obj, "pos_arm_joint6"),
            x.get(robot_obj, "pos_arm_joint7"),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        self._robot.set_joints(arm_conf)

        # Update the cube state.
        cube1_obj = x.get_object_from_name("cube1")
        cube_pose = Pose(
            (x.get(cube1_obj, "x"), x.get(cube1_obj, "y"), x.get(cube1_obj, "z")),
            (
                x.get(cube1_obj, "qx"),
                x.get(cube1_obj, "qy"),
                x.get(cube1_obj, "qz"),
                x.get(cube1_obj, "qw"),
            ),
        )
        set_pose(self._cube1, cube_pose, self._physics_client_id)

        if "cupboard_1" in x.get_object_names():
            cupboard1_obj = x.get_object_from_name("cupboard_1")
            cupboard1_shelf_pose = Pose(
                (
                    x.get(cupboard1_obj, "x"),
                    x.get(cupboard1_obj, "y"),
                    x.get(cupboard1_obj, "z"),
                ),
                (
                    x.get(cupboard1_obj, "qx"),
                    x.get(cupboard1_obj, "qy"),
                    x.get(cupboard1_obj, "qz"),
                    x.get(cupboard1_obj, "qw"),
                ),
            )
            set_pose(
                self._cupboard1_shelf_id, cupboard1_shelf_pose, self._physics_client_id
            )

    def get_ee_pose(self) -> Pose:
        """Get the end effector pose."""
        return self._robot.get_end_effector_pose()

    def get_collision_bodies(self) -> set[int]:
        """Get pybullet IDs for collision bodies."""
        return {self._cube1}

    def get_joint_distance(self, conf1: JointPositions, conf2: JointPositions) -> float:
        """Get the distance between two arm confs."""
        return self._joint_distance_fn(conf1, conf2)


class MoveArmToConfController(GroundParameterizedController[ObjectCentricState, Array]):
    """Controller for motion planning the arm to reach a target conf.

    The object parameters are:
        robot: The robot itself.

    The continuous parameters are:
        joint1_target: float
        joint2_target: float
        ...
        joint7_target: float

    The controller uses motion planning in pybullet.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._last_state: ObjectCentricState | None = None
        self._current_params: np.ndarray | None = None
        self._current_arm_joint_plan: list[JointPositions] | None = None
        self._pybullet_sim: PyBulletSim | None = None

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        # We can later implement sampling if it's helpful, but usually the user would
        # want to specify the target arm conf themselves.
        raise NotImplementedError

    def reset(self, x: ObjectCentricState, params: Any) -> None:
        # Initialize the PyBullet interface if this is the first time ever.
        if self._pybullet_sim is None:
            self._pybullet_sim = PyBulletSim(x)
        # Update the current state and parameters.
        self._last_state = x
        assert isinstance(params, np.ndarray)
        self._current_params = params.copy()
        target_joints = self._current_params.tolist() + ([0.0] * 6)
        # Reset PyBullet given the current state.
        self._pybullet_sim.set_state(x)
        # Run motion planning.
        plan = run_motion_planning(
            self._pybullet_sim.robot,
            self._pybullet_sim.get_robot_joints(),
            target_joints,
            collision_bodies=self._pybullet_sim.get_collision_bodies(),
            seed=0,  # use a constant seed to make this effectively deterministic
            physics_client_id=self._pybullet_sim.physics_client_id,
        )
        assert plan is not None, "Motion planning failed"
        self._current_arm_joint_plan = plan

    def terminated(self) -> bool:
        assert self._current_arm_joint_plan is not None
        return self._robot_is_close_to_conf(self._current_arm_joint_plan[-1])

    def step(self) -> Array:
        assert self._current_arm_joint_plan is not None
        while len(self._current_arm_joint_plan) > 1:
            peek_conf = self._current_arm_joint_plan[0]
            # Close enough, pop and continue.
            if self._robot_is_close_to_conf(peek_conf):
                self._current_arm_joint_plan.pop(0)
            # Not close enough, stop popping.
            break
        robot_conf = self._get_current_robot_arm_conf()
        gripper_pose = self._get_current_robot_gripper_pose()
        next_conf = self._current_arm_joint_plan[0]
        action = np.zeros(11, dtype=np.float32)
        action[3:10] = np.subtract(next_conf, robot_conf)[:7]
        action[-1] = gripper_pose
        return action

    def observe(self, x: ObjectCentricState) -> None:
        self._last_state = x

    def _get_current_robot_arm_conf(self) -> JointPositions:
        x = self._last_state
        assert x is not None
        robot_obj = x.get_object_from_name("robot")
        return [
            x.get(robot_obj, "pos_arm_joint1"),
            x.get(robot_obj, "pos_arm_joint2"),
            x.get(robot_obj, "pos_arm_joint3"),
            x.get(robot_obj, "pos_arm_joint4"),
            x.get(robot_obj, "pos_arm_joint5"),
            x.get(robot_obj, "pos_arm_joint6"),
            x.get(robot_obj, "pos_arm_joint7"),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

    def _get_current_robot_gripper_pose(self) -> float:
        x = self._last_state
        assert x is not None
        robot_obj = x.get_object_from_name("robot")
        if x.get(robot_obj, "pos_gripper") > 0.2:
            return GRASP_CLOSE_THRESHOLD
        return 0.0

    def _robot_is_close_to_conf(self, conf: JointPositions) -> bool:
        current_conf = self._get_current_robot_arm_conf()
        assert self._pybullet_sim is not None
        dist = self._pybullet_sim.get_joint_distance(current_conf, conf)
        return dist < 3 * 1e-2


class MoveArmToEndEffectorController(
    GroundParameterizedController[ObjectCentricState, Array]
):
    """Controller for motion planning the arm to reach a target end effector pose.

    The object parameters are:
        robot: The robot itself.

    The continuous parameters are:
        end_effector_pose: np.ndarray (x, y, z, rw, rx, ry, rz)

    The controller uses motion planning in pybullet.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._last_state: ObjectCentricState | None = None
        self._current_params: np.ndarray | None = None
        self._current_arm_joint_plan: list[JointPositions] | None = None
        self._pybullet_sim: PyBulletSim | None = None

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        # We can later implement sampling if it's helpful, but usually the user would
        # want to specify the target end effector pose themselves.
        raise NotImplementedError

    def reset(self, x: ObjectCentricState, params: Any) -> None:
        # Initialize the PyBullet interface if this is the first time ever.
        if self._pybullet_sim is None:
            self._pybullet_sim = PyBulletSim(x)
        # Update the current state and parameters.
        self._last_state = x
        assert isinstance(params, np.ndarray)
        self._current_params = params.copy()

        # Reset PyBullet given the current state.
        self._pybullet_sim.set_state(x)

        current_arm_base_pose = self._pybullet_sim.robot.get_base_pose()

        target_end_effector_pose_temp = multiply_poses(
            current_arm_base_pose,
            Pose(
                (
                    self._current_params[0],
                    self._current_params[1],
                    self._current_params[2],
                ),
                (
                    self._current_params[3],
                    self._current_params[4],
                    self._current_params[5],
                    self._current_params[6],
                ),
            ),
        )

        rotation = Pose.from_rpy((0, 0, 0), (0, 0, self._current_params[7]))
        target_end_effector_pose = multiply_poses(
            target_end_effector_pose_temp,
            rotation,
        )

        target_joints = inverse_kinematics(
            self._pybullet_sim.robot,
            target_end_effector_pose,
            set_joints=False,
        )

        # Run motion planning.
        plan = run_motion_planning(
            self._pybullet_sim.robot,
            self._pybullet_sim.get_robot_joints(),
            target_joints,
            collision_bodies=self._pybullet_sim.get_collision_bodies(),
            seed=0,  # use a constant seed to make this effectively deterministic
            physics_client_id=self._pybullet_sim.physics_client_id,
        )

        assert plan is not None, "Motion planning failed"
        self._current_arm_joint_plan = plan

    def terminated(self) -> bool:
        assert self._current_arm_joint_plan is not None
        return self._robot_is_close_to_conf(self._current_arm_joint_plan[-1])

    def step(self) -> Array:
        assert self._current_arm_joint_plan is not None
        while len(self._current_arm_joint_plan) > 1:
            peek_conf = self._current_arm_joint_plan[0]
            # Close enough, pop and continue.
            if self._robot_is_close_to_conf(peek_conf):
                self._current_arm_joint_plan.pop(0)
            # Not close enough, stop popping.
            break
        robot_conf = self._get_current_robot_arm_conf()
        gripper_pose = self._get_current_robot_gripper_pose()
        next_conf = self._current_arm_joint_plan[0]
        action = np.zeros(11, dtype=np.float32)
        action[3:10] = np.subtract(next_conf, robot_conf)[:7]
        action[-1] = gripper_pose
        return action

    def observe(self, x: ObjectCentricState) -> None:
        self._last_state = x

    def _get_current_robot_arm_conf(self) -> JointPositions:
        x = self._last_state
        assert x is not None
        robot_obj = x.get_object_from_name("robot")
        return [
            x.get(robot_obj, "pos_arm_joint1"),
            x.get(robot_obj, "pos_arm_joint2"),
            x.get(robot_obj, "pos_arm_joint3"),
            x.get(robot_obj, "pos_arm_joint4"),
            x.get(robot_obj, "pos_arm_joint5"),
            x.get(robot_obj, "pos_arm_joint6"),
            x.get(robot_obj, "pos_arm_joint7"),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

    def _get_current_robot_gripper_pose(self) -> float:
        x = self._last_state
        assert x is not None
        robot_obj = x.get_object_from_name("robot")
        if x.get(robot_obj, "pos_gripper") > 0.2:
            return GRASP_CLOSE_THRESHOLD
        return 0.0

    def _robot_is_close_to_conf(self, conf: JointPositions) -> bool:
        current_conf = self._get_current_robot_arm_conf()
        assert self._pybullet_sim is not None
        dist = self._pybullet_sim.get_joint_distance(current_conf, conf)
        return dist < 3 * 1e-2


class CloseGripperController(GroundParameterizedController[ObjectCentricState, Array]):
    """Controller for closing the gripper.

    The object parameters are:
        robot: The robot itself.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._last_state: ObjectCentricState | None = None
        self.last_gripper_state: float = 0.0

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        # We can later implement sampling if it's helpful, but usually the user would
        # want to specify the target end effector pose themselves.
        raise NotImplementedError

    def reset(self, x: ObjectCentricState, params: Any | None = None) -> None:
        # Update the current state and parameters.
        self._last_state = x

    def terminated(self) -> bool:
        return self._robot_gripper_is_closed(atol=0.02)

    def step(self) -> Array:
        self.last_gripper_state = self._get_current_gripper_pose()
        action = np.zeros(11, dtype=np.float32)
        action[-1] = 1
        return action

    def observe(self, x: ObjectCentricState) -> None:
        self._last_state = x

    def _get_current_gripper_pose(self) -> SE2:
        assert self._last_state is not None
        state = self._last_state
        robot = self.objects[0]
        return state.get(robot, "pos_gripper")

    def _robot_gripper_is_closed(self, atol: float = GRIPPER_CLOSED_THRESHOLD) -> bool:
        current_gripper_pose = self._get_current_gripper_pose()
        return current_gripper_pose > 0.2 and np.isclose(
            current_gripper_pose, self.last_gripper_state, atol=atol
        )


class OpenGripperController(GroundParameterizedController[ObjectCentricState, Array]):
    """Controller for opening the gripper.

    The object parameters are:
        robot: The robot itself.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._last_state: ObjectCentricState | None = None
        self.last_gripper_state: float = 0.0

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        # We can later implement sampling if it's helpful, but usually the user would
        # want to specify the target end effector pose themselves.
        raise NotImplementedError

    def reset(self, x: ObjectCentricState, params: Any | None = None) -> None:
        # Update the current state and parameters.
        self._last_state = x

    def terminated(self) -> bool:
        return self._robot_gripper_is_open()

    def step(self) -> Array:
        self.last_gripper_state = self._get_current_gripper_pose()
        action = np.zeros(11, dtype=np.float32)
        action[-1] = 0
        return action

    def observe(self, x: ObjectCentricState) -> None:
        self._last_state = x

    def _get_current_gripper_pose(self) -> SE2:
        assert self._last_state is not None
        state = self._last_state
        robot = self.objects[0]
        return state.get(robot, "pos_gripper")

    def _robot_gripper_is_open(self, atol: float = GRIPPER_OPEN_THRESHOLD) -> bool:
        current_gripper_pose = self._get_current_gripper_pose()
        return current_gripper_pose < atol


class PickGroundController(GroundParameterizedController[ObjectCentricState, Array]):
    """Controller for motion planning to pick up a target.

    The object parameters are:
        robot: The robot itself.
        object: The target object.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._last_state: ObjectCentricState | None = None
        self._current_params: np.ndarray | None = None
        self._current_arm_joint_plan: list[JointPositions] | None = None
        self._current_retract_plan: list[JointPositions] | None = None
        self._pybullet_sim: PyBulletSim | None = None
        self._pre_grasp: bool = False
        self._closed_gripper: bool = False
        self._lifted: bool = False
        self._last_gripper_state: float = 0.0
        self.home_joints = np.deg2rad(
            [0, -20, 180, -146, 0, -50, 90, 0, 0, 0, 0, 0, 0]
        )  # retract configuration

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        # We can later implement sampling if it's helpful, but usually the user would
        # want to specify the target end effector pose themselves.
        return None

    def reset(self, x: ObjectCentricState, params: Any | None = None) -> None:
        # Initialize the PyBullet interface if this is the first time ever.
        if self._pybullet_sim is None:
            self._pybullet_sim = PyBulletSim(x)
        # Update the current state and parameters.
        self._last_state = x

        # Reset PyBullet given the current state.
        self._pybullet_sim.set_state(x)

        target_object = self.objects[1]

        target_grap_pose_world = Pose(
            (
                x.get(target_object, "x"),
                x.get(target_object, "y"),
                x.get(target_object, "z"),
            ),
            (
                x.get(target_object, "qx"),
                x.get(target_object, "qy"),
                x.get(target_object, "qz"),
                x.get(target_object, "qw"),
            ),
        )

        target_end_effector_pose = multiply_poses(
            target_grap_pose_world,
            Pose(
                (0.005, 0, 0.035),  # offsets in end-effector local frame
                (0.707, 0.707, 0, 0),  # orientation
            ),
        )

        target_joints = inverse_kinematics(
            self._pybullet_sim.robot,
            target_end_effector_pose,
            set_joints=False,
        )

        # Run motion planning.
        plan = run_motion_planning(
            self._pybullet_sim.robot,
            self._pybullet_sim.get_robot_joints(),
            target_joints,
            collision_bodies=self._pybullet_sim.get_collision_bodies(),
            seed=0,  # use a constant seed to make this effectively deterministic
            physics_client_id=self._pybullet_sim.physics_client_id,
        )

        retract_plan = run_motion_planning(
            self._pybullet_sim.robot,
            target_joints,
            self.home_joints.tolist(),
            collision_bodies={},
            seed=0,  # use a constant seed to make this effectively deterministic
            physics_client_id=self._pybullet_sim.physics_client_id,
        )

        assert plan is not None, "Motion planning failed"
        assert retract_plan is not None, "Motion planning failed"
        self._current_arm_joint_plan = plan
        self._current_retract_plan = retract_plan

    def terminated(self) -> bool:
        assert (
            self._current_arm_joint_plan is not None
            and self._current_retract_plan is not None
        )
        return self._lifted

    def step(self) -> Array:
        assert self._current_arm_joint_plan is not None
        # first substep
        if not self._pre_grasp and not self._closed_gripper:
            while len(self._current_arm_joint_plan) > 1:
                peek_conf = self._current_arm_joint_plan[0]
                # Close enough, pop and continue.
                if self._robot_is_close_to_conf(peek_conf):
                    self._current_arm_joint_plan.pop(0)
                # Not close enough, stop popping.
                break
            if self._robot_is_close_to_conf(self._current_arm_joint_plan[-1]):
                self._pre_grasp = True
            robot_conf = self._get_current_robot_arm_conf()
            gripper_pose = self._get_current_robot_gripper_pose()
            next_conf = self._current_arm_joint_plan[0]
            action = np.zeros(11, dtype=np.float32)
            joint_infos = self._pybullet_sim.robot.joint_infos  # type: ignore
            free_joints_infos = [
                joint_info for joint_info in joint_infos if joint_info.qIndex > -1
            ]
            action[3:10] = get_jointwise_difference(
                free_joints_infos[:7], next_conf[:7], robot_conf[:7]
            )
            action[-1] = gripper_pose
            return action
        if self._pre_grasp and not self._closed_gripper:
            if self._get_current_robot_gripper_pose() > 0.2 and np.isclose(
                self._get_current_robot_gripper_pose(),
                self._last_gripper_state,
                atol=0.02,
            ):
                self._closed_gripper = True
            action = np.zeros(11, dtype=np.float32)
            action[-1] = 1
            self._last_gripper_state = self._get_current_robot_gripper_pose()
            return action
        if self._pre_grasp and self._closed_gripper:
            while len(self._current_retract_plan) > 1:  # type: ignore
                peek_conf = self._current_retract_plan[0]  # type: ignore
                # Close enough, pop and continue.
                if self._robot_is_close_to_conf(peek_conf):
                    self._current_retract_plan.pop(0)  # type: ignore
                # Not close enough, stop popping.
                break
            if self._robot_is_close_to_conf(self._current_retract_plan[-1]):  # type: ignore # pylint: disable=line-too-long
                self._lifted = True
            robot_conf = self._get_current_robot_arm_conf()
            gripper_pose = self._get_current_robot_gripper_pose()
            next_conf = self._current_retract_plan[0]  # type: ignore
            action = np.zeros(11, dtype=np.float32)
            joint_infos = self._pybullet_sim.robot.joint_infos  # type: ignore
            free_joints_infos = [
                joint_info for joint_info in joint_infos if joint_info.qIndex > -1
            ]
            action[3:10] = get_jointwise_difference(
                free_joints_infos[:7], next_conf[:7], robot_conf[:7]
            )
            action[-1] = gripper_pose
            return action
        raise ValueError("Invalid state")

    def observe(self, x: ObjectCentricState) -> None:
        self._last_state = x

    def _get_current_robot_arm_conf(self) -> JointPositions:
        x = self._last_state
        assert x is not None
        robot_obj = x.get_object_from_name("robot")
        return [
            x.get(robot_obj, "pos_arm_joint1"),
            x.get(robot_obj, "pos_arm_joint2"),
            x.get(robot_obj, "pos_arm_joint3"),
            x.get(robot_obj, "pos_arm_joint4"),
            x.get(robot_obj, "pos_arm_joint5"),
            x.get(robot_obj, "pos_arm_joint6"),
            x.get(robot_obj, "pos_arm_joint7"),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

    def _get_current_robot_gripper_pose(self) -> float:
        x = self._last_state
        assert x is not None
        robot_obj = x.get_object_from_name("robot")
        if x.get(robot_obj, "pos_gripper") > 0.2:
            return GRASP_CLOSE_THRESHOLD
        return 0.0

    def _robot_is_close_to_conf(self, conf: JointPositions) -> bool:
        current_conf = self._get_current_robot_arm_conf()
        assert self._pybullet_sim is not None
        dist = self._pybullet_sim.get_joint_distance(current_conf, conf)
        return dist < 3 * 1e-2


class PlaceGroundController(GroundParameterizedController[ObjectCentricState, Array]):
    """Controller for motion planning to place a target.

    The object parameters are:
        robot: The robot itself.
        object: The target object.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._last_state: ObjectCentricState | None = None
        self._current_params: np.ndarray | None = None
        self._current_arm_joint_plan: list[JointPositions] | None = None
        self._current_retract_plan: list[JointPositions] | None = None
        self._pybullet_sim: PyBulletSim | None = None
        self._pre_place: bool = False
        self._open_gripper: bool = False
        self._returned: bool = False
        self._last_gripper_state: float = 0.0
        self.home_joints = np.deg2rad(
            [0, -20, 180, -146, 0, -50, 90, 0, 0, 0, 0, 0, 0]
        )  # retract configuration

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        # We can later implement sampling if it's helpful, but usually the user would
        # want to specify the target end effector pose themselves.
        pass

    def reset(self, x: ObjectCentricState, params: Any | None = None) -> None:  # type: ignore # pylint: disable=arguments-differ
        # Initialize the PyBullet interface if this is the first time ever.
        if self._pybullet_sim is None:
            self._pybullet_sim = PyBulletSim(x)
        # Update the current state and parameters.
        self._last_state = x

        # Reset PyBullet given the current state.
        self._pybullet_sim.set_state(x)

        current_arm_base_pose = self._pybullet_sim.robot.get_base_pose()

        target_end_effector_pose = Pose((0.7, 0.0, 0.03), (0.5, 0.5, 0.5, 0.5))

        target_end_effector_pose = multiply_poses(
            current_arm_base_pose, target_end_effector_pose
        )

        target_joints = inverse_kinematics(
            self._pybullet_sim.robot,
            target_end_effector_pose,
            set_joints=False,
        )

        # Run motion planning.
        plan = run_motion_planning(
            self._pybullet_sim.robot,
            self._pybullet_sim.get_robot_joints(),
            target_joints,
            collision_bodies={},
            seed=0,  # use a constant seed to make this effectively deterministic
            physics_client_id=self._pybullet_sim.physics_client_id,
        )

        retract_plan = run_motion_planning(
            self._pybullet_sim.robot,
            target_joints,
            self.home_joints.tolist(),
            collision_bodies=self._pybullet_sim.get_collision_bodies(),
            seed=0,  # use a constant seed to make this effectively deterministic
            physics_client_id=self._pybullet_sim.physics_client_id,
        )

        assert plan is not None, "Motion planning failed"
        assert retract_plan is not None, "Motion planning failed"
        self._current_arm_joint_plan = plan
        self._current_retract_plan = retract_plan

    def terminated(self) -> bool:
        assert (
            self._current_arm_joint_plan is not None
            and self._current_retract_plan is not None
        )
        return self._returned

    def step(self) -> Array:
        assert self._current_arm_joint_plan is not None
        # first substep
        if not self._pre_place and not self._open_gripper:
            while len(self._current_arm_joint_plan) > 1:
                peek_conf = self._current_arm_joint_plan[0]
                # Close enough, pop and continue.
                if self._robot_is_close_to_conf(peek_conf):
                    self._current_arm_joint_plan.pop(0)
                # Not close enough, stop popping.
                break
            if self._robot_is_close_to_conf(self._current_arm_joint_plan[-1]):
                self._pre_place = True
            robot_conf = self._get_current_robot_arm_conf()
            gripper_pose = self._get_current_robot_gripper_pose()
            next_conf = self._current_arm_joint_plan[0]
            action = np.zeros(11, dtype=np.float32)
            joint_infos = self._pybullet_sim.robot.joint_infos  # type: ignore
            free_joints_infos = [
                joint_info for joint_info in joint_infos if joint_info.qIndex > -1
            ]
            action[3:10] = get_jointwise_difference(
                free_joints_infos[:7], next_conf[:7], robot_conf[:7]
            )
            action[-1] = gripper_pose
            return action
        if self._pre_place and not self._open_gripper:
            if self._get_current_robot_gripper_pose() < GRIPPER_OPEN_THRESHOLD:
                self._open_gripper = True
            action = np.zeros(11, dtype=np.float32)
            action[-1] = 0
            self._last_gripper_state = self._get_current_robot_gripper_pose()
            return action
        if self._pre_place and self._open_gripper:
            while len(self._current_retract_plan) > 1:  # type: ignore
                peek_conf = self._current_retract_plan[0]  # type: ignore
                # Close enough, pop and continue.
                if self._robot_is_close_to_conf(peek_conf):
                    self._current_retract_plan.pop(0)  # type: ignore
                # Not close enough, stop popping.
                break
            if self._robot_is_close_to_conf(self._current_retract_plan[-1]):  # type: ignore # pylint: disable=line-too-long
                self._returned = True
            robot_conf = self._get_current_robot_arm_conf()
            gripper_pose = self._get_current_robot_gripper_pose()
            next_conf = self._current_retract_plan[0]  # type: ignore
            action = np.zeros(11, dtype=np.float32)
            joint_infos = self._pybullet_sim.robot.joint_infos  # type: ignore
            free_joints_infos = [
                joint_info for joint_info in joint_infos if joint_info.qIndex > -1
            ]
            action[3:10] = get_jointwise_difference(
                free_joints_infos[:7], next_conf[:7], robot_conf[:7]
            )
            action[-1] = gripper_pose
            return action
        raise ValueError("Invalid state")

    def observe(self, x: ObjectCentricState) -> None:
        self._last_state = x

    def _get_current_robot_arm_conf(self) -> JointPositions:
        x = self._last_state
        assert x is not None
        robot_obj = x.get_object_from_name("robot")
        return [
            x.get(robot_obj, "pos_arm_joint1"),
            x.get(robot_obj, "pos_arm_joint2"),
            x.get(robot_obj, "pos_arm_joint3"),
            x.get(robot_obj, "pos_arm_joint4"),
            x.get(robot_obj, "pos_arm_joint5"),
            x.get(robot_obj, "pos_arm_joint6"),
            x.get(robot_obj, "pos_arm_joint7"),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

    def _get_current_robot_gripper_pose(self) -> float:
        x = self._last_state
        assert x is not None
        robot_obj = x.get_object_from_name("robot")
        if x.get(robot_obj, "pos_gripper") > 0.2:
            return GRASP_CLOSE_THRESHOLD
        return 0.0

    def _robot_is_close_to_conf(self, conf: JointPositions) -> bool:
        current_conf = self._get_current_robot_arm_conf()
        assert self._pybullet_sim is not None
        dist = self._pybullet_sim.get_joint_distance(current_conf, conf)
        return dist < 4 * 1e-2


def create_lifted_controllers(
    action_space: TidyBot3DRobotActionSpace,
    init_constant_state: ObjectCentricState | None = None,
) -> dict[str, LiftedParameterizedController]:
    """Create lifted parameterized controllers for the TidyBot3D ground environment."""

    del action_space, init_constant_state  # not used

    # Controllers.

    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    target = Variable("?target", MujocoObjectType)

    LiftedMoveToTargetController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, target],
            MoveToTargetGroundController,
        )
    )

    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    target = Variable("?target", MujocoObjectType)
    prev_target = Variable("?prev_target", MujocoObjectType)

    LiftedMoveToTargetFromOtherTargetController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, target, prev_target],
            MoveToTargetGroundController,
        )
    )

    # Move arm to conf controller.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)

    LiftedMoveArmToConfController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot],
            MoveArmToConfController,
        )
    )

    # Move arm to end effector controller.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)

    LiftedMoveArmToEndEffectorController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot],
            MoveArmToEndEffectorController,
        )
    )

    # Close gripper controller.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)

    LiftedCloseGripperController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot],
            CloseGripperController,
        )
    )

    # Open gripper controller.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)

    LiftedOpenGripperController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot],
            OpenGripperController,
        )
    )

    # Pick ground controller.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    target = Variable("?target", MujocoMovableObjectType)

    LiftedPickGroundController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, target],
            PickGroundController,
        )
    )

    # Place controller.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    target = Variable("?target", MujocoMovableObjectType)
    target_place = Variable("?target_place", MujocoFixtureObjectType)

    LiftedPlaceGroundController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, target, target_place],
            PlaceGroundController,
        )
    )

    return {
        "move_to_target": LiftedMoveToTargetController,
        "move_to_target_from_other_target": LiftedMoveToTargetFromOtherTargetController,
        "move_arm_to_conf": LiftedMoveArmToConfController,
        "move_arm_to_end_effector": LiftedMoveArmToEndEffectorController,
        "close_gripper": LiftedCloseGripperController,
        "open_gripper": LiftedOpenGripperController,
        "pick_ground": LiftedPickGroundController,
        "place_ground": LiftedPlaceGroundController,
    }
