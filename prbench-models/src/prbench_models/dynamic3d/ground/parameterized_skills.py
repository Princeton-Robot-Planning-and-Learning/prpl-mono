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
from prbench.envs.geom3d.utils import extend_joints_to_include_fingers
from prpl_utils.utils import get_signed_angle_distance
from pybullet_helpers.geometry import Pose, multiply_poses, set_pose
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.inverse_kinematics import (
    inverse_kinematics,
    set_robot_joints_with_held_object,
)
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
    Object,
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
GRASP_CLOSE_THRESHOLD = 1.0  # for stable grasp
GRIPPER_CLOSED_THRESHOLD = 0.02
WAYPOINT_TOL = 1e-2
MOVE_TO_TARGET_DISTANCE_BOUNDS = (0.45, 0.6)
MOVE_TO_TARGET_ROT_BOUNDS = (-np.pi, np.pi)
WORLD_X_BOUNDS = (-2.5, 2.5)  # we should move these later
WORLD_Y_BOUNDS = (-2.5, 2.5)  # we should move these later
ROBOT_ARM_POSE_TO_BASE = Pose((0.12, 0.0, 0.4))
GRASP_TRANSFORM_TO_OBJECT = Pose((0.005, 0, 0.035), (0.707, 0.707, 0, 0))
BASE_DISTANCE_TO_CUPBOARD = 0.95
ARM_MOVEMENT_CUPBOARD = Pose((0.8, 0.0, 0.25), (0.5, 0.5, 0.5, 0.5))
PLACE_SAMPLER_COLLISION_THRESHOLD = 0.05
PLACE_SAMPLER_X_OFFSET_BOUNDS = (-0.10, 0)
PLACE_SAMPLER_Y_OFFSET_BOUNDS = (-0.15, 0.15)
MAX_SAMPLER_ATTEMPTS = 100
BASE_TO_CUPBOARD_ROTATION = -np.pi / 2


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

    def __init__(
        self, initial_state: ObjectCentricState, rendering: bool = False
    ) -> None:
        """NOTE: for now, this is extremely specific to the Ground environment where
        there is exactly one cube. We will generalize this later."""

        # Hardcode the transform from the base pose to the arm pose.
        self._base_to_arm_pose = ROBOT_ARM_POSE_TO_BASE

        # Create the PyBullet simulator.
        if rendering:
            self._physics_client_id = create_gui_connection(
                camera_pitch=-90, background_rgb=(1.0, 1.0, 1.0)
            )
        else:
            self._physics_client_id = p.connect(p.DIRECT)

        # Create the robot, assuming that it is a kinova gen3.
        self._robot = create_pybullet_robot(
            "kinova-gen3",
            self._physics_client_id,
            fixed_base=False,
            control_mode="reset",
        )

        self.base_link_to_held_obj: Pose | None = None

        # Create all the cubes.
        self._cubes: dict[str, int] = {}
        for cube_name in initial_state.get_object_names():
            if "cube" in cube_name:
                cube_obj = initial_state.get_object_from_name(cube_name)
                cube_half_extents = (
                    initial_state.get(cube_obj, "bb_x") / 2,
                    initial_state.get(cube_obj, "bb_y") / 2,
                    initial_state.get(cube_obj, "bb_z") / 2,
                )
                self._cubes[cube_name] = create_pybullet_block(
                    color=(1.0, 0.0, 0.0, 1.0),  # doesn't matter,
                    half_extents=cube_half_extents,
                    physics_client_id=self._physics_client_id,
                )

        self._cupboard1_shelf_id = None
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

    def set_state(
        self, x: ObjectCentricState, held_object: Object | None = None
    ) -> None:
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
        for cube_name in x.get_object_names():
            if "cube" in cube_name:
                cube_obj = x.get_object_from_name(cube_name)
                cube_pose = Pose(
                    (x.get(cube_obj, "x"), x.get(cube_obj, "y"), x.get(cube_obj, "z")),
                    (
                        x.get(cube_obj, "qx"),
                        x.get(cube_obj, "qy"),
                        x.get(cube_obj, "qz"),
                        x.get(cube_obj, "qw"),
                    ),
                )
                set_pose(self._cubes[cube_name], cube_pose, self._physics_client_id)

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
            assert self._cupboard1_shelf_id is not None
            set_pose(
                self._cupboard1_shelf_id,
                cupboard1_shelf_pose,
                self._physics_client_id,
            )

        if held_object:
            held_object_id = self._cubes[held_object.name]
            set_robot_joints_with_held_object(
                self._robot,
                self._physics_client_id,
                held_object_id,
                self.base_link_to_held_obj,
                extend_joints_to_include_fingers(arm_conf[:7]),
            )

    def get_ee_pose(self) -> Pose:
        """Get the end effector pose."""
        return self._robot.get_end_effector_pose()

    def get_collision_bodies(self, held_object: int | None = None) -> set[int]:
        """Get pybullet IDs for collision bodies."""
        collision_bodies: set[int] = set()
        collision_bodies.update(self._cubes.values())
        if self._cupboard1_shelf_id is not None:
            collision_bodies.add(self._cupboard1_shelf_id)
        if held_object is not None:
            collision_bodies.discard(held_object)
        return collision_bodies

    def get_joint_distance(self, conf1: JointPositions, conf2: JointPositions) -> float:
        """Get the distance between two arm confs."""
        return self._joint_distance_fn(conf1, conf2)

    def close(self) -> None:
        """Close the PyBullet simulator."""
        p.disconnect(self._physics_client_id)


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

    def __init__(
        self, *args, pybullet_sim: PyBulletSim | None = None, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._last_state: ObjectCentricState | None = None
        self._current_params: np.ndarray | None = None
        self._current_arm_joint_plan: list[JointPositions] | None = None
        self._current_retract_plan: list[JointPositions] | None = None
        self._current_base_motion_plan: list[SE2] | None = None
        self._pybullet_sim: PyBulletSim | None = pybullet_sim
        self._navigated: bool = False
        self._pre_grasp: bool = False
        self._closed_gripper: bool = False
        self._lifted: bool = False
        self._last_gripper_state: float = 0.0
        self.home_joints = np.deg2rad(
            [0, -20, 180, -146, 0, -50, 90, 0, 0, 0, 0, 0, 0]
        )  # retract configuration

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        target_object = self.objects[1]
        target_object_pose = get_overhead_object_se2_pose(x, target_object)

        for _ in range(MAX_SAMPLER_ATTEMPTS):
            distance = rng.uniform(*MOVE_TO_TARGET_DISTANCE_BOUNDS)  # type: ignore
            rot = rng.uniform(*MOVE_TO_TARGET_ROT_BOUNDS)
            target_base_pose = get_target_robot_pose_from_parameters(
                target_object_pose, distance, rot
            )
            collision = False
            for other_object in x.get_objects(MujocoMovableObjectType):
                if (
                    "cube" in other_object.name
                    and other_object.name != target_object.name
                ):
                    other_object_pose = get_overhead_object_se2_pose(x, other_object)
                    collision_distance = float(
                        np.linalg.norm(
                            [
                                target_base_pose.x - other_object_pose.x,
                                target_base_pose.y - other_object_pose.y,
                            ]
                        )
                    )
                    if collision_distance < 0.6:
                        collision = True
                        break
            if not collision:
                return np.array([distance, rot])

        raise ValueError("No valid parameters found")

    def reset(
        self,
        x: ObjectCentricState,
        params: Any,
        extend_xy_magnitude: float = 0.025,
        extend_rot_magnitude: float = np.pi / 8,
    ) -> None:
        # Initialize the PyBullet interface if this is the first time ever.
        if self._pybullet_sim is None:
            self._pybullet_sim = PyBulletSim(x)
        # Update the current state and parameters.
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
        )
        assert base_motion_plan is not None
        self._current_base_motion_plan = base_motion_plan

        plan_x = x.copy()
        robot = plan_x.get_object_from_name("robot")
        target_base_pose = self._current_base_motion_plan[-1]
        plan_x.set(robot, "pos_base_x", target_base_pose.x)
        plan_x.set(robot, "pos_base_y", target_base_pose.y)
        plan_x.set(robot, "pos_base_rot", target_base_pose.theta())

        # Reset PyBullet given the current state.
        self._pybullet_sim.set_state(plan_x)

        target_object = self.objects[1]

        target_grap_pose_world = Pose(
            (
                plan_x.get(target_object, "x"),
                plan_x.get(target_object, "y"),
                plan_x.get(target_object, "z"),
            ),
            (
                plan_x.get(target_object, "qx"),
                plan_x.get(target_object, "qy"),
                plan_x.get(target_object, "qz"),
                plan_x.get(target_object, "qw"),
            ),
        )

        target_end_effector_pose = multiply_poses(
            target_grap_pose_world,
            GRASP_TRANSFORM_TO_OBJECT,
        )

        self._pybullet_sim.base_link_to_held_obj = multiply_poses(
            target_end_effector_pose.invert(),
            target_grap_pose_world,
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
            collision_bodies=self._pybullet_sim.get_collision_bodies(  # pylint: disable=protected-access
                held_object=self._pybullet_sim._cubes[  # pylint: disable=protected-access
                    target_object.name
                ]
            ),
            held_object=self._pybullet_sim._cubes[  # pylint: disable=protected-access
                target_object.name
            ],
            base_link_to_held_obj=self._pybullet_sim.base_link_to_held_obj,  # pylint: disable=protected-access
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
        assert self._current_base_motion_plan is not None
        # first substep
        if not self._navigated:
            while len(self._current_base_motion_plan) > 1:
                peek_pose = self._current_base_motion_plan[0]
                # Close enough, pop and continue.
                if self._robot_is_close_to_pose(peek_pose):
                    self._current_base_motion_plan.pop(0)
                # Not close enough, stop popping.
                break
            if self._robot_is_close_to_pose(self._current_base_motion_plan[-1]):
                self._navigated = True
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
        if self._navigated and not self._pre_grasp and not self._closed_gripper:
            while len(self._current_arm_joint_plan) > 1:
                peek_conf = self._current_arm_joint_plan[0]
                # Close enough, pop and continue.
                if len(self._current_arm_joint_plan) == 2:
                    if self._robot_is_close_to_conf(peek_conf, atol=0.02):
                        self._current_arm_joint_plan.pop(0)
                else:
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

    def _get_current_robot_pose(self) -> SE2:
        assert self._last_state is not None
        state = self._last_state
        robot = self.objects[0]
        return SE2(
            state.get(robot, "pos_base_x"),
            state.get(robot, "pos_base_y"),
            state.get(robot, "pos_base_rot"),
        )

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
        # return x.get(robot_obj, "pos_gripper")
        if x.get(robot_obj, "pos_gripper") > 0.2:
            return GRASP_CLOSE_THRESHOLD
        return 0.0

    def _robot_is_close_to_conf(
        self, conf: JointPositions, atol: float = 4 * 1e-2
    ) -> bool:
        current_conf = self._get_current_robot_arm_conf()
        assert self._pybullet_sim is not None
        dist = self._pybullet_sim.get_joint_distance(current_conf, conf)
        return dist < atol

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


class PlaceGroundController(GroundParameterizedController[ObjectCentricState, Array]):
    """Controller for motion planning to place a target.

    The object parameters are:
        robot: The robot itself.
        object: The target object.
    """

    def __init__(
        self, *args, pybullet_sim: PyBulletSim | None = None, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._last_state: ObjectCentricState | None = None
        self._current_params: np.ndarray | None = None
        self._current_arm_joint_plan: list[JointPositions] | None = None
        self._current_retract_plan: list[JointPositions] | None = None
        self._current_base_motion_plan: list[SE2] | None = None
        self._pybullet_sim: PyBulletSim | None = pybullet_sim
        self._navigated: bool = False
        self._pre_place: bool = False
        self._open_gripper: bool = False
        self._returned: bool = False
        self._last_gripper_state: float = 0.0
        self.home_joints = np.deg2rad(
            [0, -20, 180, -146, 0, -50, 90, 0, 0, 0, 0, 0, 0]
        )  # retract configuration

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        cupboard_obj = x.get_object_from_name("cupboard_1")
        cupboard_pose = get_overhead_object_se2_pose(x, cupboard_obj)
        rot = BASE_TO_CUPBOARD_ROTATION
        # sample placements
        for _ in range(MAX_SAMPLER_ATTEMPTS):
            pose_x_offset = rng.uniform(*PLACE_SAMPLER_X_OFFSET_BOUNDS)
            pose_y_offset = rng.uniform(*PLACE_SAMPLER_Y_OFFSET_BOUNDS)
            collision = False
            for other_obj in x.get_objects(MujocoMovableObjectType):
                if other_obj.name == self.objects[1].name:
                    continue
                other_object_pose = get_overhead_object_se2_pose(x, other_obj)
                if (
                    np.linalg.norm(
                        np.array(
                            [
                                pose_x_offset
                                + cupboard_pose.x
                                + (
                                    ARM_MOVEMENT_CUPBOARD.position[0]
                                    + ROBOT_ARM_POSE_TO_BASE.position[0]
                                    - BASE_DISTANCE_TO_CUPBOARD
                                ),  # the offset of the cupboard from the cubes.
                                pose_y_offset + cupboard_pose.y,
                            ]
                        )
                        - np.array([other_object_pose.x, other_object_pose.y])
                    )
                    < PLACE_SAMPLER_COLLISION_THRESHOLD
                ):
                    collision = True
                    break
            if not collision:
                return np.array(
                    [BASE_DISTANCE_TO_CUPBOARD + pose_x_offset, pose_y_offset, rot]
                )
        raise ValueError("No valid parameters found")

    def reset(
        self,
        x: ObjectCentricState,
        params: Any,
        extend_xy_magnitude: float = 0.025,
        extend_rot_magnitude: float = np.pi / 8,
    ) -> None:
        # Initialize the PyBullet interface if this is the first time ever.
        if self._pybullet_sim is None:
            self._pybullet_sim = PyBulletSim(x)
        # Update the current state and parameters.
        self._last_state = x

        assert isinstance(params, np.ndarray)
        self._current_params = params.copy()
        # Derive the target pose for the robot.
        target_distance, target_offset, target_rot = self._current_params
        target_object = self.objects[2]
        target_object_pose_temp = get_overhead_object_se2_pose(x, target_object)
        target_object_pose = SE2(
            target_object_pose_temp.x,
            target_object_pose_temp.y + target_offset,
            target_object_pose_temp.theta(),
        )
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
            disable_collision_objects=[self.objects[1].name],
        )
        assert base_motion_plan is not None
        self._current_base_motion_plan = base_motion_plan

        plan_x = x.copy()
        robot = plan_x.get_object_from_name("robot")
        target_base_pose = self._current_base_motion_plan[-1]
        plan_x.set(robot, "pos_base_x", target_base_pose.x)
        plan_x.set(robot, "pos_base_y", target_base_pose.y)
        plan_x.set(robot, "pos_base_rot", target_base_pose.theta())

        target_object_place = self.objects[1]

        assert target_object_place is not None
        # Reset PyBullet given the current state.
        self._pybullet_sim.set_state(plan_x, target_object_place)

        current_arm_base_pose = self._pybullet_sim.robot.get_base_pose()

        target_end_effector_pose = ARM_MOVEMENT_CUPBOARD

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
            collision_bodies=self._pybullet_sim.get_collision_bodies(
                held_object=self._pybullet_sim._cubes[  # pylint: disable=protected-access
                    target_object_place.name
                ]
            ),
            seed=0,  # use a constant seed to make this effectively deterministic
            held_object=self._pybullet_sim._cubes[  # pylint: disable=protected-access
                target_object_place.name
            ],
            base_link_to_held_obj=self._pybullet_sim.base_link_to_held_obj,
            physics_client_id=self._pybullet_sim.physics_client_id,
        )

        retract_plan = run_motion_planning(
            self._pybullet_sim.robot,
            target_joints,
            self.home_joints.tolist(),
            collision_bodies=self._pybullet_sim.get_collision_bodies(
                held_object=self._pybullet_sim._cubes[  # pylint: disable=protected-access
                    target_object_place.name
                ]
            ),
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
        assert self._current_base_motion_plan is not None
        # first substep
        if not self._navigated:
            while len(self._current_base_motion_plan) > 1:
                peek_pose = self._current_base_motion_plan[0]
                # Close enough, pop and continue.
                if self._robot_is_close_to_pose(peek_pose):
                    self._current_base_motion_plan.pop(0)
                # Not close enough, stop popping.
                break
            if self._robot_is_close_to_pose(self._current_base_motion_plan[-1]):
                self._navigated = True
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
        if self._navigated and not self._pre_place and not self._open_gripper:
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

    def _get_current_robot_pose(self) -> SE2:
        assert self._last_state is not None
        state = self._last_state
        robot = self.objects[0]
        return SE2(
            state.get(robot, "pos_base_x"),
            state.get(robot, "pos_base_y"),
            state.get(robot, "pos_base_rot"),
        )

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
        # return x.get(robot_obj, "pos_gripper")
        if x.get(robot_obj, "pos_gripper") > 0.2:
            return GRASP_CLOSE_THRESHOLD
        return 0.0

    def _robot_is_close_to_conf(self, conf: JointPositions) -> bool:
        current_conf = self._get_current_robot_arm_conf()
        assert self._pybullet_sim is not None
        dist = self._pybullet_sim.get_joint_distance(current_conf, conf)
        return dist < 4 * 1e-2

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


def create_lifted_controllers(
    action_space: TidyBot3DRobotActionSpace,
    init_constant_state: ObjectCentricState | None = None,
    pybullet_sim: PyBulletSim | None = None,
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

    # Create wrapper class that captures pybullet_sim
    class PickController(PickGroundController):
        """Pick controller with pre-configured PyBullet sim."""

        def __init__(self, objects):
            super().__init__(pybullet_sim=pybullet_sim, objects=objects)

    class PlaceController(PlaceGroundController):
        """Place controller with pre-configured PyBullet sim."""

        def __init__(self, objects):
            super().__init__(pybullet_sim=pybullet_sim, objects=objects)

    # Pick ground controller.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    target = Variable("?target", MujocoMovableObjectType)

    LiftedPickGroundController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, target],
            PickController,
        )
    )

    # Place controller.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    target = Variable("?target", MujocoMovableObjectType)
    target_place = Variable("?target_place", MujocoFixtureObjectType)

    LiftedPlaceGroundController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, target, target_place],
            PlaceController,
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
