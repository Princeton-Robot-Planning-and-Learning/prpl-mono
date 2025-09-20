"""Base class for Dynamic2D (PyMunk) robot environments."""

import abc
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import gymnasium
import numpy as np
import pymunk
from numpy.typing import NDArray
from relational_structs import (
    Array,
    Object,
    ObjectCentricState,
    ObjectCentricStateSpace,
    Type,
)
from relational_structs.utils import create_state_from_dict

from prbench.core import ObjectCentricPRBenchEnv, PRBenchEnvConfig, RobotActionSpace
from prbench.envs.dynamic2d.object_types import Dynamic2DRobotEnvTypeFeatures
from prbench.envs.dynamic2d.utils import (
    DYNAMIC_COLLISION_TYPE,
    ROBOT_COLLISION_TYPE,
    STATIC_COLLISION_TYPE,
    KinRobot,
    KinRobotActionSpace,
    PDController,
    get_fingered_robot_action_from_gui_input,
    on_collision_w_static,
    on_gripper_grasp,
)
from prbench.envs.geom2d.structs import MultiBody2D
from prbench.envs.utils import render_2dstate


@dataclass(frozen=True)
class Dynamic2DRobotEnvConfig(PRBenchEnvConfig):
    """Scene config for a Dynamic2DRobotEnv."""

    # The world is oriented like a standard X/Y coordinate frame.
    world_min_x: float = 0.0
    world_max_x: float = 10.0
    world_min_y: float = 0.0
    world_max_y: float = 10.0

    # Action space parameters.
    min_dx: float = -1e-2
    max_dx: float = 1e-2
    min_dy: float = -1e-2
    max_dy: float = 1e-2
    min_dtheta: float = -np.pi / 16
    max_dtheta: float = np.pi / 16
    min_darm: float = -5e-2
    max_darm: float = 5e-2
    min_dgripper: float = -0.02
    max_dgripper: float = 0.02

    # Robot parameters
    init_robot_pos: tuple[float, float] = (5.0, 5.0)
    robot_base_radius: float = 0.4
    robot_arm_length_max: float = 0.8
    gripper_base_width: float = 0.01
    gripper_base_height: float = 0.1
    gripper_finger_width: float = 0.1
    gripper_finger_height: float = 0.01
    finger_moving_threshold: float = 1e-3  # threshold to consider finger moving

    # Controller parameters
    # NOTE: Do not modify these parameters and the control_hz, sim_hz unless you
    # understand the implications. These parameters have been tuned to work well
    # with the simulation parameters.
    kp_pos: float = 50.0
    kv_pos: float = 5.0
    kp_rot: float = 50.0
    kv_rot: float = 5.0

    # Physics parameters
    gravity_y: float = -9.8
    collision_slop: float = 0.001  # Allow small interpenetration, depends on env scale
    control_hz: int = 10  # Control frequency (fps in rendering)
    sim_hz: int = 100  # Simulation frequency (dt in simulation)

    # For rendering.
    render_dpi: int = 50


_ConfigType = TypeVar("_ConfigType", bound=Dynamic2DRobotEnvConfig)


class ObjectCentricDynamic2DRobotEnv(
    ObjectCentricPRBenchEnv[ObjectCentricState, Array, _ConfigType],
    Generic[_ConfigType],
):
    """Base class for Dynamic2D robot environments using PyMunk physics.

    This environment uses PyMunk for physics simulation with a KinRobot.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # PyMunk physics space
        self.pymunk_space: pymunk.Space | None = None
        self.robot: KinRobot | None = None
        self.pd_controller = PDController(
            kp_pos=self.config.kp_pos,
            kv_pos=self.config.kv_pos,
            kp_rot=self.config.kp_rot,
            kv_rot=self.config.kv_rot,
        )

        # Initialized by reset().
        self._current_state: ObjectCentricState | None = None
        # Note: We assume each object corresponds to exactly one pymunk body.
        # This does not include the robot, which is handled separately.
        self._state_obj_to_pymunk_body: dict[Object, pymunk.Body] = {}
        # Used for collision checking with Geom2D.
        self._static_object_body_cache: dict[Object, MultiBody2D] = {}

    def _create_observation_space(self, config: _ConfigType) -> ObjectCentricStateSpace:
        types = set(self.type_features)
        return ObjectCentricStateSpace(types)

    def _create_action_space(self, config: _ConfigType) -> RobotActionSpace:
        return KinRobotActionSpace(
            min_dx=config.min_dx,
            max_dx=config.max_dx,
            min_dy=config.min_dy,
            max_dy=config.max_dy,
            min_dtheta=config.min_dtheta,
            max_dtheta=config.max_dtheta,
            min_darm=config.min_darm,
            max_darm=config.max_darm,
            min_dgripper=config.min_dgripper,
            max_dgripper=config.max_dgripper,
        )

    def _create_constant_initial_state(self) -> ObjectCentricState:
        initial_state_dict = self._create_constant_initial_state_dict()
        return create_state_from_dict(initial_state_dict, Dynamic2DRobotEnvTypeFeatures)

    def _setup_physics_space(self) -> None:
        """Set up the PyMunk physics space."""
        self.pymunk_space = pymunk.Space()
        self.pymunk_space.gravity = 0, self.config.gravity_y
        self.pymunk_space.collision_slop = self.config.collision_slop

        # Create robot
        self.robot = KinRobot(
            init_pos=pymunk.Vec2d(*self.config.init_robot_pos),
            base_radius=self.config.robot_base_radius,
            arm_length_max=self.config.robot_arm_length_max,
            gripper_base_width=self.config.gripper_base_width,
            gripper_base_height=self.config.gripper_base_height,
            gripper_finger_width=self.config.gripper_finger_width,
            gripper_finger_height=self.config.gripper_finger_height,
        )
        self.robot.add_to_space(self.pymunk_space)

        # Set up collision handlers
        self.pymunk_space.on_collision(
            DYNAMIC_COLLISION_TYPE,
            ROBOT_COLLISION_TYPE,
            post_solve=on_gripper_grasp,
            data=self.robot,
        )
        self.pymunk_space.on_collision(
            STATIC_COLLISION_TYPE,
            ROBOT_COLLISION_TYPE,
            pre_solve=on_collision_w_static,
            data=self.robot,
        )

    def _reset_robot_in_space(self, obj: Object, state: ObjectCentricState) -> None:
        """Reset the robot in the PyMunk space."""
        assert self.pymunk_space is not None, "Space not initialized"
        robot_base_x = state.get(obj, "x")
        robot_base_y = state.get(obj, "y")
        robot_theta = state.get(obj, "theta")
        robot_arm = state.get(obj, "arm_joint")
        robot_gripper = state.get(obj, "finger_gap")
        assert self.robot is not None, "Robot not initialized"
        self.robot.reset_positions(
            base_x=robot_base_x,
            base_y=robot_base_y,
            base_theta=robot_theta,
            arm_length=robot_arm,
            gripper_gap=robot_gripper,
        )

    @abc.abstractmethod
    def _add_state_to_space(self, state: ObjectCentricState) -> None:
        """Add objects from the state to the PyMunk space."""

    @abc.abstractmethod
    def _read_state_from_space(self) -> None:
        """Read the current state from the PyMunk space."""

    @abc.abstractmethod
    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        """Create the constant initial state dict."""

    @abc.abstractmethod
    def _sample_initial_state(self) -> ObjectCentricState:
        """Use self.np_random to sample an initial state."""

    @abc.abstractmethod
    def _get_reward_and_done(self) -> tuple[float, bool]:
        """Calculate reward and termination based on self._current_state."""

    @property
    def type_features(self) -> dict[Type, list[str]]:
        """The types and features for this environment."""
        return Dynamic2DRobotEnvTypeFeatures

    def _get_obs(self) -> ObjectCentricState:
        """Get observation by reading from the physics simulation."""
        self._read_state_from_space()
        assert self._current_state is not None, "Need to call reset()"
        return self._current_state.copy()

    def _get_info(self) -> dict:
        return {}  # no extra info provided right now

    @property
    def full_state(self) -> ObjectCentricState:
        """Get the full state, which includes both dynamic and static objects."""
        if self._current_state is None:
            raise RuntimeError("Current state is not initialized")
        full_state = self._current_state.copy()
        if self._initial_constant_state is not None:
            # Merge the initial constant state with the current state.
            full_state.data.update(self._initial_constant_state.data)
        return full_state

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[ObjectCentricState, dict]:
        # Reset the random seed.
        gymnasium.Env.reset(self, seed=seed)

        # Clear existing physics space
        if self.pymunk_space:
            # Remove all bodies and shapes
            for body in list(self.pymunk_space.bodies):
                for shape in list(body.shapes):
                    if body in self.pymunk_space.bodies:
                        self.pymunk_space.remove(body, shape)
            for shape in list(self.pymunk_space.shapes):
                # Some shapes are not attached to bodies (e.g., static lines)
                self.pymunk_space.remove(shape)

        # Set up new physics space
        self._setup_physics_space()
        self._static_object_body_cache = {}
        self._state_obj_to_pymunk_body = {}

        # For testing purposes only, the options may specify an initial scene.
        if options is not None and "init_state" in options:
            self._current_state = options["init_state"].copy()
        # Otherwise, set up the initial scene here.
        else:
            self._current_state = self._sample_initial_state()

        # Add objects to physics space
        self._add_state_to_space(self.full_state)

        # Calculate simulation parameters
        dt = 1.0 / self.config.sim_hz
        # Stepping physics to let things settle
        assert self.pymunk_space is not None, "Space not initialized"
        for _ in range(self.config.sim_hz):
            self.pymunk_space.step(dt)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: Array) -> tuple[ObjectCentricState, float, bool, bool, dict]:
        assert self.action_space.contains(action)
        dx, dy, dtheta, darm, dgripper = action
        assert self._current_state is not None, "Need to call reset()"
        assert self.pymunk_space is not None, "Space not initialized"
        assert self.robot is not None, "Robot not initialized"

        # Calculate simulation parameters
        sim_dt = 1.0 / self.config.sim_hz
        control_dt = 1.0 / self.config.control_hz
        n_steps = self.config.sim_hz // self.config.control_hz

        # Calculate target positions
        tgt_x = self.robot.base_pose.x + dx
        tgt_y = self.robot.base_pose.y + dy
        tgt_theta = self.robot.base_pose.theta + dtheta
        tgt_arm = max(
            min(self.robot.curr_arm_length + darm, self.robot.arm_length_max),
            self.robot.base_radius,
        )
        tgt_gripper = max(
            min(self.robot.curr_gripper + dgripper, self.robot.gripper_gap_max),
            self.robot.gripper_finger_height * 2,
        )

        # Set robot finger state (for grasping and dropping behavior)
        curr_held_obj_pymunk_idx = [
            kin_obj[0].id for kin_obj, _, _ in self.robot.held_objects
        ]
        # Allow small change in finger gap without changing state
        if tgt_gripper - self.robot.curr_gripper > self.config.finger_moving_threshold:
            self.robot.is_opening_finger = True
            self.robot.is_closing_finger = False
        elif (
            self.robot.curr_gripper - tgt_gripper > self.config.finger_moving_threshold
        ):
            self.robot.is_opening_finger = False
            self.robot.is_closing_finger = True
        else:
            self.robot.is_opening_finger = False
            self.robot.is_closing_finger = False

        # Multi-step simulation like pushT env
        for _ in range(n_steps):
            # Use PD control to compute base and gripper velocities
            (
                base_vel,
                base_ang_vel,
                gripper_base_vel,
                finger_vel_l,
                held_obj_vel,
            ) = self.pd_controller.compute_control(
                self.robot,
                tgt_x,
                tgt_y,
                tgt_theta,
                tgt_arm,
                tgt_gripper,
                control_dt,
            )
            # Update robot with the vel (PD control updates velocities)
            self.robot.update(
                base_vel,
                base_ang_vel,
                gripper_base_vel,
                finger_vel_l,
                held_obj_vel,
            )
            # Step physics simulation (more fine-grained than control freq)
            for _ in range(self.config.sim_hz // self.config.control_hz):
                self.pymunk_space.step(sim_dt)

        # NOTE: We currently assume the robot can only grasp one object at a time.
        # Assuming that the missing object body and the new body is naturally matched.
        # Need an update in the future if we allow multiple objects to be grasped.
        new_held_obj_pymunk_idx = {
            kin_obj[0].id: kin_obj[0] for kin_obj, _, _ in self.robot.held_objects
        }
        if len(list(new_held_obj_pymunk_idx.keys())) > 1:
            self.robot.is_closing_finger = False
            self.robot.is_opening_finger = True  # Grasping failed, open fingers

        if (
            list(new_held_obj_pymunk_idx.keys()) != curr_held_obj_pymunk_idx
        ) and self.robot.is_closing_finger:
            # Grasping happened
            assert len(curr_held_obj_pymunk_idx) == 0
            new_body = list(new_held_obj_pymunk_idx.values())[0]
            for state_obj, body in self._state_obj_to_pymunk_body.items():
                if body not in self.pymunk_space.bodies:
                    # This is the object that got grasped
                    self._state_obj_to_pymunk_body[state_obj] = new_body

        # Drop objects after internal steps
        if self.robot.is_opening_finger:
            # First, for all state objects, if their pymunk body is held,
            # change to dynamic body and remove the held body.
            held_body_ids_shape = {
                kin_obj[0].id: kin_obj[1] for kin_obj, _, _ in self.robot.held_objects
            }
            for state_obj, body in self._state_obj_to_pymunk_body.items():
                if body.id in held_body_ids_shape.keys():
                    # Change to dynamic body
                    kinematic_body = body
                    points = held_body_ids_shape[body.id].get_vertices()
                    mass = 1.0  # Assume uniform mass for now
                    moment = pymunk.moment_for_poly(mass, points, (0, 0))
                    dynamic_body = pymunk.Body(mass, moment)
                    dynamic_body.position = kinematic_body.position
                    dynamic_body.velocity = kinematic_body.velocity  # Preserve velocity
                    dynamic_body.angular_velocity = kinematic_body.angular_velocity
                    dynamic_body.angle = kinematic_body.angle
                    shape = pymunk.Poly(dynamic_body, points)
                    shape.friction = 1.0
                    shape.density = 1.0
                    shape.collision_type = DYNAMIC_COLLISION_TYPE
                    self.pymunk_space.add(dynamic_body, shape)
                    self.pymunk_space.remove(
                        kinematic_body, held_body_ids_shape[body.id]
                    )
                    self._state_obj_to_pymunk_body[state_obj] = dynamic_body
                    held_body_ids_shape.pop(body.id)
            # Then, for any remaining held objects not matched to state objects,
            # directly remove them from the space.
            for kin_obj, _, _ in self.robot.held_objects:
                if kin_obj[0].id in held_body_ids_shape.keys():
                    # Remove any extra objects in hand
                    self.pymunk_space.remove(kin_obj[0], kin_obj[1])
                    continue
            self.robot.held_objects = []

        self.robot.is_closing_finger = False  # reset
        self.robot.is_opening_finger = False

        reward, terminated = self._get_reward_and_done()
        truncated = False  # no maximum horizon, by default
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def render(self) -> NDArray[np.uint8]:  # type: ignore
        """Render the current state.

        The same as Geom2D render for now.
        """
        # This will be implemented later
        assert self.render_mode == "rgb_array"
        assert self._current_state is not None, "Need to call reset()"
        render_input_state = self._current_state.copy()
        if self._initial_constant_state is not None:
            # Merge the initial constant state with the current state.
            render_input_state.data.update(self._initial_constant_state.data)
        return render_2dstate(
            render_input_state,
            self._static_object_body_cache,
            self.config.world_min_x,
            self.config.world_max_x,
            self.config.world_min_y,
            self.config.world_max_y,
            self.config.render_dpi,
        )

    def get_action_from_gui_input(
        self, gui_input: dict[str, Any]
    ) -> NDArray[np.float32]:
        """Get the mapping from human inputs to actions."""
        # This will be implemented later
        assert isinstance(self.action_space, KinRobotActionSpace)
        return get_fingered_robot_action_from_gui_input(self.action_space, gui_input)
