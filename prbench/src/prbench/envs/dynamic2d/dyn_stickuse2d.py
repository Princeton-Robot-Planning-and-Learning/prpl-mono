"""Dynamic StickBlock 2D env using PyMunk physics."""

import inspect
from dataclasses import dataclass

import numpy as np
import pymunk
from relational_structs import Object, ObjectCentricState, Type
from relational_structs.utils import create_state_from_dict

from prbench.core import ConstantObjectPRBenchEnv
from prbench.envs.dynamic2d.base_env import (
    Dynamic2DRobotEnvConfig,
    ObjectCentricDynamic2DRobotEnv,
)
from prbench.envs.dynamic2d.object_types import (
    Dynamic2DRobotEnvTypeFeatures,
    DynRectangleType,
    KinRectangleType,
    KinRobotType,
)
from prbench.envs.dynamic2d.utils import (
    DYNAMIC_COLLISION_TYPE,
    STATIC_COLLISION_TYPE,
    KinRobotActionSpace,
    create_walls_from_world_boundaries,
)
from prbench.envs.geom2d.structs import MultiBody2D, SE2Pose, ZOrder
from prbench.envs.geom2d.utils import is_on
from prbench.envs.utils import PURPLE, BROWN, sample_se2_pose, state_2d_has_collision

TargetBlockType = Type("target_block", parent=DynRectangleType)
StickType = Type("stick", parent=KinRectangleType)
Dynamic2DRobotEnvTypeFeatures[TargetBlockType] = list(
    Dynamic2DRobotEnvTypeFeatures[DynRectangleType]
)
Dynamic2DRobotEnvTypeFeatures[StickType] = list(
    Dynamic2DRobotEnvTypeFeatures[KinRectangleType]
)

@dataclass(frozen=True)
class DynPushPullStick2DEnvConfig(Dynamic2DRobotEnvConfig):
    """Scene config for DynPushPullStick2DEnv()."""

    # World boundaries. Standard coordinate frame with (0, 0) in bottom left.
    world_min_x: float = 0.0
    world_max_x: float = 3.5
    world_min_y: float = 0.0
    world_max_y: float = 2.5

    # Robot parameters
    init_robot_pos: tuple[float, float] = (0.5, 0.5)
    robot_base_radius: float = 0.24
    robot_arm_length_max: float = 2 * robot_base_radius
    gripper_base_width: float = 0.06
    gripper_base_height: float = 0.32
    gripper_finger_width: float = 0.2
    gripper_finger_height: float = 0.06

    # Action space parameters.
    min_dx: float = -5e-2
    max_dx: float = 5e-2
    min_dy: float = -5e-2
    max_dy: float = 5e-2
    min_dtheta: float = -np.pi / 16
    max_dtheta: float = np.pi / 16
    min_darm: float = -1e-1
    max_darm: float = 1e-1
    min_dgripper: float = -0.02
    max_dgripper: float = 0.02

    # Controller parameters
    kp_pos: float = 50.0
    kv_pos: float = 5.0
    kp_rot: float = 50.0
    kv_rot: float = 5.0

    # Robot hyperparameters.
    robot_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(0.5, 0.5, -np.pi / 2),
        SE2Pose(3.0, 1.0, np.pi / 2),
    )

    # Middle wall hyperparameters.
    middle_wall_rgb: tuple[float, float, float] = PURPLE
    middle_wall_pose: SE2Pose = (
        (world_min_x + world_max_x) / 2,
        (world_min_y + world_max_y) / 2,
        0.0,
    )
    middle_wall_width: float = world_max_x - world_min_x
    middle_wall_height: float = 0.1

    # Stick hyperparameters.
    stick_rgb: tuple[float, float, float] = BROWN
    stick_shape: tuple[float, float] = (
        gripper_base_height / 2,
        (world_min_y + world_max_y) * 2 / 3,
    )
    stick_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(
            world_min_x, 
            (world_min_y + world_max_y) / 2 - stick_shape[1] / 4,
            np.pi / 4
        ),
        SE2Pose(
            world_max_x - stick_shape[0], 
            (world_min_y + world_max_y) / 2 + stick_shape[1] / 4, 
            3 * np.pi / 4
        ),
    )

    # Target block hyperparameters.
    target_block_rgb: tuple[float, float, float] = PURPLE
    target_block_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(
            world_min_x, (world_min_y + world_max_y) / 2 + robot_base_radius, -np.pi
        ),
        SE2Pose(
            world_max_x, world_max_y, np.pi
        ),
    )
    target_block_size_bounds: tuple[float, float] = (
        gripper_base_height / 2,
        gripper_base_height * 2 / 3
    )
    target_block_mass: float = 1.0

    # Obstruction hyperparameters (DYNAMIC).
    obstruction_rgb: tuple[float, float, float] = BROWN
    obstruction_init_pose_bounds = (
        SE2Pose(
            world_min_x, (world_min_y + world_max_y) / 2 + robot_base_radius, -np.pi
        ),
        SE2Pose(
            world_max_x, world_max_y, np.pi
        ),
    )
    obstruction_height_bounds: tuple[float, float] = (
        robot_base_radius / 2,
        2 * robot_base_radius,
    )
    obstruction_width_bounds: tuple[float, float] = (
        robot_base_radius / 2,
        2 * robot_base_radius,
    )
    obstruction_block_mass: float = 1.0
    # NOTE: obstruction poses are sampled using a 2D gaussian that is centered
    # at the target location. This hyperparameter controls the variance.
    # borrowed from clutteredretrieval2d
    obstruction_pose_init_distance_scale: float = 0.25

    # For sampling initial states.
    max_initial_state_sampling_attempts: int = 10_000

    # For rendering.
    render_dpi: int = 250


class ObjectCentricDynPushPullStick2DEnv(
    ObjectCentricDynamic2DRobotEnv[DynPushPullStick2DEnvConfig]
):
    """
    """

    def __init__(
        self,
        num_targets: int = 1,
        num_obstructions: int = 2,
        config: DynPushPullStick2DEnvConfig = DynPushPullStick2DEnvConfig(),
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)
        self._num_obstructions = num_obstructions
        self._num_targets = num_targets

        # Store object references for tracking
        self._target_blocks: list[Object] = []

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        init_state_dict: dict[Object, dict[str, float]] = {}

        # Create the middle wall.
        middle_wall = Object("middle_wall", KinRectangleType)
        init_state_dict[middle_wall] = {
            "x": self.config.middle_wall_pose.x,
            "vx": 0.0,
            "y": self.config.middle_wall_pose.y,
            "vy": 0.0,
            "theta": self.config.middle_wall_pose.theta,
            "omega": 0.0,
            "width": self.config.middle_wall_width,
            "height": self.config.middle_wall_height,
            "static": True,
            "color_r": self.config.middle_wall_rgb[0],
            "color_g": self.config.middle_wall_rgb[1],
            "color_b": self.config.middle_wall_rgb[2],
            "z_order": ZOrder.FLOOR.value, # Middle wall does not collide with hook
        }

        # Create room walls.
        assert isinstance(self.action_space, KinRobotActionSpace)
        min_dx, min_dy = self.action_space.low[:2]
        max_dx, max_dy = self.action_space.high[:2]
        wall_state_dict = create_walls_from_world_boundaries(
            self.config.world_min_x,
            self.config.world_max_x,
            self.config.world_min_y,
            self.config.world_max_y,
            min_dx,
            max_dx,
            min_dy,
            max_dy,
        )
        init_state_dict.update(wall_state_dict)

        return init_state_dict

    def _sample_initial_state(self) -> ObjectCentricState:
        """Sample an initial state for the environment."""
        static_objects = set(self.initial_constant_state)
        n = self.config.max_initial_state_sampling_attempts
        robot_pose = sample_se2_pose(
            self.config.robot_init_pose_bounds, self.np_random
        )
        state = self._create_initial_state(robot_pose)
        robot = state.get_objects(KinRobotType)[0]
        # Check for collisions with the robot and static objects.
        full_state = state.copy()
        full_state.data.update(self.initial_constant_state.data)
        assert not state_2d_has_collision(full_state, {robot}, static_objects, {})
        for _ in range(n):
            target_pose = sample_se2_pose(
                self.config.target_block_init_pose_bounds, self.np_random
            )
            target_size = self.np_random.uniform(
                *self.config.target_block_size_bounds
            )
            stick_pose = sample_se2_pose(
                self.config.stick_init_pose_bounds, self.np_random
            )
            state = self._create_initial_state(
                robot_pose,
                target_pose=target_pose,
                target_size=target_size,
                stick_pose=stick_pose,
            )
            target_block = state.get_objects(TargetBlockType)[0]
            stick = state.get_objects(StickType)[0]
            full_state = state.copy()
            full_state.data.update(self.initial_constant_state.data)
            if not state_2d_has_collision(
                full_state, {target_block}, {robot, stick} | static_objects, {}
            ):
                break
        else:
            raise RuntimeError("Failed to sample target pose.")

        # Sample obstructions one by one. Assume that the scene is never so dense
        # that we need to resample earlier choices.
        obstructions: list[tuple[SE2Pose, tuple[float, float]]] = []
        for _ in range(self._num_obstructions):
            for _ in range(n):
                # Sample xy, relative to the target.
                x, y = self.np_random.normal(
                    loc=(target_pose.x, target_pose.y),
                    scale=self.config.obstruction_pose_init_distance_scale,
                    size=(2,),
                )
                # Make sure in bounds.
                if not (
                    self.config.world_min_x < x < self.config.world_max_x
                    and self.config.world_min_y < y < self.config.world_max_y
                ):
                    continue
                # Sample theta.
                theta = self.np_random.uniform(-np.pi, np.pi)
                # Check for collisions.
                obstruction_pose = SE2Pose(x, y, theta)
                # Sample shape.
                obstruction_shape = (
                    self.np_random.uniform(*self.config.obstruction_width_bounds),
                    self.np_random.uniform(*self.config.obstruction_height_bounds),
                )
                possible_obstructions = obstructions + [
                    (obstruction_pose, obstruction_shape)
                ]
                state = self._create_initial_state(
                    robot_pose,
                    target_pose=target_pose,
                    target_size=target_size,
                    stick_pose=stick_pose,
                    obstructions=possible_obstructions,
                )
                obj_name_to_obj = {o.name: o for o in state}
                full_state = state.copy()
                full_state.data.update(self.initial_constant_state.data)
                new_obstruction = obj_name_to_obj[f"obstruction{len(obstructions)}"]
                assert new_obstruction.name.startswith("obstruction")
                if not state_2d_has_collision(
                    full_state, {new_obstruction}, set(full_state), {}
                ):
                    break
            else:
                raise RuntimeError("Failed to sample obstruction pose.")
            # Update obstructions.
            obstructions = possible_obstructions
        # The state should already be finalized.
        return state

    def _create_initial_state(
        self,
        robot_pose: SE2Pose,
        target_pose: SE2Pose | None = None,
        target_size: float | None = None,
        stick_pose: SE2Pose | None = None,
        obstructions: list[tuple[SE2Pose, tuple[float, float]]] | None = None,
    ) -> ObjectCentricState:
        # Shallow copy should be okay because the constant objects should not
        # ever change in this method.
        init_state_dict: dict[Object, dict[str, float]] = {}

        # Create the robot.
        robot = Object("robot", KinRobotType)
        init_state_dict[robot] = {
            "x": robot_pose.x,
            "y": robot_pose.y,
            "theta": robot_pose.theta,
            "vx_base": 0.0,
            "vy_base": 0.0,
            "omega_base": 0.0,
            "static": False,
            "base_radius": self.config.robot_base_radius,
            "arm_joint": self.config.robot_base_radius,
            "arm_length": self.config.robot_arm_length_max,
            "gripper_base_width": self.config.gripper_base_width,
            "gripper_base_height": self.config.gripper_base_height,
            "finger_gap": self.config.gripper_base_height,
            "finger_height": self.config.gripper_finger_height,
            "finger_width": self.config.gripper_finger_width,
        }

        # Create the stick.
        if target_pose is not None:
            assert target_size is not None
            target_block = Object("target_block", TargetBlockType)
            init_state_dict[target_block] = {
                "x": target_pose.x,
                "vx": 0.0,
                "y": target_pose.y,
                "vy": 0.0,
                "theta": target_pose.theta,
                "omega": 0.0,
                "width": target_size,
                "height": target_size,
                "static": True,
                "held": False,
                "mass": self.config.target_block_mass,
                "color_r": self.config.target_block_rgb[0],
                "color_g": self.config.target_block_rgb[1],
                "color_b": self.config.target_block_rgb[2],
                "z_order": ZOrder.SURFACE.value, # Hook does not collide with middle wall
            }

        # Create the stick.
        if stick_pose is not None:
            target_block = Object("stick", StickType)
            init_state_dict[target_block] = {
                "x": stick_pose.x,
                "vx": 0.0,
                "y": stick_pose.y,
                "vy": 0.0,
                "theta": stick_pose.theta,
                "omega": 0.0,
                "width": self.config.stick_shape[0],
                "height": self.config.stick_shape[1],
                "static": False,
                "held": False,
                "color_r": self.config.stick_rgb[0],
                "color_g": self.config.stick_rgb[1],
                "color_b": self.config.stick_rgb[2],
                "z_order": ZOrder.SURFACE.value, # Hook does not collide with middle wall
            }

        # Create obstructions.
        if obstructions:
            for i, (obstruction_pose, obstruction_shape) in enumerate(obstructions):
                obstruction = Object(f"obstruction{i}", DynRectangleType)
                init_state_dict[obstruction] = {
                    "x": obstruction_pose.x,
                    "vx": 0.0,
                    "y": obstruction_pose.y,
                    "vy": 0.0,
                    "theta": obstruction_pose.theta,
                    "omega": 0.0,
                    "mass": self.config.obstruction_block_mass,
                    "width": obstruction_shape[0],
                    "height": obstruction_shape[1],
                    "static": False,
                    "color_r": self.config.obstruction_rgb[0],
                    "color_g": self.config.obstruction_rgb[1],
                    "color_b": self.config.obstruction_rgb[2],
                    "z_order": ZOrder.ALL.value,
                }

        # Finalize state.
        return create_state_from_dict(init_state_dict, Dynamic2DRobotEnvTypeFeatures)

    def _add_state_to_space(self, state: ObjectCentricState) -> None:
        """Add objects from the state to the PyMunk space."""
        raise NotImplementedError("TODO")

    def _read_state_from_space(self) -> None:
        """Read the current state from the PyMunk space."""
        raise NotImplementedError("TODO")

    def _target_satisfied(
        self,
        state: ObjectCentricState,
        static_object_body_cache: dict[Object, MultiBody2D],
    ) -> bool:
        """Check if the target condition is satisfied.
        """
        raise NotImplementedError("TODO")

    def _get_reward_and_done(self) -> tuple[float, bool]:
        """Calculate reward and termination."""
        assert self._current_state is not None
        terminated = self._target_satisfied(
            self._current_state,
            self._static_object_body_cache,
        )
        return -1.0, terminated


class DynPushPullStick2DEnv(ConstantObjectPRBenchEnv):
    """Dynamic Push-Pull Stick 2D env with a constant number of objects."""

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricDynPushPullStick2DEnv:
        return ObjectCentricDynPushPullStick2DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        constant_objects = ["robot", "stick"]
        for obj in sorted(exemplar_state):
            if obj.name.startswith("target_block"):
                constant_objects.append(obj.name)
            if obj.name.startswith("obstruction"):
                constant_objects.append(obj.name)
        return constant_objects

    def _create_env_markdown_description(self) -> str:
        # Count obstruction objects (exclude target_surface, target_block, and robot)
        num_obstructions = len(
            [obj for obj in self._constant_objects if obj.name.startswith("obstruct")]
        )
        # pylint: disable=line-too-long
        if num_obstructions > 0:
            obstruction_sentence = f"\nThe target surface may be initially obstructed. In this environment, there are always {num_obstructions} obstacle blocks.\n"
        else:
            obstruction_sentence = ""

        return f"""A 2D physics-based environment where the goal is to place a target block onto a target surface using a fingered robot with PyMunk physics simulation. The block must be completely on the surface.
{obstruction_sentence}
The robot has a movable circular base and an extendable arm with gripper fingers. Objects can be grasped and released through gripper actions. All objects follow realistic physics including gravity, friction, and collisions.

**Observation Space**: The observation is a fixed-size vector containing the state of all objects:
- **Robot**: position (x,y), orientation (θ), velocities (vx,vy,ω), arm extension, gripper gap
- **Target Block**: position, orientation, velocities, dimensions (dynamic physics object)
- **Target Surface**: position, orientation, dimensions (kinematic physics object)
{f"- **Obstruction Blocks** ({num_obstructions}): position, orientation, velocities, dimensions (dynamic physics objects)" if num_obstructions > 0 else ""}

Each object includes physics properties like mass, moment of inertia (for dynamic objects), and color information for rendering.
"""

    def _create_reward_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return f"""A penalty of -1.0 is given at every time step until termination, which occurs when the target block is completely "on" the target surface.

**Termination Condition**: The episode terminates when the target block is successfully placed on the target surface. The "on" condition requires that the bottom vertices of the target block are within the bounds of the target surface, accounting for physics-based positioning.

The definition of "on" is implemented using geometric collision detection:
```python
{inspect.getsource(is_on)}```

**Physics Integration**: Since this environment uses PyMunk physics simulation, objects have realistic dynamics including:
- Gravity (objects fall if not supported)
- Friction between surfaces
- Collision response and momentum transfer
- Realistic grasping and manipulation dynamics
"""

    def _create_references_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return """This is a physics-based version of manipulation environments commonly used in robotics research. It extends the geometric obstruction environment to include realistic physics simulation using PyMunk.

**Key Features**:
- **PyMunk Physics Engine**: Provides realistic 2D rigid body dynamics
- **Dynamic Objects**: Target and obstruction blocks have mass, inertia, and respond to forces
- **Kinematic Robot**: Multi-DOF robot with base movement, arm extension, and gripper control
- **Collision Detection**: Physics-based collision handling for grasping and object interactions
- **Gravity Simulation**: Objects fall and settle naturally under gravitational forces

**Research Applications**:
- Robot manipulation learning with realistic physics
- Grasping and placement strategy development  
- Multi-object interaction scenarios
- Physics-aware motion planning validation
- Comparative studies between geometric and physics-based environments

This environment enables more realistic evaluation of manipulation policies compared to purely geometric versions, as agents must account for momentum, friction, and gravitational effects.
"""
