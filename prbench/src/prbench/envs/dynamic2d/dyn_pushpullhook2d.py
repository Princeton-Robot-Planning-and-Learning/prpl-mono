"""Dynamic PushPullHook 2D env using PyMunk physics."""

import inspect
from dataclasses import dataclass

import numpy as np
import pymunk
from relational_structs import Object, ObjectCentricState
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
    LObjectType,
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
from prbench.envs.utils import PURPLE, RED, BROWN, sample_se2_pose, state_2d_has_collision


@dataclass(frozen=True)
class DynPushPullHook2DEnvConfig(Dynamic2DRobotEnvConfig):
    """Scene config for DynPushPullHook2DEnv()."""

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

    # Hook hyperparameters.
    hook_rgb: tuple[float, float, float] = BROWN
    hook_shape: tuple[float, float, float] = (
        gripper_base_height / 2,
        (world_min_y + world_max_y) * 2 / 3,
        (world_min_x + world_max_x) / 6,
    )
    hook_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(
            world_min_x, 
            (world_min_y + world_max_y) / 2 - hook_shape[1] / 4,
            np.pi / 4
        ),
        SE2Pose(
            world_max_x - hook_shape[0], 
            (world_min_y + world_max_y) / 2 + hook_shape[1] / 4, 
            3 * np.pi / 4
        ),
    )

    # Target block hyperparameters.
    target_block_rgb: tuple[float, float, float] = RED
    target_block_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(
            world_min_x, (world_min_y + world_max_y) / 2 + robot_base_radius, -np.pi
        ),
        SE2Pose(
            world_max_x, world_max_y, np.pi
        ),
    )
    target_block_size_bounds: tuple[float, float] = (
        hook_shape[0],
        hook_shape[2],
    )
    target_block_mass: float = 1.0

    # Obstruction hyperparameters (DYNAMIC).
    obstruction_rgb: tuple[float, float, float] = RED
    obstruction_init_pose_bounds = (
        SE2Pose(
            world_min_x, (world_min_y + world_max_y) / 2 + robot_base_radius, -np.pi
        ),
        SE2Pose(
            world_max_x, world_max_y, np.pi
        ),
    )
    obstruction_size_bounds: tuple[float, float] = (
        hook_shape[0],
        hook_shape[2],
    )
    obstruction_block_mass: float = 1.0

    # For sampling initial states.
    max_initial_state_sampling_attempts: int = 10_000

    # For rendering.
    render_dpi: int = 250


class ObjectCentricDynPushPullHook2DEnv(
    ObjectCentricDynamic2DRobotEnv[DynPushPullHook2DEnvConfig]
):
    """
    """

    def __init__(
        self,
        num_targets: int = 1,
        num_obstructions: int = 2,
        config: DynPushPullHook2DEnvConfig = DynPushPullHook2DEnvConfig(),
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
        n = self.config.max_initial_state_sampling_attempts
        for _ in range(n):
            # Sample all randomized values.
            robot_pose = sample_se2_pose(
                self.config.robot_init_pose_bounds, self.np_random
            )
            hook_pose = sample_se2_pose(
                self.config.hook_init_pose_bounds, self.np_random
            )
            targets: list[tuple[SE2Pose, tuple[float, float]]] = []
            for _ in range(self._num_targets):
                target_block_shape = self.np_random.uniform(
                    *self.config.target_block_size_bounds, size=2
                )
                target_block_pose = sample_se2_pose(
                    self.config.target_block_init_pose_bounds, self.np_random
                )
                targets.append((target_block_pose, target_block_shape))

            obstructions: list[tuple[SE2Pose, tuple[float, float]]] = []
            for _ in range(self._num_obstructions):
                # For now just random sample everywhere.
                obstruction_shape = self.np_random.uniform(
                    *self.config.obstruction_size_bounds, size=2
                )
                obstruction_pose = sample_se2_pose(self.config.obstruction_init_pose_bounds, self.np_random)
                obstructions.append((obstruction_pose, obstruction_shape))

            state = self._create_initial_state(
                robot_pose,
                hook_pose,
                targets,
                obstructions,
            )

            # Check initial state validity: goal not satisfied and no collisions.
            if self._target_satisfied(state, {}):
                continue
            full_state = state.copy()
            if self._initial_constant_state is not None:
                full_state.data.update(self._initial_constant_state.data)
            all_objects = set(full_state)
            # We use Geom2D collision checker for now, maybe need to update it.
            if state_2d_has_collision(full_state, all_objects, all_objects, {}):
                continue
            return state

        raise RuntimeError(f"Failed to sample initial state after {n} attempts")

    def _create_initial_state(
        self,
        robot_pose: SE2Pose,
        hook_pose: SE2Pose,
        targets: list[tuple[SE2Pose, tuple[float, float]]],
        obstructions: list[tuple[SE2Pose, tuple[float, float]]],
    ) -> ObjectCentricState:
        # Shallow copy should be okay because the constant objects should not
        # ever change in this method.
        init_state_dict: dict[Object, dict[str, float]] = {}

        # Create the robot.
        robot = Object("robot", KinRobotType)
        init_state_dict[robot] = {
            "x": robot_pose.x,
            "vx": 0.0,
            "y": robot_pose.y,
            "vy": 0.0,
            "theta": robot_pose.theta,
            "omega": 0.0,
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

        # Create the hook.
        target_block = Object("hook", LObjectType)
        init_state_dict[target_block] = {
            "x": hook_pose.x,
            "vx": 0.0,
            "y": hook_pose.y,
            "vy": 0.0,
            "theta": hook_pose.theta,
            "omega": 0.0,
            "width": self.config.hook_shape[0],
            "length_side1": self.config.hook_shape[1],
            "length_side2": self.config.hook_shape[2],
            "static": False,
            "mass": self.config.target_block_mass,
            "color_r": self.config.target_block_rgb[0],
            "color_g": self.config.target_block_rgb[1],
            "color_b": self.config.target_block_rgb[2],
            "z_order": ZOrder.SURFACE.value, # Hook does not collide with middle wall
        }

        # Create the target blocks.
        for i, (target_block_pose, target_block_shape) in enumerate(targets):
            target_block = Object(f"target_block{i}", DynRectangleType)
            init_state_dict[target_block] = {
                "x": target_block_pose.x,
                "vx": 0.0,
                "y": target_block_pose.y,
                "vy": 0.0,
                "theta": target_block_pose.theta,
                "omega": 0.0,
                "width": target_block_shape[0],
                "height": target_block_shape[1],
                "static": False,
                "mass": self.config.target_block_mass,
                "color_r": self.config.target_block_rgb[0],
                "color_g": self.config.target_block_rgb[1],
                "color_b": self.config.target_block_rgb[2],
                "z_order": ZOrder.ALL.value,
            }
            self._target_blocks.append(target_block)

        # Create obstructions.
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
        assert self.pymunk_space is not None, "Space not initialized"

        # Add static objects (table, walls)
        for obj in state:
            if obj.is_instance(KinRobotType):
                self._reset_robot_in_space(obj, state)
            else:
                # Everything else are rectangles in this environment.
                x = state.get(obj, "x")
                y = state.get(obj, "y")
                width = state.get(obj, "width")
                height = state.get(obj, "height")
                theta = state.get(obj, "theta")

                if (
                    (obj.name == "table")
                    or "wall" in obj.name
                ):
                    # Static objects
                    # We use Pymunk kinematic bodies for static objects
                    b2 = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
                    vs = [
                        (-width / 2, -height / 2),
                        (-width / 2, height / 2),
                        (width / 2, height / 2),
                        (width / 2, -height / 2),
                    ]
                    shape = pymunk.Poly(b2, vs)
                    shape.friction = 1.0
                    shape.density = 1.0
                    shape.mass = 1.0
                    shape.elasticity = 0.99
                    shape.collision_type = STATIC_COLLISION_TYPE
                    self.pymunk_space.add(b2, shape)
                    b2.position = x, y
                    b2.angle = theta
                    self._state_obj_to_pymunk_body[obj] = b2
                elif obj.is_instance(DynRectangleType):
                    # Dynamic blocks
                    mass = state.get(obj, "mass")
                    moment = pymunk.moment_for_box(mass, (width, height))
                    body = pymunk.Body()
                    vs = [
                        (-width / 2, -height / 2),
                        (-width / 2, height / 2),
                        (width / 2, height / 2),
                        (width / 2, -height / 2),
                    ]
                    shape = pymunk.Poly(body, vs)
                    shape.friction = 1.0
                    shape.density = 1.0
                    shape.collision_type = DYNAMIC_COLLISION_TYPE
                    shape.mass = mass
                    assert shape.body is not None
                    shape.body.moment = moment
                    shape.body.mass = mass
                    self.pymunk_space.add(body, shape)
                    body.position = x, y
                    body.angle = theta
                    self._state_obj_to_pymunk_body[obj] = body
                elif obj.is_instance(LObjectType):
                    # Dynamic L-shaped object (the hook)
                    mass = state.get(obj, "mass")
                    x, y = state.get(obj, "x"), state.get(obj, "y")
                    theta = state.get(obj, "theta")
                    l1 = state.get(obj, "length_side1")
                    l2 = state.get(obj, "length_side2")
                    w = state.get(obj, "width")
                    # Approximate moment of inertia for L-shape as two rectangles
                    moment1 = pymunk.moment_for_box(mass / 2, (width, length_side1))
                    moment2 = pymunk.moment_for_box(mass / 2, (width, length_side2))
                    moment = moment1 + moment2
                    body = pymunk.Body()
                    vertices = np.array(
                        [
                            (0, 0),
                            (-l1, 0),
                            (-l1, -w),
                            (-w, -w),
                            (-w, -l2),
                            (0, -l2),
                            (0, -w),
                            (-w, 0),
                        ]
                    )
                    vs_l1 = (
                        vertices[0],
                        vertices[1],
                        vertices[2],
                        vertices[6],
                    )
                    vs_l2 = (
                        vertices[4],
                        vertices[5],
                        vertices[0],
                        vertices[7],
                    )
                    shape1 = pymunk.Poly(body, vs_l1)
                    shape2 = pymunk.Poly(body, vs_l2)
                    shape1.friction = 1.0
                    shape1.density = 1.0
                    shape1.collision_type = DYNAMIC_COLLISION_TYPE
                    shape1.mass = mass / 2
                    shape2.friction = 1.0
                    shape2.density = 1.0
                    shape2.collision_type = DYNAMIC_COLLISION_TYPE
                    shape2.mass = mass / 2
                    self.pymunk_space.add(body, shape1, shape2)
                    body.position = x, y
                    body.angle = theta
                    body.moment = moment
                    body.mass = mass
                    self._state_obj_to_pymunk_body[obj] = body

    def _read_state_from_space(self) -> None:
        """Read the current state from the PyMunk space."""
        assert self.pymunk_space is not None, "Space not initialized"
        assert self._current_state is not None, "Current state not initialized"

        state = self._current_state.copy()

        # Update dynamic object positions from PyMunk simulation
        for obj in state:
            if state.get(obj, "static"):
                continue
            if obj.is_instance(KinRobotType):
                # Update robot state from its body
                assert self.robot is not None, "Robot not initialized"
                robot_obj = state.get_objects(KinRobotType)[0]
                state.set(robot_obj, "x", self.robot.base_pose.x)
                state.set(robot_obj, "y", self.robot.base_pose.y)
                state.set(robot_obj, "theta", self.robot.base_pose.theta)
                state.set(robot_obj, "vx", self.robot.base_vel[0].x)
                state.set(robot_obj, "vy", self.robot.base_vel[0].y)
                state.set(robot_obj, "omega", self.robot.base_vel[1])
                state.set(robot_obj, "arm_joint", self.robot.curr_arm_length)
                state.set(robot_obj, "finger_gap", self.robot.curr_gripper)
            else:
                assert (
                    obj in self._state_obj_to_pymunk_body
                ), f"Object {obj.name} not found in pymunk body cache"
                pymunk_body = self._state_obj_to_pymunk_body[obj]
                # Update object state from body
                state.set(obj, "x", pymunk_body.position.x)
                state.set(obj, "y", pymunk_body.position.y)
                state.set(obj, "theta", pymunk_body.angle)
                state.set(obj, "vx", pymunk_body.velocity.x)
                state.set(obj, "vy", pymunk_body.velocity.y)
                state.set(obj, "omega", pymunk_body.angular_velocity)

        # Update the current state
        self._current_state = state

    def _target_satisfied(
        self,
        state: ObjectCentricState,
        static_object_body_cache: dict[Object, MultiBody2D],
    ) -> bool:
        """Check if the target condition is satisfied.
        """
        return False

    def _get_reward_and_done(self) -> tuple[float, bool]:
        """Calculate reward and termination."""
        assert self._current_state is not None
        terminated = self._target_satisfied(
            self._current_state,
            self._static_object_body_cache,
        )
        return -1.0, terminated


class DynPushPullHook2DEnv(ConstantObjectPRBenchEnv):
    """Dynamic Push-Pull Hook 2D env with a constant number of objects."""

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricDynPushPullHook2DEnv:
        return ObjectCentricDynPushPullHook2DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        constant_objects = ["robot", "hook"]
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
