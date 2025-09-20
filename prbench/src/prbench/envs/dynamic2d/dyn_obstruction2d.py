"""Dynamic Obstruction 2D env using PyMunk physics."""

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
from prbench.envs.utils import PURPLE, sample_se2_pose, state_2d_has_collision

# Define custom object types for the obstruction environment
TargetBlockType = Type("target_block", parent=DynRectangleType)
TargetSurfaceType = Type("target_surface", parent=KinRectangleType)
Dynamic2DRobotEnvTypeFeatures[TargetBlockType] = list(
    Dynamic2DRobotEnvTypeFeatures[DynRectangleType]
)
Dynamic2DRobotEnvTypeFeatures[TargetSurfaceType] = list(
    Dynamic2DRobotEnvTypeFeatures[KinRectangleType]
)


@dataclass(frozen=True)
class DynObstruction2DEnvConfig(Dynamic2DRobotEnvConfig):
    """Scene config for DynObstruction2DEnv()."""

    # World boundaries. Standard coordinate frame with (0, 0) in bottom left.
    world_min_x: float = 0.0
    world_max_x: float = 1 + np.sqrt(5)  # golden ratio :)
    world_min_y: float = 0.0
    world_max_y: float = 2.0

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
        SE2Pose(0.2, 0.2, -np.pi / 2),
        SE2Pose(0.8, 0.8, np.pi / 2),
    )

    # Table hyperparameters.
    table_rgb: tuple[float, float, float] = (0.75, 0.75, 0.75)
    table_height: float = 0.1
    table_width: float = world_max_x - world_min_x
    # The table pose is defined at the center
    table_pose: SE2Pose = SE2Pose(
        world_min_x + table_width / 2, world_min_y + table_height / 2, 0.0
    )

    # Target surface hyperparameters.
    target_surface_rgb: tuple[float, float, float] = PURPLE
    target_surface_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(world_min_x, table_pose.y, 0.0),
        SE2Pose(world_max_x - robot_base_radius, table_pose.y, 0.0),
    )
    target_surface_height: float = table_height
    # This adds to the width of the target block.
    target_surface_width_addition_bounds: tuple[float, float] = (
        robot_base_radius / 5,
        robot_base_radius / 2,
    )

    # Target block hyperparameters.
    target_block_rgb: tuple[float, float, float] = PURPLE
    target_block_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(
            world_min_x + robot_base_radius, table_pose.y + table_height + 1e-6, 0.0
        ),
        SE2Pose(
            world_max_x - robot_base_radius, table_pose.y + table_height + 1e-6, 0.0
        ),
    )
    target_block_height_bounds: tuple[float, float] = (
        robot_base_radius,
        2 * robot_base_radius,
    )
    target_block_width_bounds: tuple[float, float] = (
        gripper_base_height / 2,
        2 * robot_base_radius,
    )
    target_block_mass: float = 1.0

    # Obstruction hyperparameters (DYNAMIC).
    obstruction_rgb: tuple[float, float, float] = (0.75, 0.1, 0.1)
    obstruction_init_pose_bounds = (
        SE2Pose(
            world_min_x + robot_base_radius, table_pose.y + table_height + 1e-6, 0.0
        ),
        SE2Pose(
            world_max_x - robot_base_radius, table_pose.y + table_height + 1e-6, 0.0
        ),
    )
    obstruction_height_bounds: tuple[float, float] = (
        robot_base_radius,
        2 * robot_base_radius,
    )
    obstruction_width_bounds: tuple[float, float] = (
        gripper_base_height / 2,
        2 * robot_base_radius,
    )
    obstruction_block_mass: float = 1.0

    # NOTE: this is not the "real" probability, but rather, the probability
    # that we will attempt to sample the obstruction somewhere on the target
    # surface during each round of rejection sampling during reset().
    obstruction_init_on_target_prob: float = 0.9

    # For sampling initial states.
    max_initial_state_sampling_attempts: int = 10_000

    # For rendering.
    render_dpi: int = 250


class ObjectCentricDynObstruction2DEnv(
    ObjectCentricDynamic2DRobotEnv[DynObstruction2DEnvConfig]
):
    """Dynamic environment where a block must be placed on an obstructed target. Uses
    PyMunk physics simulation.

    Key difference from Geom2DEnv is that the robot can interact with dynamic objects
    with realistic physics (friction, collisions, etc). This means some objects should
    be *pushed* instead of *grasped*.
    """

    def __init__(
        self,
        num_obstructions: int = 2,
        config: DynObstruction2DEnvConfig = DynObstruction2DEnvConfig(),
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)
        self._num_obstructions = num_obstructions

        # Store object references for tracking
        self._target_block: Object | None = None
        self._target_surface: Object | None = None

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        init_state_dict: dict[Object, dict[str, float]] = {}

        # Create the table.
        table = Object("table", KinRectangleType)
        init_state_dict[table] = {
            "x": self.config.table_pose.x,
            "vx": 0.0,
            "y": self.config.table_pose.y,
            "vy": 0.0,
            "theta": self.config.table_pose.theta,
            "omega": 0.0,
            "width": self.config.table_width,
            "height": self.config.table_height,
            "static": True,
            "color_r": self.config.table_rgb[0],
            "color_g": self.config.table_rgb[1],
            "color_b": self.config.table_rgb[2],
            "z_order": ZOrder.ALL.value,
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
            target_block_pose = sample_se2_pose(
                self.config.target_block_init_pose_bounds, self.np_random
            )
            target_block_shape = (
                self.np_random.uniform(*self.config.target_block_width_bounds),
                self.np_random.uniform(*self.config.target_block_height_bounds),
            )
            target_surface_pose = sample_se2_pose(
                self.config.target_surface_init_pose_bounds, self.np_random
            )
            target_surface_width_addition = self.np_random.uniform(
                *self.config.target_surface_width_addition_bounds
            )
            target_surface_shape = (
                target_block_shape[0] + target_surface_width_addition,
                self.config.target_surface_height,
            )

            obstructions: list[tuple[SE2Pose, tuple[float, float]]] = []
            for _ in range(self._num_obstructions):
                obstruction_shape = (
                    self.np_random.uniform(*self.config.obstruction_width_bounds),
                    self.np_random.uniform(*self.config.obstruction_height_bounds),
                )
                obstruction_init_on_target = (
                    self.np_random.uniform()
                    < self.config.obstruction_init_on_target_prob
                )
                if obstruction_init_on_target:
                    old_lb, old_ub = self.config.obstruction_init_pose_bounds
                    new_x_lb = target_surface_pose.x - obstruction_shape[0]
                    new_x_ub = target_surface_pose.x + target_surface_shape[0]
                    new_lb = SE2Pose(new_x_lb, old_lb.y, old_lb.theta)
                    new_ub = SE2Pose(new_x_ub, old_ub.y, old_ub.theta)
                    pose_bounds = (new_lb, new_ub)
                else:
                    pose_bounds = self.config.obstruction_init_pose_bounds
                obstruction_pose = sample_se2_pose(pose_bounds, self.np_random)
                obstructions.append((obstruction_pose, obstruction_shape))

            state = self._create_initial_state(
                robot_pose,
                target_surface_pose,
                target_surface_shape,
                target_block_pose,
                target_block_shape,
                obstructions,
            )

            # Check initial state validity: goal not satisfied and no collisions.
            if self._target_satisfied(state, {}):
                continue
            full_state = state.copy()
            full_state.data.update(self.initial_constant_state.data)
            all_objects = set(full_state)
            # We use Geom2D collision checker for now, maybe need to update it.
            if state_2d_has_collision(full_state, all_objects, all_objects, {}):
                continue
            return state

        raise RuntimeError(f"Failed to sample initial state after {n} attempts")

    def _create_initial_state(
        self,
        robot_pose: SE2Pose,
        target_surface_pose: SE2Pose,
        target_surface_shape: tuple[float, float],
        target_block_pose: SE2Pose,
        target_block_shape: tuple[float, float],
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

        # Create the target surface.
        target_surface = Object("target_surface", TargetSurfaceType)
        init_state_dict[target_surface] = {
            "x": target_surface_pose.x,
            "vx": 0.0,
            "y": target_surface_pose.y,
            "vy": 0.0,
            "theta": target_surface_pose.theta,
            "omega": 0.0,
            "width": target_surface_shape[0],
            "height": target_surface_shape[1],
            "static": True,
            "color_r": self.config.target_surface_rgb[0],
            "color_g": self.config.target_surface_rgb[1],
            "color_b": self.config.target_surface_rgb[2],
            "z_order": ZOrder.NONE.value,
        }

        # Create the target block.
        target_block = Object("target_block", TargetBlockType)
        init_state_dict[target_block] = {
            "x": target_block_pose.x,
            "vx": 0.0,
            "y": target_block_pose.y + target_block_shape[1] / 2,
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

        # Create obstructions.
        for i, (obstruction_pose, obstruction_shape) in enumerate(obstructions):
            obstruction = Object(f"obstruction{i}", DynRectangleType)
            init_state_dict[obstruction] = {
                "x": obstruction_pose.x,
                "vx": 0.0,
                "y": obstruction_pose.y + obstruction_shape[1] / 2,
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
                    or obj.is_instance(TargetSurfaceType)
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
                else:
                    # Dynamic objects
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

        This is borrowed from geom2d obstruction env for now.
        """
        target_objects = state.get_objects(TargetBlockType)
        assert len(target_objects) == 1
        target_object = target_objects[0]
        target_surfaces = state.get_objects(TargetSurfaceType)
        assert len(target_surfaces) == 1
        target_surface = target_surfaces[0]
        return is_on(state, target_object, target_surface, static_object_body_cache)

    def _get_reward_and_done(self) -> tuple[float, bool]:
        """Calculate reward and termination."""
        # Terminate when target object is on the target surface. Give -1 reward
        # at every step until then to encourage fast completion.
        assert self._current_state is not None
        terminated = self._target_satisfied(
            self._current_state,
            self._static_object_body_cache,
        )
        return -1.0, terminated


class DynObstruction2DEnv(ConstantObjectPRBenchEnv):
    """Dynamic Obstruction 2D env with a constant number of objects."""

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricDynamic2DRobotEnv:
        return ObjectCentricDynObstruction2DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        constant_objects = ["target_surface", "target_block"]
        for obj in sorted(exemplar_state):
            if obj.name.startswith("obstruct"):
                constant_objects.append(obj.name)
            if obj.name == "robot":
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
