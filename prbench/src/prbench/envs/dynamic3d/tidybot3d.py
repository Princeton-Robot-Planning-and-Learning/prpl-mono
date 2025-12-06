"""TidyBot 3D environment wrapper for PRBench."""

import abc
import json
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
from gymnasium.spaces import Space
from numpy.typing import NDArray
from relational_structs import Array, Object, ObjectCentricState
from relational_structs.utils import create_state_from_dict

from prbench.core import ConstantObjectPRBenchEnv, FinalConfigMeta, PRBenchEnvConfig
from prbench.envs.dynamic3d.base_env import (
    ObjectCentricDynamic3DRobotEnv,
)
from prbench.envs.dynamic3d.object_types import (
    MujocoObjectTypeFeatures,
    MujocoRBY1ARobotObjectType,
    MujocoTidyBotRobotObjectType,
)
from prbench.envs.dynamic3d.objects import (
    MujocoFixture,
    MujocoObject,
    get_fixture_class,
    get_object_class,
)
from prbench.envs.dynamic3d.placement_samplers import (
    sample_collision_free_positions,
    sample_pose_in_region,
)
from prbench.envs.dynamic3d.robots import (
    RBY1ARobotActionSpace,
    RBY1ARobotEnv,
    TidyBot3DRobotActionSpace,
    TidyBotRobotEnv,
)
from prbench.envs.dynamic3d.tidybot_rewards import create_reward_calculator
from prbench.envs.dynamic3d.utils import check_in_region


@dataclass(frozen=True)
class TidyBot3DConfig(PRBenchEnvConfig, metaclass=FinalConfigMeta):
    """Configuration for TidyBot3D environment."""

    control_frequency: int = 10
    horizon: int = 1000
    camera_names: list[str] = field(default_factory=lambda: ["overview"])
    camera_width: int = 640
    camera_height: int = 480
    show_viewer: bool = False
    act_delta: bool = True


class ObjectCentricRobotEnv(ObjectCentricDynamic3DRobotEnv[TidyBot3DConfig]):
    """TidyBot 3D environment with mobile manipulation tasks."""

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        config: TidyBot3DConfig = TidyBot3DConfig(),
        seed: int | None = None,
        scene_type: str = "ground",
        num_objects: int = 3,
        task_config_path: str | None = None,
        render_images: bool = False,
        show_images: bool = False,
    ) -> None:
        # Initialize ObjectCentricPRBenchEnv first
        super().__init__(config)

        # Store instance attributes from kwargs
        self.scene_type = scene_type
        self.num_objects = num_objects
        self.render_images = render_images
        self.camera_names = config.camera_names
        self.show_images = show_images
        self.seed = seed
        self.config = config

        # Parse task configuration
        if task_config_path is None:
            # Default task config based on scene_type and num_objects
            task_config_path = (
                f"./tasks/tidybot-{self.scene_type}-o{self.num_objects}.json"
            )
        if not os.path.isabs(task_config_path):
            task_config_path = str(Path(__file__).parent / task_config_path)
        assert os.path.exists(
            task_config_path
        ), f"task_config_path {task_config_path} does not exist."
        with open(task_config_path, "r", encoding="utf-8") as f:
            self.task_config = json.load(f)

        # Initialize robot environment
        robot_cls = {"tidybot": TidyBotRobotEnv, "rby1a": RBY1ARobotEnv}[
            self.task_config["robots"][0]
        ]
        self._robot_env = robot_cls(
            control_frequency=self.config.control_frequency,
            act_delta=self.config.act_delta,
            horizon=self.config.horizon,
            camera_names=self.camera_names,
            camera_width=self.config.camera_width,
            camera_height=self.config.camera_height,
            seed=seed if seed is not None else self.seed,
            render_images=self.render_images,
            show_viewer=self.config.show_viewer,
        )

        self._render_camera_name: str | None = "overview"

        # Cannot show images if not rendering images
        if show_images and not render_images:
            raise ValueError("Cannot show images if render_images is False")

        # Initialize empty object list
        self._objects: list[MujocoObject] = []
        self._objects_dict: dict[str, MujocoObject] = {}
        self._fixtures_dict: dict[str, MujocoFixture] = {}

        self._reward_calculator = create_reward_calculator(scene_type, num_objects)

        # Store current state
        self._current_state: ObjectCentricState | None = None

    def _vectorize_observation(self, obs: dict[str, Any]) -> NDArray[np.float32]:
        """Convert TidyBot observation dict to vector."""
        obs_vector: list[float] = []
        for key in sorted(obs.keys()):  # Sort for consistency
            value = obs[key]
            obs_vector.extend(value.flatten())
        return np.array(obs_vector, dtype=np.float32)

    def _create_scene_xml(self) -> str:
        """Create the MuJoCo XML string for the current scene configuration."""

        # Set model path to local models directory
        model_base_path = Path(__file__).parent / "models" / "stanford_tidybot"
        model_file = "ground_scene.xml"
        # Construct absolute path to model file
        absolute_model_path = model_base_path / model_file

        with open(absolute_model_path, "r", encoding="utf-8") as f:
            xml_string = f.read()

        # Insert objects in scene
        tree = ET.parse(str(absolute_model_path))
        root = tree.getroot()
        worldbody = root.find("worldbody")
        if worldbody is not None:
            # Remove all existing cube bodies
            for body in list(worldbody):
                if body.tag == "body" and body.attrib.get("name", "").startswith(
                    "cube"
                ):
                    worldbody.remove(body)

            # Add tables and objects based on task configuration
            if self.task_config is not None:
                # Find and remove the existing table body
                for body in list(worldbody):
                    if body.tag == "body" and body.attrib.get("name") == "table":
                        worldbody.remove(body)
                        break

                all_fixtures = self.task_config.get("fixtures", {})
                fixtures: dict[str, dict[str, dict[str, Any]]] = {}

                # Create fixture ranges dict based on initial state predicates
                fixture_ranges = {}
                init_predicates = self.task_config.get("initial_state", [])
                for pred in init_predicates:
                    if pred[0] == "on" and len(pred) == 3:
                        fixture_name = pred[1]
                        region_name = pred[2]

                        # Check if this fixture exists in any fixture type and add to
                        # fixtures dict
                        fixture_found = False
                        for fixture_type, fixture_configs in all_fixtures.items():
                            if fixture_name in fixture_configs:
                                fixture_found = True
                                # Add this fixture type and fixture to filtered
                                # fixtures dict
                                if fixture_type not in fixtures:
                                    fixtures[fixture_type] = {}
                                fixtures[fixture_type][fixture_name] = fixture_configs[
                                    fixture_name
                                ]
                                break

                        if fixture_found:
                            region_config = self.task_config["regions"][region_name]
                            # Assert that the region target is ground
                            assert region_config["target"] == "ground", (
                                f"Region {region_name} for fixture {fixture_name} "
                                f"must have target 'ground', got "
                                f"'{region_config['target']}'"
                            )
                            # Extract randomly sampled range tuple
                            # (x_min, y_min, x_max, y_max)
                            available_ranges = region_config["ranges"]
                            selected_range = self.np_random.choice(
                                len(available_ranges)
                            )
                            ranges = available_ranges[selected_range]
                            fixture_ranges[fixture_name] = tuple(ranges)

                # Sample collision-free positions for all fixtures
                fixture_poses = sample_collision_free_positions(
                    fixtures, self.np_random, fixture_ranges
                )

                # Insert filtered fixtures
                for fixture_type, fixture_configs in fixtures.items():

                    for fixture_name, fixture_config in fixture_configs.items():
                        # Sample collision-free position for the fixture
                        fixture_pose = fixture_poses[fixture_type][fixture_name]
                        fixture_pos = fixture_pose["position"]
                        fixture_yaw = fixture_pose["yaw"]

                        # Find regions for this fixture if specified
                        regions_in_fixture = {}
                        all_regions = self.task_config.get("regions", {})
                        for region_name, region_config in all_regions.items():
                            if region_config["target"] == fixture_name:
                                regions_in_fixture[region_name] = region_config

                        # Create new fixture with configuration dictionary
                        fixture_cls = get_fixture_class(fixture_type)
                        new_fixture = fixture_cls(
                            name=fixture_name,
                            fixture_config=fixture_config,
                            position=fixture_pos,
                            yaw=fixture_yaw,
                            regions=regions_in_fixture,
                        )
                        new_fixture.visualize_regions()
                        self._fixtures_dict[fixture_name] = new_fixture
                        fixture_body = new_fixture.xml_element
                        worldbody.append(fixture_body)

                # Insert all objects
                objects = self.task_config.get("objects", {})
                for object_type, object_configs in objects.items():
                    for object_name, object_config in object_configs.items():
                        obj_cls = get_object_class(object_type)
                        obj_options = {
                            "size": object_config["size"],
                            "rgba": " ".join(map(str, object_config["rgba"])),
                            "mass": object_config["mass"],
                        }
                        obj = obj_cls(
                            name=object_name,
                            env=self._robot_env,
                            options=obj_options,
                        )
                        body = obj.xml_element
                        worldbody.append(body)
                        self._objects.append(obj)
                        self._objects_dict[object_name] = obj

            # Get XML string from tree
            xml_string = ET.tostring(root, encoding="unicode")

        return xml_string

    def _initialize_object_poses(self) -> None:
        """Initialize object poses in the environment."""

        assert self._robot_env is not None, "Robot environment not initialized"
        assert self._robot_env.sim is not None, "Simulation not initialized"

        # Set object pose based on task configuration
        init_predicates = self.task_config.get("initial_state", [])
        for pred in init_predicates:
            if pred[0] == "on":
                obj_name = pred[1]
                if obj_name in self._fixtures_dict:
                    continue  # Skip fixtures, they are static
                if obj_name not in self._objects_dict:
                    raise ValueError(f"Object {obj_name} not found in environment.")
                region_name = pred[2]
                region_config = self.task_config["regions"][region_name]
                region_ranges = region_config["ranges"]

                if region_config["target"] == "ground":
                    # Sample pose directly on the ground using utility function
                    assert obj_name.startswith("cube"), "TODO"
                    size = self.task_config["objects"]["cube"][obj_name]["size"]
                    pos_x, pos_y, pos_z = sample_pose_in_region(
                        region_ranges,
                        self.np_random,
                        z_coordinate=size,
                    )
                else:
                    # Sample pose on a fixture (table, etc.)
                    target_fixture = region_config["target"]
                    assert target_fixture in self._fixtures_dict, (
                        f"Fixture {target_fixture} not found in environment. "
                        f"Did you provide an initialization predicate for the "
                        f"fixture?"
                    )
                    fixture = self._fixtures_dict[target_fixture]
                    pos_x, pos_y, pos_z = fixture.sample_pose_in_region(
                        region_name, self.np_random
                    )

                # Randomize orientation around Z-axis (yaw)
                theta = self.np_random.uniform(-math.pi, math.pi)
                quat = np.array([math.cos(theta / 2), 0, 0, math.sin(theta / 2)])

                # Set object pose in the environment
                obj = self._objects_dict[obj_name]
                obj.set_pose(np.array([pos_x, pos_y, pos_z]), quat)

        self._robot_env.sim.forward()

    @abc.abstractmethod
    def _create_action_space(  # type: ignore
        self, config: TidyBot3DConfig
    ) -> Space[Array]:
        """Create action space for TidyBot's control interface."""

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObjectCentricState, dict[str, Any]]:
        """Reset the environment and return object-centric observation."""

        # Reset the random seed
        self._robot_env.seed(seed=seed)
        self.np_random = self._robot_env.np_random

        # Create scene XML
        self._objects = []
        self._objects_dict = {}
        self._fixtures_dict = {}
        xml_string = self._create_scene_xml()

        # Reset the underlying TidyBot robot environment
        robot_options = options.copy() if options is not None else {}
        robot_options["xml"] = xml_string
        self._robot_env.reset(options=robot_options)

        # Initialize object poses
        self._initialize_object_poses()

        # Get object-centric observation
        self._current_state = self._get_object_centric_state()

        return self._get_current_state(), {}

    def set_state(self, state: ObjectCentricState) -> None:
        """Set the environment to the current state.

        This is useful for planning baselines.
        """
        # Reset the robot.
        self._set_robot_state(state)

        # Reset the objects.
        for mujoco_object in self._objects:
            obj = state.get_object_from_name(mujoco_object.name)
            position = [state.get(obj, "x"), state.get(obj, "y"), state.get(obj, "z")]
            orientation = [
                state.get(obj, "qw"),
                state.get(obj, "qx"),
                state.get(obj, "qy"),
                state.get(obj, "qz"),
            ]
            mujoco_object.set_pose(position, orientation)
            linear_velocity = [
                state.get(obj, "vx"),
                state.get(obj, "vy"),
                state.get(obj, "vz"),
            ]
            angular_velocity = [
                state.get(obj, "wx"),
                state.get(obj, "wy"),
                state.get(obj, "wz"),
            ]
            mujoco_object.set_velocity(linear_velocity, angular_velocity)
        # NOTE: Fixtures are static (without joints), so we cannot set their state.

        assert self._robot_env is not None, "Robot environment not initialized"
        assert self._robot_env.sim is not None, "Simulation not initialized"
        self._robot_env.sim.forward()

        # Update the cached current state
        self._current_state = self._get_object_centric_state()

    def _visualize_image_in_window(
        self, image: NDArray[np.uint8], window_name: str
    ) -> None:
        """Visualize an image in an OpenCV window."""
        if image.dtype == np.uint8 and len(image.shape) == 3:
            # Convert RGB to BGR for proper color display in OpenCV
            display_image = cv.cvtColor(  # pylint: disable=no-member
                image, cv.COLOR_RGB2BGR  # pylint: disable=no-member
            )
            cv.imshow(window_name, display_image)  # pylint: disable=no-member
            cv.waitKey(1)  # pylint: disable=no-member

    def _get_current_state(self) -> ObjectCentricState:
        """Get the current object-centric observation."""
        assert self._current_state is not None, "Need to call reset() first"
        return self._current_state.copy()

    def _get_obs(self) -> dict[str, Any]:
        """Get the current raw observation (for compatibility with reward functions)."""
        assert self._robot_env is not None, "Robot environment not initialized"
        obs = self._robot_env.get_obs()
        vec_obs = self._vectorize_observation(obs)
        object_centric_state = self._get_object_centric_state()
        return {"vec": vec_obs, "object_centric_state": object_centric_state}

    def _get_object_centric_state(self) -> ObjectCentricState:
        """Get the current object-centric state of the environment."""
        # Collect object-centric data for all objects
        state_dict = {}
        for obj in self._objects:
            obj_data = obj.get_object_centric_data()
            state_dict[obj.symbolic_object] = obj_data
        for fixture in self._fixtures_dict.values():
            fixture_data = fixture.get_object_centric_data()
            state_dict[fixture.symbolic_object] = fixture_data
        # Add robot into object-centric state.
        robot_state_dict = self._get_object_centric_robot_data()
        state_dict.update(robot_state_dict)
        return create_state_from_dict(state_dict, MujocoObjectTypeFeatures)

    def step(
        self, action: Array
    ) -> tuple[ObjectCentricState, float, bool, bool, dict[str, Any]]:
        """Step the environment and return object-centric observation."""
        # Run the action through the underlying environment
        assert self._robot_env is not None, "Robot environment not initialized"
        self._robot_env.step(action)

        # Update object-centric state
        self._current_state = self._get_object_centric_state()

        # Get raw observation for reward calculation
        raw_obs = self._get_obs()

        # Visualization loop for rendered image
        if self.show_images:
            for camera_name in self._robot_env.camera_names:
                self._visualize_image_in_window(
                    raw_obs[f"{camera_name}_image"],
                    f"TidyBot {camera_name} camera",
                )

        # Calculate reward and termination
        reward = self.reward(raw_obs)
        terminated = self._is_terminated(raw_obs)
        truncated = False

        return self._get_current_state(), reward, terminated, truncated, {}

    def _check_goals(self) -> bool:
        """Check if the goal has been achieved."""
        state = self._get_current_state()
        goal_predicates = self.task_config.get("goal_state", [])
        successes = []
        for pred in goal_predicates:
            if pred[0] == "on":
                obj_name = pred[1]
                region_name = pred[2]
                obj = state.get_object_from_name(obj_name)
                position = np.array(
                    [
                        state.get(obj, "x"),
                        state.get(obj, "y"),
                        state.get(obj, "z"),
                    ],
                    dtype=np.float32,
                )
                region_config = self.task_config["regions"][region_name]

                if region_config["target"] == "ground":
                    # Check pose directly on the ground in the world frame
                    region_ranges = region_config["ranges"]
                    in_region = check_in_region(position, region_ranges)
                else:
                    # Sample pose on a fixture (table, etc.)
                    fixture = self._fixtures_dict[region_config["target"]]
                    in_region = fixture.check_in_region(position, region_name)

                successes.append(in_region)
            else:
                raise NotImplementedError(
                    f"Goal predicate {pred[0]} not implemented in _check_goals"
                )
        return all(successes)

    def reward(self, obs: dict[str, Any]) -> float:
        """Calculate reward based on task completion."""
        return self._reward_calculator.calculate_reward(obs)

    def _is_terminated(self, obs: dict[str, Any]) -> bool:
        """Check if episode should terminate."""
        return self._reward_calculator.is_terminated(obs)

    def render(self) -> NDArray[np.uint8]:  # type: ignore
        """Render the environment."""
        if self.render_mode == "rgb_array":
            assert self._robot_env is not None, "Robot environment not initialized"
            images = self._robot_env.get_camera_images()
            if images is not None:
                if self._render_camera_name and self._render_camera_name in images:
                    return images[self._render_camera_name]
                # Otherwise, return the first available image.
                for _, value in images.items():
                    return value
            raise RuntimeError("No camera image available in observation.")
        raise NotImplementedError(f"Render mode {self.render_mode} not supported")

    def close(self) -> None:
        """Close the environment."""
        if self.show_images:
            # Close OpenCV windows
            cv.destroyAllWindows()  # pylint: disable=no-member
        if self._robot_env is not None:
            self._robot_env.close()

    def set_render_camera(self, camera_name: str | None) -> None:
        """Set the camera to use for rendering."""
        self._render_camera_name = camera_name

    @abc.abstractmethod
    def _get_object_centric_robot_data(self) -> dict[Object, dict[str, float]]:
        """Get object-centric data for the robot.

        This method should be implemented by subclasses to provide robot-specific state
        data.
        """

    @abc.abstractmethod
    def _set_robot_state(self, state: ObjectCentricState) -> None:
        """Set the robot state in the simulation.

        This method should be implemented by subclasses to set the robot's state in the
        simulation.
        """


class ObjectCentricTidyBot3DEnv(ObjectCentricRobotEnv):
    """TidyBot-specific implementation of object-centric robot environment."""

    def _create_action_space(  # type: ignore
        self, config: TidyBot3DConfig
    ) -> Space[Array]:
        """Create action space for TidyBot's control interface."""
        return TidyBot3DRobotActionSpace()

    def _get_object_centric_robot_data(self) -> dict[Object, dict[str, float]]:
        assert self.task_config["robots"][0] == "tidybot"
        assert self._robot_env is not None, "Robot environment not initialized"
        robot = Object("robot", MujocoTidyBotRobotObjectType)
        # Build this super explicitly, even though verbose, to be careful.
        assert self._robot_env.qpos is not None
        assert self._robot_env.qvel is not None
        state_dict = {}
        state_dict[robot] = {
            "pos_base_x": self._robot_env.qpos["base"][0],
            "pos_base_y": self._robot_env.qpos["base"][1],
            "pos_base_rot": self._robot_env.qpos["base"][2],
            "pos_arm_joint1": self._robot_env.qpos["arm"][0],
            "pos_arm_joint2": self._robot_env.qpos["arm"][1],
            "pos_arm_joint3": self._robot_env.qpos["arm"][2],
            "pos_arm_joint4": self._robot_env.qpos["arm"][3],
            "pos_arm_joint5": self._robot_env.qpos["arm"][4],
            "pos_arm_joint6": self._robot_env.qpos["arm"][5],
            "pos_arm_joint7": self._robot_env.qpos["arm"][6],
            "pos_gripper": self._robot_env.ctrl["gripper"][0] / 255.0,
            "vel_base_x": self._robot_env.qvel["base"][0],
            "vel_base_y": self._robot_env.qvel["base"][1],
            "vel_base_rot": self._robot_env.qvel["base"][2],
            "vel_arm_joint1": self._robot_env.qvel["arm"][0],
            "vel_arm_joint2": self._robot_env.qvel["arm"][1],
            "vel_arm_joint3": self._robot_env.qvel["arm"][2],
            "vel_arm_joint4": self._robot_env.qvel["arm"][3],
            "vel_arm_joint5": self._robot_env.qvel["arm"][4],
            "vel_arm_joint6": self._robot_env.qvel["arm"][5],
            "vel_arm_joint7": self._robot_env.qvel["arm"][6],
            "vel_gripper": self._robot_env.qvel["gripper"][0],
        }
        return state_dict

    def _set_robot_state(self, state: ObjectCentricState) -> None:
        """Set the robot state in the simulation."""
        assert self._robot_env is not None, "Robot environment not initialized"

        robot_obj = state.get_object_from_name("robot")

        # Reset the robot base position.
        robot_base_pos = [
            state.get(robot_obj, "pos_base_x"),
            state.get(robot_obj, "pos_base_y"),
            state.get(robot_obj, "pos_base_rot"),
        ]
        assert self._robot_env.qpos is not None
        self._robot_env.qpos["base"][:] = robot_base_pos

        # Reset the robot arm position.
        robot_arm_pos = [state.get(robot_obj, f"pos_arm_joint{i}") for i in range(1, 8)]
        assert self._robot_env.qpos is not None
        self._robot_env.qpos["arm"][:] = robot_arm_pos

        # Reset the robot gripper position.
        gripper_pos = state.get(robot_obj, "pos_gripper")
        assert self._robot_env.ctrl is not None
        self._robot_env.ctrl["gripper"][:] = gripper_pos * 255.0

        # Reset the robot base velocity.
        robot_base_vel = [
            state.get(robot_obj, "vel_base_x"),
            state.get(robot_obj, "vel_base_y"),
            state.get(robot_obj, "vel_base_rot"),
        ]
        assert self._robot_env.qvel is not None
        self._robot_env.qvel["base"][:] = robot_base_vel

        # Reset the robot arm velocity.
        robot_arm_vel = [state.get(robot_obj, f"vel_arm_joint{i}") for i in range(1, 8)]
        assert self._robot_env.qvel is not None
        self._robot_env.qvel["arm"][:] = robot_arm_vel

        # Reset the robot gripper velocity.
        gripper_vel = state.get(robot_obj, "vel_gripper")
        assert self._robot_env.qvel is not None
        self._robot_env.qvel["gripper"][:] = gripper_vel


class TidyBot3DEnv(ConstantObjectPRBenchEnv):
    """TidyBot env with a constant number of objects."""

    def _create_object_centric_env(self, *args, **kwargs) -> ObjectCentricTidyBot3DEnv:
        return ObjectCentricTidyBot3DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        return [o.name for o in sorted(exemplar_state)]

    def _create_env_markdown_description(self) -> str:
        """Create environment description (policy-agnostic)."""
        scene_description = ""
        env = self._object_centric_env
        assert isinstance(env, ObjectCentricTidyBot3DEnv)
        if env.scene_type == "ground":
            scene_description = (
                " In the 'ground' scene, objects are placed randomly on a flat "
                "ground plane."
            )

        return f"""A 3D mobile manipulation environment using the TidyBot platform.

The robot has a holonomic mobile base with powered casters and a Kinova Gen3 arm.
Scene type: {env.scene_type} with {env.num_objects} objects.{scene_description}

The robot can control:
- Base pose (x, y, theta)
- Arm position (x, y, z)
- Arm orientation (quaternion)
- Gripper position (open/close)
"""

    def _create_obs_markdown_description(self) -> str:
        """Create observation space description."""
        return """Observation includes:
- Robot state: base pose, arm position/orientation, gripper state
- Object states: positions and orientations of all objects
- Camera images: RGB images from base and wrist cameras
- Scene-specific features: handle positions for cabinets/drawers
"""

    def _create_action_markdown_description(self) -> str:
        """Create action space description."""
        return """Actions control:
- base_pose: [x, y, theta] - Mobile base position and orientation
- arm_pos: [x, y, z] - End effector position in world coordinates
- arm_quat: [x, y, z, w] - End effector orientation as quaternion
- gripper_pos: [pos] - Gripper open/close position (0=closed, 1=open)
"""

    def _create_reward_markdown_description(self) -> str:
        """Create reward description."""
        env = self._object_centric_env
        assert isinstance(env, ObjectCentricRobotEnv)
        if env.scene_type == "ground":
            return (
                "The primary reward is for successfully placing objects at their "
                "target locations.\n"
                "- A reward of +1.0 is given for each object placed within a 5cm "
                "tolerance of its target.\n"
                "- A smaller positive reward is given for objects within a 10cm "
                "tolerance to guide the robot.\n"
                "- A small negative reward (-0.01) is applied at each timestep to "
                "encourage efficiency.\n"
                "The episode terminates when all objects are placed at their "
                "respective targets.\n"
            )
        return """Reward function depends on the specific task:
- Object stacking: Reward for successfully stacking objects
- Drawer/cabinet tasks: Reward for opening/closing and placing objects
- General manipulation: Reward for successful pick-and-place operations

Currently returns a small negative reward (-0.01) per timestep to encourage exploration.
"""

    def _create_references_markdown_description(self) -> str:
        """Create references description."""
        return """TidyBot++: An Open-Source Holonomic Mobile Manipulator
for Robot Learning
- Jimmy Wu, William Chong, Robert Holmberg, Aaditya Prasad, Yihuai Gao,
  Oussama Khatib, Shuran Song, Szymon Rusinkiewicz, Jeannette Bohg
- Conference on Robot Learning (CoRL), 2024

https://github.com/tidybot2/tidybot2
"""


class ObjectCentricRBY1A3DEnv(ObjectCentricRobotEnv):
    """RBY1A-specific implementation of object-centric robot environment."""

    def _create_action_space(  # type: ignore
        self, config: TidyBot3DConfig
    ) -> Space[Array]:
        """Create action space for TidyBot's control interface."""
        return RBY1ARobotActionSpace()

    def _get_object_centric_robot_data(self) -> dict[Object, dict[str, float]]:
        assert self.task_config["robots"][0] == "rby1a"
        assert self._robot_env is not None, "Robot environment not initialized"
        robot = Object("robot", MujocoRBY1ARobotObjectType)
        # Build this super explicitly, even though verbose, to be careful.
        state_dict = {}
        assert self._robot_env.qpos is not None
        state_dict[robot] = {
            "pos_base_right": self._robot_env.qpos["base"][0],
            "pos_base_left": self._robot_env.qpos["base"][1],
            # TODO add more attributes  # pylint: disable=fixme
        }
        return state_dict

    def _set_robot_state(self, state: ObjectCentricState) -> None:
        """Set the robot state in the simulation."""
        assert self._robot_env is not None, "Robot environment not initialized"

        robot_obj = state.get_object_from_name("robot")

        # Reset the robot base position.
        assert self._robot_env.qpos is not None
        robot_base_pos = [
            state.get(robot_obj, "pos_base_right"),
            state.get(robot_obj, "pos_base_left"),
        ]
        self._robot_env.qpos["base"][:] = robot_base_pos

        # TODO add more attributes  # pylint: disable=fixme


class RBY1A3DEnv(ConstantObjectPRBenchEnv):
    """RBY1A env with a constant number of objects."""

    def _create_object_centric_env(self, *args, **kwargs) -> ObjectCentricRBY1A3DEnv:
        return ObjectCentricRBY1A3DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        return [o.name for o in sorted(exemplar_state)]

    def _create_env_markdown_description(self) -> str:
        """Create environment description (policy-agnostic)."""
        scene_description = ""
        env = self._object_centric_env
        assert isinstance(env, ObjectCentricRBY1A3DEnv)
        if env.scene_type == "ground":
            scene_description = (
                " In the 'ground' scene, objects are placed randomly on a flat "
                "ground plane."
            )

        return f"""A 3D mobile manipulation environment using the RBY1A platform.

The robot has a holonomic mobile base with powered casters and a Kinova Gen3 arm.
Scene type: {env.scene_type} with {env.num_objects} objects.{scene_description}

The robot can control:
- Base pose (x, y, theta)
- Arm position (x, y, z)
- Arm orientation (quaternion)
- Gripper position (open/close)
"""

    def _create_obs_markdown_description(self) -> str:
        """Create observation space description."""
        return """Observation includes:
- Robot state: base pose, arm position/orientation, gripper state
- Object states: positions and orientations of all objects
- Camera images: RGB images from base and wrist cameras
- Scene-specific features: handle positions for cabinets/drawers
"""

    def _create_action_markdown_description(self) -> str:
        """Create action space description."""
        return """Actions control:
- base_pose: [x, y, theta] - Mobile base position and orientation
- arm_pos: [x, y, z] - End effector position in world coordinates
- arm_quat: [x, y, z, w] - End effector orientation as quaternion
- gripper_pos: [pos] - Gripper open/close position (0=closed, 1=open)
"""

    def _create_reward_markdown_description(self) -> str:
        """Create reward description."""
        env = self._object_centric_env
        assert isinstance(env, ObjectCentricRobotEnv)
        if env.scene_type == "ground":
            return (
                "The primary reward is for successfully placing objects at their "
                "target locations.\n"
                "- A reward of +1.0 is given for each object placed within a 5cm "
                "tolerance of its target.\n"
                "- A smaller positive reward is given for objects within a 10cm "
                "tolerance to guide the robot.\n"
                "- A small negative reward (-0.01) is applied at each timestep to "
                "encourage efficiency.\n"
                "The episode terminates when all objects are placed at their "
                "respective targets.\n"
            )
        return """Reward function depends on the specific task:
- Object stacking: Reward for successfully stacking objects
- Drawer/cabinet tasks: Reward for opening/closing and placing objects
- General manipulation: Reward for successful pick-and-place operations

Currently returns a small negative reward (-0.01) per timestep to encourage exploration.
"""

    def _create_references_markdown_description(self) -> str:
        """Create references description."""
        return """TODO
"""
