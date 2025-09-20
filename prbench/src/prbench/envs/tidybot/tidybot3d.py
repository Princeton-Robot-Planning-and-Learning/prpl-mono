"""TidyBot 3D environment wrapper for PRBench."""

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
from gymnasium.spaces import Space
from numpy.typing import NDArray
from prpl_utils.spaces import FunctionalSpace

from prbench.envs.tidybot.mujoco_utils import MjAct, MjObs
from prbench.envs.tidybot.tidybot_rewards import create_reward_calculator
from prbench.envs.tidybot.tidybot_robot_env import TidyBotRobotEnv


class TidyBot3DEnv(TidyBotRobotEnv):
    """TidyBot 3D environment with mobile manipulation tasks."""

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        control_frequency: int = 20,
        horizon: int = 1000,
        camera_names: list[str] | None = None,
        camera_width: int = 640,
        camera_height: int = 480,
        scene_type: str = "ground",
        num_objects: int = 3,
        render_mode: str | None = None,
        custom_grasp: bool = False,
        render_images: bool = True,
        seed: int | None = None,
        show_viewer: bool = False,
        show_images: bool = False,
    ) -> None:
        super().__init__(
            control_frequency,
            horizon=horizon,
            camera_names=camera_names,
            camera_width=camera_width,
            camera_height=camera_height,
            seed=seed,
            show_viewer=show_viewer,
        )

        self.scene_type = scene_type
        self.num_objects = num_objects
        self.render_mode = render_mode
        self.custom_grasp = custom_grasp
        self.render_images = render_images
        self.show_images = show_images
        self._render_camera_name: str | None = "overview"

        # Cannot show images if not rendering images
        if self.show_images:
            if not self.render_images:
                raise ValueError("Cannot show images if render_images is False")

        # Initialize empty object list
        self._object_names: list[str] = []

        self._reward_calculator = create_reward_calculator(
            self.scene_type, self.num_objects
        )

        # Define observation and action spaces
        self.observation_space = self._create_observation_space()
        self.action_space = self._create_action_space()

        # Add metadata for documentation
        self.metadata.update(
            {
                "description": self._create_env_markdown_description(),
                "observation_space_description": (
                    self._create_obs_markdown_description()
                ),
                "action_space_description": (
                    self._create_action_markdown_description()
                ),
                "reward_description": self._create_reward_markdown_description(),
                "references": self._create_references_markdown_description(),
                "render_fps": 20,
            }
        )

    def _create_observation_space(self) -> Space[MjObs]:
        """Create observation space based on TidyBot's observation structure."""
        # NOTE: this will be refactored soon after we introduce object-centric structs.
        return FunctionalSpace(contains_fn=lambda _: True)

    def _create_action_space(self) -> Space[MjAct]:
        """Create action space for TidyBot's control interface."""
        # TidyBot actions: base_pose (3), arm_pos (3), arm_quat (4), gripper_pos (1)
        low = np.array(
            [-1.0, -1.0, -np.pi, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0]
        )
        high = np.array([1.0, 1.0, np.pi, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        def _contains_fn(x: Any) -> bool:
            return isinstance(x, MjAct)

        def _sample_fn(rng: np.random.Generator) -> MjAct:
            ctrl = rng.uniform(low, high)
            return MjAct(position_ctrl=ctrl)

        return FunctionalSpace(contains_fn=_contains_fn, sample_fn=_sample_fn)

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
        if self.scene_type == "cupboard":
            model_file = "cupboard_scene.xml"
        elif self.scene_type == "table":
            model_file = "table_scene.xml"
        else:
            model_file = "ground_scene.xml"
        # Construct absolute path to model file
        absolute_model_path = model_base_path / model_file

        # --- Dynamic object insertion logic ---
        needs_dynamic_objects = self.scene_type in ["ground", "table"]
        if needs_dynamic_objects:
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
                # Insert new cubes
                for i in range(self.num_objects):
                    name = f"cube{i+1}"
                    body = ET.Element("body")
                    ET.SubElement(body, "freejoint", name=f"{name}_joint")
                    ET.SubElement(
                        body,
                        "geom",
                        type="box",
                        size="0.02 0.02 0.02",
                        rgba=".5 .7 .5 1",
                        mass="0.1",
                    )
                    worldbody.append(body)
                    self._object_names.append(name)

                # Get XML string from tree
                xml_string = ET.tostring(root, encoding="unicode")
            else:
                with open(absolute_model_path, "r", encoding="utf-8") as f:
                    xml_string = f.read()
        else:
            with open(absolute_model_path, "r", encoding="utf-8") as f:
                xml_string = f.read()

        return xml_string

    def _set_object_pos_quat(
        self, name: str, pos: NDArray[np.float32], quat: NDArray[np.float32]
    ) -> None:
        """Set object position and orientation in the environment."""

        assert self.sim is not None, "Simulation not initialized"
        joint_id = self.sim.model.get_joint_qpos_addr(f"{name}_joint")
        self.sim.data.qpos[joint_id : joint_id + 7] = np.array(
            [float(x) for x in pos] + [float(q) for q in quat]
        )

    def get_object_pos_quat(self, name: str) -> tuple[float, float]:
        """Set object position and orientation in the environment."""

        assert self.sim is not None, "Simulation not initialized"
        joint_id = self.sim.model.get_joint_qpos_addr(f"{name}_joint")
        pos = self.sim.data.qpos[joint_id : joint_id + 3]
        quat = self.sim.data.qpos[joint_id + 3 : joint_id + 7]
        return pos, quat

    def _initialize_object_poses(self) -> None:
        """Initialize object poses in the environment."""

        assert self.sim is not None, "Simulation not initialized"

        for name in self._object_names:
            pos = np.array([0.0, 0.0, 0.0])
            if self.scene_type == "cupboard":
                pass  # no position randomization for cupboard scene
            elif self.scene_type == "table":
                # Randomize position within a reasonable range
                # for the table environment
                x = round(self.np_random.uniform(0.2, 0.8), 3)
                y = round(self.np_random.uniform(-0.15, 0.15), 3)
                z = 0.44
                pos = np.array([x, y, z])
            else:
                # Randomize position within a reasonable range
                # for the ground environment
                x = round(self.np_random.uniform(0.4, 0.8), 3)
                y = round(self.np_random.uniform(-0.3, 0.3), 3)
                z = 0.02
                pos = np.array([x, y, z])
            # Randomize orientation around Z-axis (yaw)
            theta = self.np_random.uniform(-math.pi, math.pi)
            quat = np.array([math.cos(theta / 2), 0, 0, math.sin(theta / 2)])

            # Set object pose in the environment
            self._set_object_pos_quat(name, pos, quat)

        self.sim.forward()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[MjObs, dict[str, Any]]:
        # Create scene XML
        self._object_names = []
        xml_string = self._create_scene_xml()

        # Reset the underlying TidyBot robot environment
        robot_options = options.copy() if options is not None else {}
        robot_options["xml"] = xml_string
        super().reset(seed=seed, options=robot_options)

        # Initialize object poses
        self._initialize_object_poses()

        # Get observation and vectorize
        obs = super().get_obs()

        vec_obs = self._vectorize_observation(obs)

        # NOTE: this will be refactored soon after we introduce object-centric structs.
        final_obs = {"vec": vec_obs}

        return final_obs, {}

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

    def step(self, action: MjAct) -> tuple[MjObs, float, bool, bool, dict[str, Any]]:
        # Run the action.
        super().step(action)

        # Get observation
        obs = self.get_obs()
        vec_obs = self._vectorize_observation(obs)

        # Visualization loop for rendered image
        if self.show_images:
            for camera_name in self.camera_names:
                self._visualize_image_in_window(
                    obs[f"{camera_name}_image"],
                    f"TidyBot {camera_name} camera",
                )

        # Calculate reward and termination
        reward = self.reward(obs)
        terminated = self._is_terminated(obs)
        truncated = False

        # NOTE: this will be refactored soon after we introduce object-centric structs.
        final_obs = {"vec": vec_obs}

        return final_obs, reward, terminated, truncated, {}

    def reward(self, obs: MjObs) -> float:
        """Calculate reward based on task completion."""
        return self._reward_calculator.calculate_reward(obs)

    def _is_terminated(self, obs: MjObs) -> bool:
        """Check if episode should terminate."""
        return self._reward_calculator.is_terminated(obs)

    def render(self) -> Any:
        """Render the environment."""
        if self.render_mode == "rgb_array":
            obs = super().get_obs()
            # If a specific camera is requested, use it.
            if self._render_camera_name:
                key = f"{self._render_camera_name}_image"
                if key in obs:
                    return obs[key]
            # Otherwise, fall back to the first available image.
            for key, value in obs.items():
                if key.endswith("_image"):
                    return value
            raise RuntimeError("No camera image available in observation.")
        return None

    def close(self) -> None:
        """Close the environment."""
        if self.show_images:
            # Close OpenCV windows
            cv.destroyAllWindows()  # pylint: disable=no-member
        super().close()

    def set_render_camera(self, camera_name: str | None) -> None:
        """Set the camera to use for rendering."""
        self._render_camera_name = camera_name

    def _create_env_markdown_description(self) -> str:
        """Create environment description (policy-agnostic)."""
        scene_description = ""
        if self.scene_type == "ground":
            scene_description = (
                " In the 'ground' scene, objects are placed randomly on a flat "
                "ground plane."
            )

        return f"""A 3D mobile manipulation environment using the TidyBot platform.

The robot has a holonomic mobile base with powered casters and a Kinova Gen3 arm.
Scene type: {self.scene_type} with {self.num_objects} objects.{scene_description}

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
        if self.scene_type == "ground":
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

    @classmethod
    def get_available_environments(cls) -> list[str]:
        """Get list of available TidyBot environment IDs (policy-agnostic)."""
        scene_configs = [
            ("ground", [3, 5, 7]),
        ]
        env_ids = []
        for scene_type, object_counts in scene_configs:
            for num_objects in object_counts:
                env_ids.append(f"prbench/TidyBot3D-{scene_type}-o{num_objects}-v0")
        return env_ids
