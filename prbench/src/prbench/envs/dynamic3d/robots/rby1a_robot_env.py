"""This module defines the RBY1ARobotEnv class, which is the base class for the RBY-1A
robot in simulation."""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional

import mujoco
import numpy as np
from numpy.typing import NDArray
from relational_structs import Array

from prbench.core import RobotActionSpace
from prbench.envs.dynamic3d.mujoco_utils import MjObs
from prbench.envs.dynamic3d.robots.base import RobotEnv


class RBY1ARobotActionSpace(RobotActionSpace):
    """An action in a MuJoCo environment; used to set sim.data.ctrl in MuJoCo."""

    def __init__(self) -> None:
        # Robot actions: joint positions for 2 base joints, 6 torso joints,
        # 7 right arm joints, 7 left arm joints, 2 head joints
        low = np.array([-300] * 24)
        high = np.array([300] * 24)
        super().__init__(low, high)

    def create_markdown_description(self) -> str:
        """Create a human-readable markdown description of this space."""
        return (
            """Actions: joint positions for 2 base joints, 6 torso joints, """
            """7 right arm joints, 7 left arm joints, 2 head joints"""
        )


class RBY1ARobotEnv(RobotEnv):
    """This is the base class for RBY-1A environments that use MuJoCo for sim.

    It is still abstract: subclasses define rewards and add objects to the env.
    """

    def __init__(
        self,
        control_frequency: float,
        act_delta: bool = True,
        horizon: int = 1000,
        camera_names: Optional[list[str]] = None,
        camera_width: int = 640,
        camera_height: int = 480,
        seed: Optional[int] = None,
        show_viewer: bool = False,
    ) -> None:
        """
        Args:
            control_frequency: Frequency at which control actions are applied (in Hz).
            horizon: Maximum number of steps per episode.
            camera_names: List of camera names to use for rendering.
            camera_width: Width of camera images.
            camera_height: Height of camera images.
            seed: Random seed for reproducibility.
            show_viewer: Whether to show the MuJoCo viewer.
        """
        super().__init__(
            control_frequency,
            horizon=horizon,
            camera_names=camera_names,
            camera_width=camera_width,
            camera_height=camera_height,
            seed=seed,
            show_viewer=show_viewer,
        )

        self.act_delta = act_delta

        # Initialize robot state attributes
        self.joint_indices: list[int] = []
        self.joint_indices_ctrl: list[int] = []
        self.exclude_parts: list[str] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[MjObs, dict[str, Any]]:
        """Reset the RBY-1A robot environment.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options for resetting the environment.

        Returns:
            A tuple containing the observation and info dict.
        """
        # Access the original xml.
        assert options is not None and "xml" in options, "XML required to reset env"
        xml_string = options["xml"]
        # Insert the robot into the xml string.
        xml_string = self._insert_robot_into_xml(xml_string)
        super().reset(seed=seed, options={"xml": xml_string})

        # Setup references to robot state/actuator buffers
        self._setup_robot_references()

        # Randomize the base pose of the robot in the sim
        self._randomize_base_pose()
        self._randomize_arm_and_torso_pose()

        return self.get_obs(), {}

    def _insert_robot_into_xml(self, xml_string: str) -> str:
        """Insert the robot model into the provided XML string."""
        # Parse the provided XML string
        input_tree = ET.ElementTree(ET.fromstring(xml_string))
        input_root = input_tree.getroot()

        # Read the scene XML content
        models_dir = Path(__file__).parent.parent / "models" / "rby1a"
        robot_path = models_dir / "rby1a_model_v1.2.xml"
        assets_dir = models_dir / "assets"
        # NOTE: currently manually handling duplicate geoms.xml
        # by creating duplicate asset directories. Probably
        # handle that in code through recursive include.

        with open(robot_path, "r", encoding="utf-8") as f:
            robot_content = f.read()

        # Parse robot XML
        robot_tree = ET.ElementTree(ET.fromstring(robot_content))
        robot_root = robot_tree.getroot()
        if robot_root is None:
            raise ValueError("Missing robot element")

        # Update compiler meshdir to absolute path in robot content
        robot_compiler = robot_root.find("compiler")  # type: ignore[union-attr]
        if robot_compiler is not None:
            robot_compiler.set("meshdir", str(assets_dir.resolve()))

        # Helper function to recursively make include file paths absolute
        def make_include_paths_absolute(element: ET.Element) -> None:
            """Recursively process an element and its children to make include file
            paths absolute."""
            if element.tag == "include" and element.get("file") is not None:
                file_path = element.get("file")
                if file_path and not Path(file_path).is_absolute():
                    # Make the file path absolute relative to the models directory
                    absolute_path = models_dir / file_path
                    element.set("file", str(absolute_path.resolve()))

            # Recursively process all children
            for child_elem in element:
                make_include_paths_absolute(child_elem)

        # Merge the robot content into the input XML
        # Copy all children from robot root to input root (except mujoco tag itself)
        for child in list(robot_root):
            if child.tag == "worldbody":
                # Merge worldbody content
                input_worldbody = input_root.find(  # type:ignore[union-attr]
                    "worldbody"
                )
                if input_worldbody is not None:
                    for robot_body in list(child):
                        # Process any include tags within robot_body and its children
                        make_include_paths_absolute(robot_body)
                        input_worldbody.append(robot_body)
                else:
                    input_root.append(child)  # type: ignore[union-attr]
            elif child.tag == "default":
                # Merge or append default sections
                input_section = input_root.find(child.tag)  # type: ignore[union-attr]
                if input_section is not None:
                    for sub_child in list(child):
                        input_section.append(sub_child)
                else:
                    input_root.append(child)  # type: ignore[union-attr]
            elif child.tag == "asset":
                # Merge or append asset sections
                input_section = input_root.find(child.tag)  # type: ignore[union-attr]
                if input_section is not None:
                    for sub_child in list(child):
                        # Check if the asset element has a "file" attribute
                        # and make it absolute
                        if sub_child.get("file") is not None:
                            file_path = sub_child.get("file")
                            if file_path and not Path(file_path).is_absolute():
                                # Make the file path absolute relative to the
                                # assets directory
                                absolute_path = models_dir / file_path
                                sub_child.set("file", str(absolute_path.resolve()))
                        input_section.append(sub_child)
                else:
                    input_root.append(child)  # type: ignore[union-attr]
            else:
                # For other sections (compiler, actuator, contact, etc.), just append
                input_root.append(child)  # type: ignore[union-attr]

        if input_root is None:
            raise ValueError("input_root is None, cannot serialize to string")

        # Return the merged XML as string
        return ET.tostring(input_root, encoding="unicode")

    def _setup_robot_references(self) -> None:
        """Setup references to robot state/actuator buffers."""
        assert self.sim is not None, "Simulation must be initialized."

        robot_joint_names = {
            "base": ["right_wheel", "left_wheel"],
            "torso": ["torso_0", "torso_1", "torso_2", "torso_3", "torso_4", "torso_5"],
            "right_arm": [
                "right_arm_0",
                "right_arm_1",
                "right_arm_2",
                "right_arm_3",
                "right_arm_4",
                "right_arm_5",
                "right_arm_6",
            ],
            "left_arm": [
                "left_arm_0",
                "left_arm_1",
                "left_arm_2",
                "left_arm_3",
                "left_arm_4",
                "left_arm_5",
                "left_arm_6",
            ],
            "head": ["head_0", "head_1"],
        }
        robot_actuator_names = {
            "base": ["right_wheel_act", "left_wheel_act"],
            "torso": [
                "link1_act",
                "link2_act",
                "link3_act",
                "link4_act",
                "link5_act",
                "link6_act",
            ],
            "right_arm": [
                "right_arm_1_act",
                "right_arm_2_act",
                "right_arm_3_act",
                "right_arm_4_act",
                "right_arm_5_act",
                "right_arm_6_act",
                "right_arm_7_act",
            ],
            "left_arm": [
                "left_arm_1_act",
                "left_arm_2_act",
                "left_arm_3_act",
                "left_arm_4_act",
                "left_arm_5_act",
                "left_arm_6_act",
                "left_arm_7_act",
            ],
            "head": ["head_0_act", "head_1_act"],
        }

        # Joint positions: joint_id corresponds to qpos index
        qpos_indices = {
            part: [
                self.sim.model.get_joint_qpos_addr(joint_name)
                for joint_name in joint_names
            ]
            for part, joint_names in robot_joint_names.items()
        }

        # Joint velocities: joint_id corresponds to qvel index
        qvel_indices = {
            part: [
                self.sim.model.get_joint_qvel_addr(joint_name)
                for joint_name in joint_names
            ]
            for part, joint_names in robot_joint_names.items()
        }

        # Actuators: actuator_id corresponds to ctrl index
        ctrl_indices = {
            part: [
                self.sim.model._actuator_name2id[  # pylint: disable=protected-access
                    actuator_name
                ]
                for actuator_name in actuator_names
            ]
            for part, actuator_names in robot_actuator_names.items()
        }

        # Verify indices are contiguous for slicing
        for part in qpos_indices:
            indices = qpos_indices[part]
            assert indices == list(
                range(min(indices), max(indices) + 1)
            ), f"Non-contiguous qpos indices for part {part}"
        for part in qvel_indices:
            indices = qvel_indices[part]
            assert indices == list(
                range(min(indices), max(indices) + 1)
            ), f"Non-contiguous qvel indices for part {part}"
        for part in ctrl_indices:
            indices = ctrl_indices[part]
            assert indices == list(
                range(min(indices), max(indices) + 1)
            ), f"Non-contiguous ctrl indices for part {part}"

        # Create views using correct slice ranges
        qpos_start_end = {
            part: (min(indices), max(indices) + 1)
            for part, indices in qpos_indices.items()
        }
        qvel_start_end = {
            part: (min(indices), max(indices) + 1)
            for part, indices in qvel_indices.items()
        }
        ctrl_start_end = {
            part: (min(indices), max(indices) + 1)
            for part, indices in ctrl_indices.items()
        }

        self.qpos = {
            part: self.sim.data._data.qpos[  # pylint: disable=protected-access
                start:end
            ]
            for part, (start, end) in qpos_start_end.items()
        }
        self.qvel = {
            part: self.sim.data._data.qvel[  # pylint: disable=protected-access
                start:end
            ]
            for part, (start, end) in qvel_start_end.items()
        }
        self.ctrl = {
            part: self.sim.data._data.ctrl[  # pylint: disable=protected-access
                start:end
            ]
            for part, (start, end) in ctrl_start_end.items()
        }

        # Store all joint indices (in qvel) for which joint torques will be computed.
        self.joint_indices.clear()
        self.joint_indices_ctrl.clear()  # This could be used to set ctrl directly
        self.exclude_parts = ["base"]  # Exclude base joints from jacobian
        for part in qvel_indices:
            if part not in self.exclude_parts:  # exclude base joints from jacobian
                self.joint_indices.extend(qvel_indices[part])
                self.joint_indices_ctrl.extend(ctrl_indices[part])

    def _randomize_base_pose(self) -> None:
        """Randomize the base pose of the robot within defined limits."""
        assert (
            self.sim is not None
        ), "Simulation must be initialized before randomizing base pose."
        assert self.qpos["base"] is not None, "Base qpos must be initialized first"
        assert self.ctrl["base"] is not None, "Base ctrl must be initialized first"

        # Define limits for x, y, and theta
        left_limit = (-1.0, 1.0)
        right_limit = (-1.0, 1.0)
        # Sample random values within the limits
        left = self.np_random.uniform(*left_limit)
        right = self.np_random.uniform(*right_limit)
        # Set the base position and orientation in the simulation
        self.qpos["base"][:] = [left, right]
        self.ctrl["base"][:] = [left, right]
        self.sim.forward()  # Update the simulation state

    def _randomize_arm_and_torso_pose(self) -> None:
        """Randomize the arm and torso pose of the robot within defined limits."""
        assert (
            self.sim is not None
        ), "Simulation must be initialized before randomizing arm and torso pose."
        assert self.qpos["torso"] is not None, "Torso qpos must be initialized first"
        assert self.ctrl["torso"] is not None, "Torso ctrl must be initialized first"
        assert (
            self.qpos["right_arm"] is not None
        ), "Right arm qpos must be initialized first"
        assert (
            self.ctrl["right_arm"] is not None
        ), "Right arm ctrl must be initialized first"
        assert (
            self.qpos["left_arm"] is not None
        ), "Left arm qpos must be initialized first"
        assert (
            self.ctrl["left_arm"] is not None
        ), "Left arm ctrl must be initialized first"

        # Initial pose for torso and arms
        torso_pose = np.deg2rad([0.0, 45.0, -90.0, 45.0, 0.0, 0.0])
        right_arm_pose = np.deg2rad([0.0, -5.0, 0.0, -120.0, 0.0, 70.0, 0.0])
        left_arm_pose = np.deg2rad([0.0, 5.0, 0.0, -120.0, 0.0, 70.0, 0.0])

        # Define limits for torso and arms
        torso_limits = [(-0.5, 0.5)] * 6  # Example limits for 6 DOF torso
        arm_limits = [(-1.0, 1.0)] * 7  # Example limits for 7 DOF arms

        # Randomize the torso and arm poses within defined limits
        torso_pose = np.clip(
            torso_pose
            + np.deg2rad(
                [self.np_random.uniform() * 10 - 5 for _ in range(len(torso_pose))]
            ),
            [low for low, high in torso_limits],
            [high for low, high in torso_limits],
        )
        right_arm_pose = np.clip(
            right_arm_pose
            + np.deg2rad(
                [self.np_random.uniform() * 10 - 5 for _ in range(len(right_arm_pose))]
            ),
            [low for low, high in arm_limits],
            [high for low, high in arm_limits],
        )
        left_arm_pose = np.clip(
            left_arm_pose
            + np.deg2rad(
                [self.np_random.uniform() * 10 - 5 for _ in range(len(left_arm_pose))]
            ),
            [low for low, high in arm_limits],
            [high for low, high in arm_limits],
        )

        # Set the torso and arm positions in the simulation
        self.qpos["torso"][:] = torso_pose
        self.ctrl["torso"][:] = torso_pose
        self.qpos["right_arm"][:] = right_arm_pose
        self.ctrl["right_arm"][:] = right_arm_pose
        self.qpos["left_arm"][:] = left_arm_pose
        self.ctrl["left_arm"][:] = left_arm_pose

        self.sim.forward()  # Update the simulation state

    @property
    def jacobian_mat(self) -> NDArray[np.float64]:
        """Returns the pos and ori jacobian for the robot joints."""
        assert self.sim is not None, "Simulation must be initialized."
        body_name = "EE_BODY_R"  # End-effector body name (using right arm only)
        jacobian_pos = \
            self.sim.data.get_body_jacp(body_name)[  # type: ignore[no-untyped-call]
                :, self.joint_indices
            ]  # (3, num_joints)
        jacobian_ori = \
            self.sim.data.get_body_jacr(body_name)[  # type: ignore[no-untyped-call]
                :, self.joint_indices
            ]  # (3, num_joints)
        jacobian = np.concatenate([jacobian_pos, jacobian_ori], 0)  # (6, num_joints)
        return jacobian

    @property
    def mass_mat(self) -> NDArray[np.float64]:
        """Returns the mass matrix for the robot joints."""
        assert self.sim is not None, "Simulation must be initialized."
        mass_matrix: NDArray[np.float64] = np.ndarray(
            shape=(
                self.sim.model._model.nv,  # pylint: disable=protected-access
                self.sim.model._model.nv,  # pylint: disable=protected-access
            ),
            dtype=np.float64,
        )
        mujoco.mj_fullM(  # pylint: disable=no-member
            self.sim.model._model,  # pylint: disable=no-member,protected-access
            mass_matrix,
            self.sim.data._data.qM,  # pylint: disable=no-member,protected-access
        )
        mass_matrix = np.reshape(
            mass_matrix,
            (
                self.sim.model._model.nv,  # pylint: disable=protected-access
                self.sim.model._model.nv,  # pylint: disable=protected-access
            ),
        )
        mass_matrix = mass_matrix[self.joint_indices, :][:, self.joint_indices]
        return mass_matrix

    @property
    def lambda_mat(self) -> NDArray[np.float64]:
        """Returns the lambda matrix for the robot."""

        jacobian = self.jacobian_mat
        mass_matrix_inv = np.linalg.inv(self.mass_mat)

        # J M^-1 J^T
        lambda_full_inv = np.dot(
            np.dot(jacobian, mass_matrix_inv), jacobian.transpose()
        )

        # take the inverses, but zero out small singular values for stability
        lambda_full = np.linalg.pinv(lambda_full_inv)

        return lambda_full

    @property
    def torque_compensation(self) -> NDArray[np.float64]:
        """Return torque compensation values."""
        assert self.sim is not None, "Simulation must be initialized."
        return self.sim.data._data.qfrc_bias[  # pylint: disable=protected-access
            self.joint_indices
        ]

    def _update_ctrl(self, action) -> None:
        start = 0
        for part in self.ctrl:
            # if part not in self.exclude_parts:
            end = start + len(self.ctrl[part])
            self.ctrl[part][:] = action[start:end]
            start = end

    def step(self, action: Array) -> tuple[MjObs, float, bool, bool, dict[str, Any]]:
        """Step the RBY-1A robot environment with the given action.

        Args:
            action: The action to take in the environment.

        Returns:
            A tuple containing (observation, reward, terminated, truncated, info).
        """
        if self.act_delta:  # Interpret action as delta.
            # Compute absolute joint action.
            curr_qpos = np.concatenate([self.qpos[part] for part in self.qpos], -1)
            abs_action = curr_qpos + action
            return super().step(abs_action)
        # Use action as-is.
        return super().step(action)

    def reward(self, obs: MjObs) -> float:
        """Compute the reward from an observation.

        This is a placeholder implementation for the RBY-1A robot.

        Args:
            obs: The observation to compute reward from.

        Returns:
            The computed reward value.
        """
        # Placeholder reward - always returns 0.0
        return 0.0
