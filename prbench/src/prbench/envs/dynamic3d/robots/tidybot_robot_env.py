"""This module defines the TidyBotRobotEnv class, which is the base class for the
TidyBot robot in simulation."""

from pathlib import Path
from typing import Any, Optional

import numpy as np
from relational_structs import Array

from prbench.core import RobotActionSpace
from prbench.envs.dynamic3d.mujoco_utils import MjObs
from prbench.envs.dynamic3d.robots.base import RobotEnv


class TidyBot3DRobotActionSpace(RobotActionSpace):
    """An action in a MuJoCo environment; used to set sim.data.ctrl in MuJoCo."""

    def __init__(self) -> None:
        # TidyBot actions: base pos and yaw (3), arm joints (7), gripper pos (1)
        low = np.array(
            [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, 0.0]
        )
        high = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0])
        super().__init__(low, high)

    def create_markdown_description(self) -> str:
        """Create a human-readable markdown description of this space."""
        return """Actions: base pos and yaw (3), arm joints (7), gripper pos (1)"""


class TidyBotRobotEnv(RobotEnv):
    """This is the base class for TidyBot environments that use MuJoCo for sim.

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
            act_delta: Whether to interpret actions as deltas or absolute values.
            horizon: Maximum number of steps per episode.
            camera_names: List of camera names for rendering.
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

    def _setup_robot_references(self) -> None:
        """Setup references to robot state/actuator buffers in the simulation data."""
        assert self.sim is not None, "Simulation must be initialized."

        # Joint names for the base and arm
        base_joint_names: list[str] = ["joint_x", "joint_y", "joint_th"]
        arm_joint_names: list[str] = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "joint_7",
        ]
        gripper_joint_names = ["right_driver_joint", "left_driver_joint"]

        # Joint positions: joint_id corresponds to qpos index
        base_qpos_indices = [
            self.sim.model.get_joint_qpos_addr(name) for name in base_joint_names
        ]
        arm_qpos_indices = [
            self.sim.model.get_joint_qpos_addr(name) for name in arm_joint_names
        ]
        gripper_qpos_indices = [
            self.sim.model.get_joint_qpos_addr(name) for name in gripper_joint_names
        ]

        # Joint velocities: joint_id corresponds to qvel index
        base_qvel_indices = [
            self.sim.model.get_joint_qvel_addr(name) for name in base_joint_names
        ]
        arm_qvel_indices = [
            self.sim.model.get_joint_qvel_addr(name) for name in arm_joint_names
        ]
        gripper_qvel_indices = [
            self.sim.model.get_joint_qvel_addr(name) for name in gripper_joint_names
        ]

        # Actuators: actuator_id corresponds to ctrl index
        base_ctrl_indices = [
            self.sim.model._actuator_name2id[name]  # pylint: disable=protected-access
            for name in base_joint_names
        ]
        arm_ctrl_indices = [
            self.sim.model._actuator_name2id[name]  # pylint: disable=protected-access
            for name in arm_joint_names
        ]
        gripper_ctrl_indices = [
            self.sim.model._actuator_name2id[name]  # pylint: disable=protected-access
            for name in ["fingers_actuator"]
        ]

        # Verify indices are contiguous for slicing
        assert base_qpos_indices == list(
            range(min(base_qpos_indices), max(base_qpos_indices) + 1)
        ), "Base qpos indices not contiguous"
        assert arm_qpos_indices == list(
            range(min(arm_qpos_indices), max(arm_qpos_indices) + 1)
        ), "Arm qpos indices not contiguous"
        assert base_qvel_indices == list(
            range(min(base_qvel_indices), max(base_qvel_indices) + 1)
        ), "Base qvel indices not contiguous"
        assert arm_qvel_indices == list(
            range(min(arm_qvel_indices), max(arm_qvel_indices) + 1)
        ), "Arm qvel indices not contiguous"
        assert base_ctrl_indices == list(
            range(min(base_ctrl_indices), max(base_ctrl_indices) + 1)
        ), "Base ctrl indices not contiguous"
        assert arm_ctrl_indices == list(
            range(min(arm_ctrl_indices), max(arm_ctrl_indices) + 1)
        ), "Arm ctrl indices not contiguous"

        # Create views using correct slice ranges
        base_qpos_start, base_qpos_end = (
            min(base_qpos_indices),
            max(base_qpos_indices) + 1,
        )
        base_qvel_start, base_qvel_end = (
            min(base_qvel_indices),
            max(base_qvel_indices) + 1,
        )
        arm_qpos_start, arm_qpos_end = min(arm_qpos_indices), max(arm_qpos_indices) + 1
        arm_qvel_start, arm_qvel_end = (
            min(arm_qvel_indices),
            max(arm_qvel_indices) + 1,
        )
        base_ctrl_start, base_ctrl_end = (
            min(base_ctrl_indices),
            max(base_ctrl_indices) + 1,
        )
        arm_ctrl_start, arm_ctrl_end = min(arm_ctrl_indices), max(arm_ctrl_indices) + 1

        self.qpos["base"] = self.sim.data.mj_data.qpos[base_qpos_start:base_qpos_end]
        self.qvel["base"] = self.sim.data.mj_data.qvel[base_qvel_start:base_qvel_end]
        self.ctrl["base"] = self.sim.data.mj_data.ctrl[base_ctrl_start:base_ctrl_end]

        self.qpos["arm"] = self.sim.data.mj_data.qpos[arm_qpos_start:arm_qpos_end]
        self.qvel["arm"] = self.sim.data.mj_data.qvel[arm_qvel_start:arm_qvel_end]
        self.ctrl["arm"] = self.sim.data.mj_data.ctrl[arm_ctrl_start:arm_ctrl_end]

        # Create a custom wrapper that maintains references for
        # non-contiguous gripper indices
        class IndexedView:
            """A view that provides indexed access to non-contiguous array elements."""

            def __init__(self, array: Any, indices: list[int]) -> None:
                self.array = array
                self.indices = indices

            def __setitem__(self, key: int, value: Any) -> None:
                self.array[self.indices[key]] = value

            def __getitem__(self, key: int) -> Any:
                return self.array[self.indices[key]]

            def __len__(self) -> int:
                return len(self.indices)

        self.qpos["gripper"] = IndexedView(  # type: ignore[assignment]
            self.sim.data.mj_data.qpos, gripper_qpos_indices
        )
        self.qvel["gripper"] = IndexedView(  # type: ignore[assignment]
            self.sim.data.mj_data.qvel, gripper_qvel_indices
        )
        self.ctrl["gripper"] = IndexedView(  # type: ignore[assignment]
            self.sim.data.mj_data.ctrl, gripper_ctrl_indices
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[MjObs, dict[str, Any]]:
        """Reset the robot environment.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options, must contain 'xml' key.

        Returns:
            Tuple of observation and info dict.
        """
        # Access the original xml.
        assert options is not None and "xml" in options, "XML required to reset env"
        xml_string = options["xml"]

        # Insert the robot into the xml string.
        xml_string = self._insert_robot_into_xml(
            xml_string,
            str(Path(__file__).parents[1] / "models" / "stanford_tidybot"),
            "tidybot.xml",
            str(Path(__file__).parents[1] / "models" / "assets"),
        )
        super().reset(seed=seed, options={"xml": xml_string})

        # Setup references to robot state/actuator buffers
        self._setup_robot_references()

        # Randomize the base pose of the robot in the sim
        self._randomize_base_pose()
        self._randomize_arm_pose()

        return self.get_obs(), {}

    def _randomize_base_pose(self) -> None:
        """Randomize the base pose of the robot within defined limits."""
        assert (
            self.sim is not None
        ), "Simulation must be initialized before randomizing base pose."
        assert self.qpos is not None, "Base qpos must be initialized first"
        assert self.ctrl is not None, "Base ctrl must be initialized first"

        # Define limits for x, y, and theta
        x_limit = (-1.0, 1.0)
        y_limit = (-1.0, 1.0)
        theta_limit = (-np.pi, np.pi)
        # Sample random values within the limits
        x = self.np_random.uniform(*x_limit)
        y = self.np_random.uniform(*y_limit)
        theta = self.np_random.uniform(*theta_limit)
        # Set the base position and orientation in the simulation
        self.qpos["base"][:] = [x, y, theta]
        self.ctrl["base"][:] = [x, y, theta]
        self.sim.forward()  # Update the simulation state

    def _randomize_arm_pose(self) -> None:
        """Randomize the arm pose of the robot within defined limits."""
        assert (
            self.sim is not None
        ), "Simulation must be initialized before randomizing base pose."
        assert self.qpos is not None, "Base qpos must be initialized first"
        assert self.ctrl is not None, "Base ctrl must be initialized first"

        # Sample random values within limits
        num_joints: int = self.qpos["arm"].shape[0]
        theta = self.np_random.uniform(-np.pi, np.pi, num_joints).astype(np.float64)
        # Set the arm joint positions in the simulation
        self.qpos["arm"][:] = theta
        self.ctrl["arm"][:] = theta
        self.sim.forward()  # Update the simulation state

    def _update_ctrl(self, action: Array) -> None:
        """Update control values from action array.

        Args:
            action: Action array to apply to robot controls.
        """
        start = 0
        for _, ctrl_part in self.ctrl.items():
            end = start + len(ctrl_part)
            ctrl_part[:] = action[start:end]
            start = end

    def step(self, action: Array) -> tuple[MjObs, float, bool, bool, dict[str, Any]]:
        # Map gripper action from [0, 1] to [0, 255].
        action = action.copy()
        action[-1] = action[-1] * 255.0
        # Ctrl values > 127 apply closing force, < 127 apply opening force;
        # hence, 0 = fully open, 255 = fully closed, 127 = no force applied.
        # To ensure full closure, map 1.0 to 255, as lower forces may not fully close
        # the gripper due to opposing spring forces.

        # Apply delta or absolute action
        if self.act_delta:  # Interpret action as delta.
            # Compute absolute joint action.
            curr_qpos = np.concatenate([self.qpos["base"], self.qpos["arm"]], -1)
            abs_action = curr_qpos + action[:-1]
            # Add gripper action
            abs_action = np.concatenate([abs_action, [action[-1]]], -1)
            return super().step(abs_action)
        # Use action as-is.
        return super().step(action)

    def reward(self, obs: MjObs) -> float:
        """Compute the reward from an observation.

        This is a placeholder implementation since TidyBotRobotEnv is used as a
        component in TidyBot3DEnv which handles rewards separately.
        """
        return 0.0
