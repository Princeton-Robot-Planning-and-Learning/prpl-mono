"""ArmController module for TidyBot.

This module defines the ArmController class, which provides control logic for the
robotic arm using inverse kinematics and online trajectory generation (Ruckig). It is
designed to be used within the TidyBot simulation and control framework, and supports
smooth, constrained motion for the arm and gripper.

The current controller is part of the environment.
"""

import math
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from prbench.envs.tidybot.motion3d import Motion3DEnvSpec
from ruckig import (  # pylint: disable=no-name-in-module
    InputParameter,
    OutputParameter,
    Result,
    Ruckig,
)

from prbench_models.tidybot.ik_solver import TidybotIKSolver


class ArmController:
    """Controller for robotic arm movement using inverse kinematics and trajectory
    generation.

    This class implements a controller for the robotic arm using inverse kinematics to
    convert end-effector poses to joint configurations, and Ruckig's online trajectory
    generation for smooth motion control.

    Attributes:
        qpos: Current joint positions (rad).
        qvel: Current joint velocities (rad/s).
        ctrl: Actuator targets for arm joints (rad).
        qpos_gripper: Current gripper position/state (optional).
        ctrl_gripper: Actuator target for gripper (0..gripper_scale).
        ik_solver: Inverse kinematics solver (configured with ``ee_offset``).
        otg: Ruckig trajectory generator instance.
        otg_inp: Ruckig input parameters buffer.
        otg_out: Ruckig output parameters buffer.
        otg_res: Latest Ruckig result status (e.g., Working/Finished).
        motion3d_config: Environment timing/specs (e.g., ``policy_control_period``).
        command_timeout_factor: Multiplier applied to ``policy_control_period`` to
            determine command timeout.
        gripper_scale: Scales normalized gripper command to actuator units.
        retract_qpos: Joint configuration used by ``reset()``.
    """

    def __init__(
        self,
        qpos: NDArray[np.float64],
        qvel: NDArray[np.float64],
        ctrl: NDArray[np.float64],
        qpos_gripper: Optional[NDArray[np.float64]],
        ctrl_gripper: NDArray[np.float64],
        timestep: float,
        ee_offset: float = 0.12,
        num_dofs: int = 7,
        max_velocity_deg: Optional[Sequence[float]] = None,
        max_acceleration_deg: Optional[Sequence[float]] = None,
        command_timeout_factor: float = 2.5,
        gripper_scale: float = 255.0,
        retract_qpos: Optional[NDArray[np.float64]] = None,
    ) -> None:
        self.qpos = qpos
        self.qvel = qvel
        self.ctrl = ctrl
        self.qpos_gripper = qpos_gripper
        self.ctrl_gripper = ctrl_gripper
        self.ik_solver = TidybotIKSolver(ee_offset=ee_offset)
        # Ruckig (online trajectory generation)
        self.otg = Ruckig(num_dofs, timestep)
        self.otg_inp = InputParameter(num_dofs)
        self.otg_out = OutputParameter(num_dofs)
        if max_velocity_deg is None:
            max_velocity_deg = [80, 80, 80, 80, 140, 140, 140]
        if max_acceleration_deg is None:
            max_acceleration_deg = [240, 240, 240, 240, 450, 450, 450]
        self.otg_inp.max_velocity = [math.radians(v) for v in max_velocity_deg]
        self.otg_inp.max_acceleration = [math.radians(a) for a in max_acceleration_deg]
        self.otg_res = None
        self.motion3d_config = Motion3DEnvSpec()
        self.command_timeout_factor = command_timeout_factor
        self.gripper_scale = gripper_scale
        self.retract_qpos = (
            retract_qpos
            if retract_qpos is not None
            else np.array(
                [
                    0.0,
                    -0.34906585,
                    3.14159265,
                    -2.54818071,
                    0.0,
                    -0.87266463,
                    1.57079633,
                ],
                dtype=np.float64,
            )
        )

    def reset(self) -> None:
        """Reset the arm controller to retract configuration."""
        self.qpos[:] = self.retract_qpos
        self.ctrl[:] = self.qpos
        self.ctrl_gripper[:] = 0.0
        # Initialize OTG
        self.otg_inp.current_position = self.qpos
        self.otg_inp.current_velocity = self.qvel
        self.otg_inp.target_position = self.qpos
        self.otg_res = Result.Finished

    def run_controller(self, action) -> None:
        """Run the controller to update the arm position based on OTG."""

        # Update arm actuators
        target_joints = None

        if "arm_pos" in action:
            # Run inverse kinematics on new target pose
            target_joints = self.ik_solver.solve(
                action["arm_pos"], action["arm_quat"], self.qpos
            )
        elif "arm_joints" in action:
            # Direct joint command mode
            target_joints = np.array(action["arm_joints"], dtype=np.float64)

        if target_joints is not None:
            # Unwrap joint angles to avoid large jumps
            target_joints = (
                self.qpos
                + np.mod((target_joints - self.qpos) + np.pi, 2 * np.pi)
                - np.pi
            )

            # Set target arm qpos for trajectory generation
            self.otg_inp.target_position = target_joints
            self.otg_res = Result.Working

            # Generate the next step in the trajectory
            self.otg_res = self.otg.update(self.otg_inp, self.otg_out)

            # Pass output back to input for next iteration
            self.otg_out.pass_to_input(self.otg_inp)

            # Apply the smoothed position to the controller
            self.ctrl[:] = self.otg_out.new_position

        # Update gripper actuator
        if "gripper_pos" in action:
            # Set target gripper pos
            self.ctrl_gripper[:] = self.gripper_scale * action["gripper_pos"]
