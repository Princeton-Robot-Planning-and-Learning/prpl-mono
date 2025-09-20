"""BaseController module for TidyBot.

This module defines the BaseController class, which provides control logic for the
mobile base using online trajectory generation (Ruckig). It is designed to be used
within the TidyBot simulation and control framework, and supports smooth, constrained
motion for the robot base.

The current controller is part of the environment.
"""

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


class BaseController:
    """Controller for mobile base movement using online trajectory generation.

    This class implements a controller for the mobile base using Ruckig's online
    trajectory generation to ensure smooth, constrained motion with velocity and
    acceleration limits.

    Attributes:
        qpos: Current base pose/state vector, typically ``[x, y, theta]``.
        qvel: Current base velocity vector, typically ``[vx, vy, omega]``.
        ctrl: Actuator target for the base state (same shape as ``qpos``).
        otg: Ruckig trajectory generator instance for the base.
        otg_inp: Ruckig input parameters buffer (contains target/current states and
            motion limits such as ``max_velocity`` and ``max_acceleration``).
        otg_out: Ruckig output parameters buffer (provides new positions for control).
        otg_res: Latest Ruckig result status (e.g., Working/Finished).
        motion3d_config: Environment timing/specs (e.g., ``policy_control_period``).
        command_timeout_factor: Multiplier applied to ``policy_control_period`` for
            detecting command timeouts.
        reset_qpos: Default pose used by ``reset()`` when initializing the base.
    """

    def __init__(
        self,
        qpos: NDArray[np.float64],
        qvel: NDArray[np.float64],
        ctrl: NDArray[np.float64],
        timestep: float,
        num_dofs: int = 3,
        max_velocity: Optional[Sequence[float]] = None,
        max_acceleration: Optional[Sequence[float]] = None,
    ) -> None:
        self.qpos = qpos
        self.qvel = qvel
        self.ctrl = ctrl
        # Use Ruckig for online trajectory generation
        self.otg = Ruckig(num_dofs, timestep)
        self.otg_inp = InputParameter(num_dofs)
        self.otg_out = OutputParameter(num_dofs)
        if max_velocity is None:
            max_velocity = [0.5, 0.5, 3.14]
        if max_acceleration is None:
            max_acceleration = [0.5, 0.5, 2.36]
        self.otg_inp.max_velocity = list(max_velocity)
        self.otg_inp.max_acceleration = list(max_acceleration)
        self.otg_res = None
        self.motion3d_config = Motion3DEnvSpec()

    def reset(self) -> None:
        """Reset the base controller to origin position."""
        self.ctrl[:] = self.qpos
        # Initialize OTG
        self.otg_inp.current_position = self.qpos
        self.otg_inp.current_velocity = self.qvel
        self.otg_inp.target_position = self.qpos
        self.otg_res = Result.Finished

    def run_controller(self, action) -> None:
        """Run the controller to update the base position based on OTG."""

        # Set target base qpos
        self.otg_inp.target_position = action["base_pose"]
        self.otg_res = Result.Working

        # Generate the next step in the trajectory
        self.otg_res = self.otg.update(self.otg_inp, self.otg_out)

        # Pass output back to input for next iteration
        self.otg_out.pass_to_input(self.otg_inp)

        # Apply the smoothed position to the controller
        self.ctrl[:] = self.otg_out.new_position
