"""Inverse kinematics solver for tidybot arm control.

This module provides an inverse kinematics solver that uses the MuJoCo physics
engine to compute joint configurations that achieve desired end-effector poses.
The solver implements a damped least squares approach with nullspace optimization
to maintain joint limits and preferred configurations.

Adapted from
https://github.com/jimmyyhwu/tidybot2

Other References:
- https://github.com/bulletphysics/bullet3/
  blob/master/examples/ThirdPartyLibs/BussIK/Jacobian.cpp
- https://github.com/kevinzakka/mjctrl/blob/main/diffik_nullspace.py
- https://github.com/google-deepmind/dm_control/
  blob/main/dm_control/utils/inverse_kinematics.py
"""

from pathlib import Path

import mujoco
import numpy as np
import prbench


class TidybotIKSolver:
    """Inverse kinematics solver for Tidybot arm control.

    This class provides methods to solve inverse kinematics problems using MuJoCo
    physics engine. It implements a damped least squares approach with nullspace
    optimization to maintain joint limits and preferred configurations.
    """

    def __init__(
        self,
        ee_offset: float = 0.0,
        damping_coeff: float = 1e-12,
        max_angle_change: float = np.deg2rad(45),
    ) -> None:
        # Load arm without gripper
        model_path = (
            Path(prbench.__file__).parent
            / "envs"
            / "tidybot"
            / "models"
            / "kinova_gen3"
            / "gen3.xml"
        )
        self.model = mujoco.MjModel.from_xml_path(  # pylint: disable=no-member
            str(model_path)
        )
        self.data = mujoco.MjData(self.model)  # pylint: disable=no-member
        self.model.body_gravcomp[:] = 1.0

        # Cache references
        self.qpos0 = self.model.key("retract").qpos
        self.site_id = self.model.site("pinch_site").id
        self.site_pos = self.data.site(self.site_id).xpos
        self.site_mat = self.data.site(self.site_id).xmat

        # Add end effector offset for gripper
        self.model.site(self.site_id).pos = np.array(
            [0.0, 0.0, -0.061525 - ee_offset]
        )  # 0.061525 comes from the Kinova URDF

        # Preallocate arrays
        self.err = np.empty(6)
        self.err_pos, self.err_rot = self.err[:3], self.err[3:]
        self.site_quat = np.empty(4)
        self.site_quat_inv = np.empty(4)
        self.err_quat = np.empty(4)
        self.jac = np.empty((6, self.model.nv))
        self.jac_pos, self.jac_rot = self.jac[:3], self.jac[3:]
        self.damping = damping_coeff * np.eye(6)
        self.eye = np.eye(self.model.nv)
        self.max_angle_change = max_angle_change

    def solve(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
        curr_qpos: np.ndarray,
        max_iters: int = 20,
        err_thresh: float = 1e-4,
    ) -> np.ndarray:
        """Solve inverse kinematics to achieve desired end-effector pose.

        Args:
            pos: Target position (x, y, z) in meters
            quat: Target orientation as quaternion (x, y, z, w)
            curr_qpos: Current joint positions
            max_iters: Maximum number of iterations (default: 20)
            err_thresh: Error threshold for convergence (default: 1e-4)

        Returns:
            Joint positions that achieve the target pose
        """
        quat = quat[[3, 0, 1, 2]]  # (x, y, z, w) -> (w, x, y, z)

        # Set arm to initial joint configuration
        self.data.qpos = curr_qpos

        for _ in range(max_iters):
            # Update site pose
            mujoco.mj_kinematics(self.model, self.data)  # pylint: disable=no-member
            mujoco.mj_comPos(self.model, self.data)  # pylint: disable=no-member

            # Translational error
            self.err_pos[:] = pos - self.site_pos

            # Rotational error
            mujoco.mju_mat2Quat(  # pylint: disable=no-member
                self.site_quat, self.site_mat
            )
            mujoco.mju_negQuat(  # pylint: disable=no-member
                self.site_quat_inv, self.site_quat
            )
            mujoco.mju_mulQuat(  # pylint: disable=no-member
                self.err_quat, quat, self.site_quat_inv
            )
            mujoco.mju_quat2Vel(  # pylint: disable=no-member
                self.err_rot, self.err_quat, 1.0
            )

            # Check if target pose reached
            if np.linalg.norm(self.err) < err_thresh:
                break

            # Calculate update
            mujoco.mj_jacSite(  # pylint: disable=no-member
                self.model, self.data, self.jac_pos, self.jac_rot, self.site_id
            )
            update = self.jac.T @ np.linalg.solve(
                self.jac @ self.jac.T + self.damping, self.err
            )
            qpos0_err = np.mod(self.qpos0 - self.data.qpos + np.pi, 2 * np.pi) - np.pi
            update += (
                self.eye
                - (self.jac.T @ np.linalg.pinv(self.jac @ self.jac.T + self.damping))
                @ self.jac
            ) @ qpos0_err

            # Enforce max angle change
            update_max = np.abs(update).max()
            if update_max > self.max_angle_change:
                update *= self.max_angle_change / update_max

            # Apply update
            mujoco.mj_integratePos(  # pylint: disable=no-member
                self.model, self.data.qpos, update, 1.0
            )

        return self.data.qpos.copy()
