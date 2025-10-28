# Author: Jimmy Wu
# Date: October 2024
#
# References:
# - https://github.com/bulletphysics/bullet3/blob/master/examples/ThirdPartyLibs/BussIK/Jacobian.cpp
# - https://github.com/kevinzakka/mjctrl/blob/main/diffik_nullspace.py
# - https://github.com/google-deepmind/dm_control/blob/main/dm_control/utils/inverse_kinematics.py

# mypy: ignore-errors
# pylint: disable=all

import mujoco
import numpy as np

DAMPING_COEFF = 1e-12
MAX_ANGLE_CHANGE = np.deg2rad(45)

class IKSolver:
    def __init__(self, ee_offset=0.0):
        # Load arm without gripper
        self.model = mujoco.MjModel.from_xml_path('models/kinova_gen3/gen3.xml')
        self.data = mujoco.MjData(self.model)
        self.model.body_gravcomp[:] = 1.0

        # Cache references
        self.qpos0 = self.model.key('retract').qpos
        self.site_id = self.model.site('pinch_site').id
        self.site_pos = self.data.site(self.site_id).xpos
        self.site_mat = self.data.site(self.site_id).xmat

        # Add end effector offset for gripper
        self.model.site(self.site_id).pos = np.array([0.0, 0.0, -0.061525 - ee_offset])  # 0.061525 comes from the Kinova URDF

        # Preallocate arrays
        self.err = np.empty(6)
        self.err_pos, self.err_rot = self.err[:3], self.err[3:]
        self.site_quat = np.empty(4)
        self.site_quat_inv = np.empty(4)
        self.err_quat = np.empty(4)
        self.jac = np.empty((6, self.model.nv))
        self.jac_pos, self.jac_rot = self.jac[:3], self.jac[3:]
        self.damping = DAMPING_COEFF * np.eye(6)
        self.eye = np.eye(self.model.nv)

    def solve(self, pos, quat, curr_qpos, max_iters=20, err_thresh=1e-4):
        quat = quat[[3, 0, 1, 2]]  # (x, y, z, w) -> (w, x, y, z)

        # Set arm to initial joint configuration
        self.data.qpos = curr_qpos

        for _ in range(max_iters):
            # Update site pose
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)

            # Translational error
            self.err_pos[:] = pos - self.site_pos

            # Rotational error
            mujoco.mju_mat2Quat(self.site_quat, self.site_mat)
            mujoco.mju_negQuat(self.site_quat_inv, self.site_quat)
            mujoco.mju_mulQuat(self.err_quat, quat, self.site_quat_inv)
            mujoco.mju_quat2Vel(self.err_rot, self.err_quat, 1.0)

            # Check if target pose reached
            if np.linalg.norm(self.err) < err_thresh:
                break

            # Calculate update
            mujoco.mj_jacSite(self.model, self.data, self.jac_pos, self.jac_rot, self.site_id)
            update = self.jac.T @ np.linalg.solve(self.jac @ self.jac.T + self.damping, self.err)
            qpos0_err = np.mod(self.qpos0 - self.data.qpos + np.pi, 2 * np.pi) - np.pi
            update += (self.eye - (self.jac.T @ np.linalg.pinv(self.jac @ self.jac.T + self.damping)) @ self.jac) @ qpos0_err

            # Enforce max angle change
            update_max = np.abs(update).max()
            if update_max > MAX_ANGLE_CHANGE:
                update *= MAX_ANGLE_CHANGE / update_max

            # Apply update
            mujoco.mj_integratePos(self.model, self.data.qpos, update, 1.0)

        return self.data.qpos.copy()

if __name__ == '__main__':
    ik_solver = IKSolver()
    home_pos, home_quat = np.array([0.456, 0.0, 0.434]), np.array([0.5, 0.5, 0.5, 0.5])
    retract_qpos = np.deg2rad([0, -20, 180, -146, 0, -50, 90])

    import time
    start_time = time.time()
    for _ in range(1000):
        qpos = ik_solver.solve(home_pos, home_quat, retract_qpos)
    elapsed_time = time.time() - start_time
    print(f'Time per call: {elapsed_time:.3f} ms')  # 0.59 ms

    # Home: 0, 15, 180, -130, 0, 55, 90
    print(np.rad2deg(ik_solver.solve(home_pos, home_quat, retract_qpos)).round())
