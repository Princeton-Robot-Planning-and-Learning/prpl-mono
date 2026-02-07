"""Forward kinematics solver for tidybot arm control.

This module provides a forward kinematics solver that uses the MuJoCo physics engine to
compute the end-effector pose from joint positions.
"""

from pathlib import Path

import kinder
import mujoco
import numpy as np


class TidybotFKSolver:
    """Forward kinematics solver for Tidybot arm control.

    This class provides methods to compute the end-effector pose from joint positions.
    """

    def __init__(
        self,
        ee_offset: float = 0.0,
    ) -> None:
        # Load arm without gripper
        model_path = (
            Path(kinder.__file__).parent
            / "envs"
            / "dynamic3d"
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

        self.site_quat = np.empty(4)

    def forward_kinematics(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute forward kinematics to get end-effector pose from joint positions.

        Args:
            qpos: Joint positions (7 values for 7-DOF arm)

        Returns:
            pos: End-effector position (x, y, z) in meters
            quat: End-effector orientation as quaternion (x, y, z, w)
        """
        assert qpos.shape == (7,)
        # Set joint positions
        self.data.qpos[:] = qpos

        # Run forward kinematics
        mujoco.mj_kinematics(self.model, self.data)  # pylint: disable=no-member
        mujoco.mj_comPos(self.model, self.data)  # pylint: disable=no-member

        # Get position from site
        pos = self.site_pos.copy()

        # Get orientation as quaternion
        mujoco.mju_mat2Quat(self.site_quat, self.site_mat)  # pylint: disable=no-member
        # Convert from (w, x, y, z) to (x, y, z, w)
        quat = self.site_quat[[1, 2, 3, 0]].copy()

        return pos, quat
