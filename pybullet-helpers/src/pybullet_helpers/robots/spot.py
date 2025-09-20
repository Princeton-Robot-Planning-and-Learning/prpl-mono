"""Boston Dynamic Spot robot."""

from pathlib import Path

import numpy as np
import numpy.typing as npt

from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.link import get_relative_link_pose
from pybullet_helpers.robots.single_arm import (
    FingeredSingleArmPyBulletRobot,
)
from pybullet_helpers.utils import get_assets_path


class SpotPyBulletRobot(FingeredSingleArmPyBulletRobot):
    """Boston Dynamic Spot robot."""

    @classmethod
    def get_name(cls) -> str:
        return "spot"

    @property
    def default_urdf_path(self) -> Path:
        dir_path = get_assets_path() / "urdf"
        return dir_path / "spot" / "spot.urdf"

    @property
    def default_home_joint_positions(self) -> JointPositions:
        # Magic joint values read off spot in standard standing position with
        # arm tucked.
        return [
            0.00010371208190917969,
            -3.115184783935547,
            3.132749557495117,
            1.5715421438217163,
            -0.01901412010192871,
            -1.5716896057128906,
            -0.008634686470031738,
        ]

    @property
    def end_effector_name(self) -> str:
        return "hand_frame_joint"

    @property
    def tool_link_name(self) -> str:
        return "arm_link_wr1"

    @property
    def finger_joint_names(self) -> list[str]:
        return ["arm_f1x"]

    @property
    def open_fingers_state(self) -> float:
        return -np.pi / 2

    @property
    def closed_fingers_state(self) -> float:
        return 0.0

    def finger_state_to_joints(self, state: float) -> list[float]:
        return [state]

    def joints_to_finger_state(self, joint_positions: list[float]) -> float:
        assert len(joint_positions) == 1
        finger_state = joint_positions[0]
        return finger_state

    @property
    def default_inverse_kinematics_method(self) -> str:
        return "custom"

    def custom_inverse_kinematics(
        self,
        end_effector_pose: Pose,
        validate: bool = True,
        best_effort: bool = False,
        validation_atol: float = 1e-3,
    ) -> JointPositions | None:
        """Analytic IK for 6-DOF spot arm.

        I believe the original author of this is Tomas Lozano-Perez.
        """
        # The target end_effector_pose is in the world frame. Convert it into the
        # robot shoulder frame.
        world_to_base = self.get_base_pose()
        world_to_shoulder = multiply_poses(world_to_base, Pose((0.292, 0.0, 0.873)))
        shoulder_to_ee_target = multiply_poses(
            world_to_shoulder.invert(), end_effector_pose
        )

        # The analytic IK code is in terms of the arm_wr1 frame, but the end effector
        # is hand_frame_joint, so we need to transform.
        arm_wr1_to_ee = get_relative_link_pose(
            self.robot_id,
            self.link_from_name("arm_link_wr1"),
            self.link_from_name("hand_frame"),
            self.physics_client_id,
        )
        shoulder_to_arm_wr1_target = multiply_poses(
            shoulder_to_ee_target, arm_wr1_to_ee
        )
        solns = _analytic_spot_ik_6(
            shoulder_to_arm_wr1_target.to_matrix(),
            self.joint_lower_limits[:6],
            self.joint_upper_limits[:6],
        )
        if not solns:
            return None
        # Arbitrarily select the first one. Later, might want to use distances.
        return list(solns[0]) + [self.get_finger_state()]


def _analytic_spot_ik_6(
    wrist_pose_rel_shoulder: npt.NDArray,
    min_limits: JointPositions,
    max_limits: JointPositions,
) -> list[JointPositions]:
    px, py, pz = wrist_pose_rel_shoulder[:3, 3]  # wrist position
    l2 = 0.3385
    l3 = np.sqrt(0.40330**2 + 0.0750**2)
    q3_off = np.arctan2(0.0750, 0.40330)
    # Solve the first three joints based on position.
    q123_sols = []
    xl = np.sqrt(px**2 + py**2)
    q23 = _IK2R(l2, l3, xl, -pz)
    if q23 is None:
        return []
    q1 = np.arctan2(py, px)
    for q2, q3 in q23:
        q123_sols.append((q1, q2, q3 + q3_off))
    q23 = _IK2R(l2, l3, -xl, -pz)
    assert q23 is not None
    q1 = q1 + np.pi
    for q2, q3 in q23:
        q123_sols.append((q1, q2, q3 + q3_off))

    # Solve for the wrist angles.
    qfull_sols = []
    for q1, q2, q3 in q123_sols:
        r3_inv = np.linalg.inv(
            _rotation_matrix(q1, (0, 0, 1)) @ _rotation_matrix(q2 + q3, (0, 1, 0))
        )
        W = r3_inv @ wrist_pose_rel_shoulder
        q5 = np.arccos(W[0, 0])
        for q in (q5, -q5):
            s5 = np.sin(q)
            q4 = np.arctan2(W[1, 0] / s5, -W[2, 0] / s5)
            q6 = np.arctan2(W[0, 1] / s5, W[0, 2] / s5)
            angles = (q1, q2, q3, q4, q, q6)
            if (
                np.less_equal(min_limits, angles).all()
                and np.less_equal(angles, max_limits).all()
            ):
                qfull_sols.append(list(angles))
    return qfull_sols


def _IK2R(L1: float, L2: float, x: float, y: float) -> list[tuple[float, float]] | None:
    xy_sq = x**2 + y**2
    c2 = (xy_sq - L1**2 - L2**2) / (2 * L1 * L2)
    if abs(c2) > 1:
        return None
    if c2 == 1.0:
        return [(np.arctan2(y, x), 0)]
    if c2 == -1.0 and xy_sq != 0.0:
        return [(np.arctan2(y, x), np.pi)]
    if c2 == -1.0 and xy_sq == 0.0:
        return [(q1, np.pi) for q1 in [0, 2 * np.pi]]
    q2_1 = np.arccos(c2)
    q2_2 = -q2_1
    theta = np.arctan2(y, x)
    q1q2 = [
        (theta - np.arctan2(L2 * np.sin(q2_i), L1 + L2 * np.cos(q2_i)), q2_i)
        for q2_i in (q2_1, q2_2)
    ]
    for q1, q2 in q1q2:
        xk = L1 * np.cos(q1) + L2 * np.cos(q1 + q2)
        yk = L1 * np.sin(q1) + L2 * np.sin(q1 + q2)
        assert abs(x - xk) < 0.0001
        assert abs(y - yk) < 0.0001
    return q1q2


def _rotation_matrix(
    angle: float, direction: tuple[float, float, float]
) -> npt.NDArray:
    """Return matrix to rotate about axis defined by point and direction."""
    sina = np.sin(angle)
    cosa = np.cos(angle)
    direction = direction[:3] / np.linalg.norm(direction[:3])  # type: ignore
    # Rotation matrix around unit vector.
    R = np.array(
        ((cosa, 0.0, 0.0), (0.0, cosa, 0.0), (0.0, 0.0, cosa)), dtype=np.float64
    )
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array(
        (
            (0.0, -direction[2], direction[1]),
            (direction[2], 0.0, -direction[0]),
            (-direction[1], direction[0], 0.0),
        ),
        dtype=np.float64,
    )
    M = np.identity(4)
    M[:3, :3] = R
    return M
