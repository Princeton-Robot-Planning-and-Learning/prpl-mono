"""Stretch SE3 robot with SG3 end effector."""

import importlib.resources as importlib_resources
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.link import get_link_pose, get_relative_link_pose
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot


class StretchPyBulletRobot(FingeredSingleArmPyBulletRobot[float]):
    """Stretch SE3 robot with SG3 end effector."""

    def __init__(
        self, physics_client_id: int, fixed_base: bool = False, **kwargs
    ) -> None:
        """By default, stretch can move its based."""
        super().__init__(physics_client_id, fixed_base=fixed_base, **kwargs)

        # The wrist is used for IK 'calibration'.
        self._wrist_id = self.link_from_name("link_arm_l0")

        # Set up values for custom IK.

        # Determine reachability bounds for the arm extending outward.
        joints = list(self.joint_lower_limits)
        self.set_joints(joints)
        min_pose = self._get_relative_wrist_pose()
        self._arm_horizontal_bounds = [0.0]
        self._arm_horizontal_joints = [
            "joint_arm_l3",
            "joint_arm_l2",
            "joint_arm_l1",
            "joint_arm_l0",
        ]
        for arm_joint_name in self._arm_horizontal_joints:
            joint_idx = self.arm_joints.index(self.joint_from_name(arm_joint_name))
            joints[joint_idx] = self.joint_upper_limits[joint_idx]
            self.set_joints(joints)
            joint_max_pose = self._get_relative_wrist_pose()
            # Only the z dimension should be changing.
            assert np.allclose(min_pose.position[:2], joint_max_pose.position[:2])
            assert np.allclose(min_pose.orientation, joint_max_pose.orientation)
            self._arm_horizontal_bounds.append(
                min_pose.position[2] - joint_max_pose.position[2]
            )

        # Determine reachability bounds for the arm moving up and down.
        joints = list(self.joint_lower_limits)
        self.set_joints(joints)
        self._arm_vertical_bounds = [0.0]
        arm_joint_name = "joint_lift"
        joint_idx = self.arm_joints.index(self.joint_from_name(arm_joint_name))
        joints[joint_idx] = self.joint_upper_limits[joint_idx]
        self.set_joints(joints)
        joint_max_pose = self._get_relative_wrist_pose()
        # Only the y dimension should be changing, but there is small error.
        assert np.isclose(min_pose.position[0], joint_max_pose.position[0], atol=1e-3)
        assert np.isclose(min_pose.position[2], joint_max_pose.position[2], atol=1e-3)
        assert np.allclose(min_pose.orientation, joint_max_pose.orientation, atol=1e-3)
        self._arm_vertical_bounds.append(
            min_pose.position[1] - joint_max_pose.position[1]
        )

        # Use an approximate Cartesian offset between the end effector and the wrist.
        self.set_joints(self.joint_lower_limits)
        ee_pose = self.get_end_effector_pose()
        wrist_pose = get_link_pose(
            self.robot_id, self._wrist_id, self.physics_client_id
        )
        self._wrist_ee_offset = Pose(
            tuple(np.subtract(wrist_pose.position, ee_pose.position))
        )

        # Reset the joints after 'calibration'.
        self.set_joints(self._home_joint_positions)

    @classmethod
    def get_name(cls) -> str:
        return "stretch"

    @property
    def default_urdf_path(self) -> Path:
        pkg = Path(str(importlib_resources.files("stretch_urdf")))
        filepath = pkg / "SE3" / "stretch_description_SE3_eoa_wrist_dw3_tool_sg3.urdf"
        with open(filepath, "r", encoding="utf-8") as f:
            urdf_str = f.read()
        # The original URDF file has lower == upper == 0 for the fingers, which
        # breaks some assumptions and prevents moving the fingers.
        urdf_str = urdf_str.replace(
            '<limit effort="0" lower="0" upper="0" velocity="0"/>',
            '<limit effort="0" lower="0" upper="0.5" velocity="0"/>',
        )
        # NOTE: it is unfortunately necessary to write in the same directory
        # as the original URDF file because otherwise the STL files won't be
        # found (because they are defined relatively in the URDF).
        new_filepath = filepath.parent / (filepath.stem + "-PYBULLET-HELPERS.urdf")
        with open(new_filepath, mode="w", encoding="utf-8") as new_f:
            new_f.write(urdf_str)
        return new_filepath

    @property
    def default_home_joint_positions(self) -> JointPositions:
        return [0.65, 0.025, 0.025, 0.04, 0.0157, 0.0, 0.0, 0.0, 0.0, 0.0]

    @property
    def end_effector_name(self) -> str:
        return "joint_gripper_s3_body"

    @property
    def tool_link_name(self) -> str:
        return "link_gripper_s3_body"

    @property
    def finger_joint_names(self) -> list[str]:
        return ["joint_gripper_finger_right", "joint_gripper_finger_left"]

    @property
    def open_fingers_state(self) -> float:
        return 0.5

    @property
    def closed_fingers_state(self) -> float:
        return 0.0

    def finger_state_to_joints(self, state: float) -> list[float]:
        return [state, state]

    def joints_to_finger_state(self, joint_positions: list[float]) -> float:
        assert len(joint_positions) == 2
        assert np.isclose(joint_positions[0], joint_positions[1])
        return joint_positions[0]

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

        # Start by transforming the end effector pose into the robot frame.
        initial_joints = self.get_joint_positions()

        self.set_joints(self.joint_lower_limits)
        world_to_nominal = get_link_pose(
            self.robot_id, self._wrist_id, self.physics_client_id
        )
        world_to_wrist_target = multiply_poses(self._wrist_ee_offset, end_effector_pose)
        nominal_to_wrist_target = multiply_poses(
            world_to_nominal.invert(), world_to_wrist_target
        )

        solution = list(self.joint_lower_limits)

        # First determine vertical lift, the easier case.
        assert len(self._arm_vertical_bounds) == 2
        wrist_vert = nominal_to_wrist_target.position[1]
        min_vert, max_vert = self._arm_vertical_bounds
        vert_scale = max_vert - min_vert
        frac_vert = np.clip((wrist_vert - min_vert) / vert_scale, 0, 1)
        arm_joint_name = "joint_lift"
        joint_idx = self.arm_joints.index(self.joint_from_name(arm_joint_name))
        joint_lower = self.joint_lower_limits[joint_idx]
        joint_upper = self.joint_upper_limits[joint_idx]
        joint_val = joint_lower + frac_vert * (joint_upper - joint_lower)
        solution[joint_idx] = joint_val

        # Next determine the horizontal joint values by completely extending
        # each until the target is reachable.
        wrist_horiz = nominal_to_wrist_target.position[2]
        num_horiz_joints = len(self._arm_horizontal_bounds)
        for i, arm_joint_name in enumerate(self._arm_horizontal_joints):
            min_horiz = self._arm_horizontal_bounds[i]
            max_horiz = self._arm_horizontal_bounds[i + 1]
            joint_idx = self.arm_joints.index(self.joint_from_name(arm_joint_name))
            joint_lower = self.joint_lower_limits[joint_idx]
            joint_upper = self.joint_upper_limits[joint_idx]
            if wrist_horiz > max_horiz and i < num_horiz_joints - 1:
                solution[joint_idx] = joint_upper
            else:
                horiz_scale = max_horiz - min_horiz
                frac_horiz = np.clip((wrist_horiz - min_horiz) / horiz_scale, 0, 1)
                joint_val = joint_lower + frac_horiz * (joint_upper - joint_lower)
                solution[joint_idx] = joint_val
                break

        # Directly set the wrist roll, pitch, yaw from the target orientation.
        world_to_wrist = get_link_pose(
            self.robot_id, self._wrist_id, self.physics_client_id
        )
        world_to_target = end_effector_pose
        wrist_to_target = multiply_poses(world_to_wrist.invert(), world_to_target)
        wrist_target_rot = Rotation.from_quat(wrist_to_target.orientation)
        roll, neg_pitch, yaw = wrist_target_rot.as_euler("zxy")
        pitch = -neg_pitch

        wrist_angle_names = ["joint_wrist_roll", "joint_wrist_pitch", "joint_wrist_yaw"]
        for v, name in zip([roll, pitch, yaw], wrist_angle_names, strict=True):
            joint_idx = self.arm_joints.index(self.joint_from_name(name))
            solution[joint_idx] = v

        self.set_joints(solution)

        # NOTE: this leads to almost no improvements. Optimizing with pybullet
        # IK also didn't help. Leaving it in the code in case someone wants
        # to improve.

        # Now that we've initialized to a good position, try to optimize to a
        # better solution.
        # target_pos = end_effector_pose.position
        # target_rot = Rotation.from_quat(end_effector_pose.orientation)

        # def opt_fn(x: JointPositions) -> float:
        #     self.set_joints(x)
        #     pose = self.get_end_effector_pose()
        #     pos_err = np.sum(np.square(np.subtract(pose.position, target_pos)))
        #     rot = Rotation.from_quat(pose.orientation)
        #     rot_diff = rot.inv() * target_rot
        #     orn_err = rot_diff.magnitude()
        #     return pos_err + orn_err

        # res = minimize(opt_fn, solution, method="L-BFGS-B",
        #   bounds=list(zip(self.joint_lower_limits, self.joint_upper_limits)),
        #                options={"disp": False, "gtol": 1e-8})
        # final_solution = res.x
        # print("Improvement:", opt_fn(solution) - opt_fn(final_solution))

        self.set_joints(initial_joints)
        return solution

    def _get_relative_wrist_pose(self) -> Pose:
        """Get the pose of the wrist."""
        return get_relative_link_pose(
            self.robot_id,
            -1,
            self._wrist_id,
            physics_client_id=self.physics_client_id,
        )
