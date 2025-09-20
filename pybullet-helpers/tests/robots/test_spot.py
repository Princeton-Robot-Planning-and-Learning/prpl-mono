"""Tests for spot.py."""

import numpy as np

from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.inverse_kinematics import inverse_kinematics
from pybullet_helpers.robots.spot import (
    SpotPyBulletRobot,
)


def test_spot_pybullet_robot(physics_client_id):
    """Tests for SpotPyBulletRobot()."""
    robot = SpotPyBulletRobot(
        physics_client_id,
        fixed_base=False,
    )
    assert robot.get_name() == "spot"
    assert robot.arm_joint_names == [
        "arm_sh0",
        "arm_sh1",
        "arm_el0",
        "arm_el1",
        "arm_wr0",
        "arm_wr1",
        "arm_f1x",
    ]
    assert np.allclose(robot.action_space.low, robot.joint_lower_limits)
    assert np.allclose(robot.action_space.high, robot.joint_upper_limits)

    # Tests for IK. These values were found through the joint visualizer script.
    retract_ee_pose = Pose(
        position=(0.5536779165267944, -0.003691227175295353, 0.946405827999115),
        orientation=(
            0.006573572289198637,
            0.7132630944252014,
            -0.0067964717745780945,
            0.7008326053619385,
        ),
    )
    raised_ee_pose = Pose(
        position=(0.5952516198158264, -0.12355716526508331, 1.2207549810409546),
        orientation=(
            0.23999500274658203,
            0.6325281262397766,
            -0.06981812417507172,
            0.7331002950668335,
        ),
    )
    right_ee_pose = Pose(
        position=(0.41684195399284363, -0.4368157684803009, 1.1002354621887207),
        orientation=(
            -0.2101231813430786,
            -0.634379506111145,
            0.703512966632843,
            -0.24182717502117157,
        ),
    )

    for base_pose_offset in [
        Pose((0, 0, 0)),
        Pose((0, 1, 0)),
        Pose.from_rpy((0, 0, 0), (np.pi / 2, 0, 0)),
    ]:
        robot.set_base(base_pose_offset)
        for name, pose in [
            ("retract", retract_ee_pose),
            ("raised", raised_ee_pose),
            ("right", right_ee_pose),
        ]:
            offset_target_pose = multiply_poses(base_pose_offset, pose)
            joint_positions = inverse_kinematics(
                robot, end_effector_pose=offset_target_pose, validate=True
            )
            robot.set_joints(joint_positions)
            recovered_pose = robot.forward_kinematics(joint_positions)
            # IK is analytic, so this should be extremely close.
            assert np.allclose(
                recovered_pose.position, offset_target_pose.position, atol=1e-6
            ), f"IK failed for {name} with offset {base_pose_offset}"
