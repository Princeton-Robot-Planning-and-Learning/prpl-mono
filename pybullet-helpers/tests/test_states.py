"""Tests for states.py."""

import numpy as np
import pybullet as p

from pybullet_helpers.geometry import Pose, get_pose, set_pose
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.states import KinematicState
from pybullet_helpers.utils import create_pybullet_block


def test_kinematic_state():
    """Tests for KinematicState()."""

    physics_client_id = p.connect(p.DIRECT)

    robot = create_pybullet_robot("panda", physics_client_id)
    block = create_pybullet_block((0, 0, 0, 1), (0.1, 0.1, 0.1), physics_client_id)
    table = create_pybullet_block((0, 0, 0, 1), (0.5, 0.5, 0.5), physics_client_id)
    robot_init_joints = robot.get_joint_positions()
    block_init_pose = Pose((0.0, 0.0, 0.0))
    table_init_pose = Pose((0.0, 0.0, -0.3))
    set_pose(block, block_init_pose, physics_client_id)
    set_pose(table, table_init_pose, physics_client_id)

    state = KinematicState.from_pybullet(robot, {block, table})
    assert np.allclose(state.robot_joints, robot_init_joints)
    assert not state.attachments
    assert set(state.object_poses) == {block, table}
    assert block_init_pose.allclose(state.object_poses[block])
    assert table_init_pose.allclose(state.object_poses[table])

    end_effector_pose = robot.get_end_effector_pose()
    set_pose(block, end_effector_pose, physics_client_id)

    state2 = KinematicState.from_pybullet(
        robot, {block, table}, attached_object_ids={block}
    )
    assert np.allclose(state2.robot_joints, robot_init_joints)
    assert set(state2.attachments) == {block}
    assert Pose.identity().allclose(state2.attachments[block])
    assert set(state2.object_poses) == {block, table}
    assert end_effector_pose.allclose(state2.object_poses[block])
    assert table_init_pose.allclose(state2.object_poses[table])

    state.set_pybullet(robot)
    assert get_pose(block, physics_client_id).allclose(block_init_pose)
