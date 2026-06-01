"""Unit tests for URDF loading and forward-kinematics correctness."""

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from prpl_kinematics.loading import load_urdf
from prpl_kinematics.tree.joints import FixedJoint, PrismaticJoint, RevoluteJoint
from prpl_kinematics.utils import get_assets_path


def _panda_path() -> str:
    return str(get_assets_path() / "urdf" / "panda_arm_hand.urdf")


def test_load_urdf_structure():
    """Loading Panda yields the expected root, actuated joints, and joint types."""
    tree = load_urdf(_panda_path())
    assert tree.root == "panda_link0"
    assert tree.actuated_joint_names() == [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]
    assert isinstance(tree.joint("panda_joint1"), RevoluteJoint)
    assert isinstance(tree.joint("panda_finger_joint1"), PrismaticJoint)
    assert isinstance(tree.joint("panda_hand_joint"), FixedJoint)


def test_forward_kinematics_matches_pybullet(physics_client_id):
    """Tree FK agrees with PyBullet's link poses across random configurations."""
    path = _panda_path()
    tree = load_urdf(path)
    body = p.loadURDF(path, useFixedBase=True, physicsClientId=physics_client_id)
    joint_index = {}
    link_index = {}
    for i in range(p.getNumJoints(body, physicsClientId=physics_client_id)):
        info = p.getJointInfo(body, i, physicsClientId=physics_client_id)
        joint_index[info[1].decode()] = i
        link_index[info[12].decode()] = i

    rng = np.random.default_rng(123)
    for _ in range(5):
        config = {}
        for name in tree.actuated_joint_names():
            joint = tree.joint(name)
            lo = max(joint.lower_limits[0], -3.0)
            hi = min(joint.upper_limits[0], 3.0)
            value = float(rng.uniform(lo, hi))
            config[name] = [value]
            p.resetJointState(
                body, joint_index[name], value, physicsClientId=physics_client_id
            )
        for link_name, index in link_index.items():
            ours = tree.forward_kinematics(link_name, config)
            state = p.getLinkState(
                body,
                index,
                computeForwardKinematics=True,
                physicsClientId=physics_client_id,
            )
            assert np.allclose(ours.t, state[4], atol=1e-5)
            pybullet_rotation = Rotation.from_quat(state[5]).as_matrix()
            assert np.allclose(ours.R, pybullet_rotation, atol=1e-5)
