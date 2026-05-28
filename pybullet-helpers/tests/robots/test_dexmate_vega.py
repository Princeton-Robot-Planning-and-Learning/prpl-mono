"""Tests for the Dexmate Vega robot."""

import importlib.util
import warnings

import numpy as np
import pytest

from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    inverse_kinematics,
)
from pybullet_helpers.robots import _dexmate_vega_ik as _vega_ik
from pybullet_helpers.robots import dexmate_vega
from pybullet_helpers.robots.dexmate_vega import DexmateVega1UPyBulletRobot

EAIK_INSTALLED = importlib.util.find_spec("eaik") is not None


def test_dexmate_vega_1u_robot(physics_client_id):
    """Tests for DexmateVega1UPyBulletRobot."""
    robot = DexmateVega1UPyBulletRobot(physics_client_id)
    assert robot.get_name() == "dexmate-vega-1u"
    assert robot.arm_joint_names == [
        "L_arm_j1",
        "L_arm_j2",
        "L_arm_j3",
        "L_arm_j4",
        "L_arm_j5",
        "L_arm_j6",
        "L_arm_j7",
    ]
    assert np.allclose(robot.action_space.low, robot.joint_lower_limits)
    assert np.allclose(robot.action_space.high, robot.joint_upper_limits)
    # Moving each joint to its midpoint produces an EE pose within reach
    # (forward_kinematics doesn't raise).
    for i in range(len(robot.arm_joints)):
        q = list(robot.home_joint_positions)
        q[i] = 0.5 * (robot.joint_lower_limits[i] + robot.joint_upper_limits[i])
        robot.forward_kinematics(q)


@pytest.mark.skipif(not EAIK_INSTALLED, reason="EAIK not installed")
def test_dexmate_vega_1u_ik_roundtrip(physics_client_id):
    """IK roundtrip: random q -> FK -> custom IK -> FK -> matches."""
    robot = DexmateVega1UPyBulletRobot(physics_client_id)
    assert robot.default_inverse_kinematics_method == "custom"

    rng = np.random.default_rng(7)
    lo = np.array(robot.joint_lower_limits)
    hi = np.array(robot.joint_upper_limits)
    n_trials = 5
    n_success = 0
    for _ in range(n_trials):
        q_true = rng.uniform(lo, hi)
        target = robot.forward_kinematics(q_true.tolist())
        try:
            q_sol = inverse_kinematics(
                robot, target, validate=True, validation_atol=1e-3
            )
        except InverseKinematicsError:
            continue
        recovered = robot.forward_kinematics(q_sol)
        pos_err = np.linalg.norm(
            np.array(recovered.position) - np.array(target.position)
        )
        if pos_err < 1e-3:
            n_success += 1
    # Allow one failure to absorb the rare Nelder-Mead local-min case.
    assert (
        n_success >= n_trials - 1
    ), f"only {n_success}/{n_trials} IK roundtrips succeeded"


def test_dexmate_vega_1u_eaik_fallback_warning(physics_client_id, monkeypatch):
    """If EAIK is unavailable, the robot falls back to pybullet IK and emits a one-time
    RuntimeWarning explaining how to install EAIK."""
    monkeypatch.setattr(_vega_ik, "EAIK_AVAILABLE", False)
    monkeypatch.setattr(dexmate_vega, "_EAIK_FALLBACK_WARNED", False)
    robot = DexmateVega1UPyBulletRobot(physics_client_id)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert robot.default_inverse_kinematics_method == "pybullet"
        assert robot.default_inverse_kinematics_method == "pybullet"  # second call
    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(runtime_warnings) == 1
    assert "EAIK" in str(runtime_warnings[0].message)
