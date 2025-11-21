"""Basic tests for the TidyBot3D environment observation and action space validity,
step, and reset."""

from relational_structs import ObjectCentricState

from prbench.envs.dynamic3d.object_types import MujocoObjectTypeFeatures
from prbench.envs.dynamic3d.tidybot3d import ObjectCentricTidyBot3DEnv


def test_tidybot3d_observation_space():
    """Test that the observation returned by TidyBot3DEnv.reset() is within the
    observation space."""
    env = ObjectCentricTidyBot3DEnv(num_objects=3, render_images=False)
    obs = env.reset()[0]
    assert env.observation_space.contains(obs), "Observation not in observation space"
    env.close()


def test_tidybot3d_action_space():
    """Test that a sampled action is within the TidyBot3DEnv action space."""
    env = ObjectCentricTidyBot3DEnv(num_objects=3, render_images=False)
    action = env.action_space.sample()
    assert env.action_space.contains(action), "Action not in action space"
    env.close()


def test_tidybot3d_step():
    """Test that stepping the environment leads to some nontrivial change."""
    env = ObjectCentricTidyBot3DEnv(num_objects=3, render_images=False)
    obs, _ = env.reset()
    action = env.action_space.sample()
    next_obs, _, _, _, _ = env.step(action)
    assert not obs.allclose(next_obs, atol=1e-6)
    env.close()


def test_tidybot3d_reset_returns_valid_observation():
    """Test that reset() returns an observation in the observation space."""
    env = ObjectCentricTidyBot3DEnv(num_objects=3, render_images=False)
    obs, info = env.reset()
    assert env.observation_space.contains(
        obs
    ), "Reset observation not in observation space"
    assert isinstance(info, dict)
    env.close()


def test_tidybot3d_reset_returns_valid_observation_with_rendering():
    """Test that reset() returns an observation in the observation space when rendering
    is enabled."""
    env = ObjectCentricTidyBot3DEnv(num_objects=3, render_images=True)
    obs, info = env.reset()
    assert env.observation_space.contains(
        obs
    ), "Reset observation not in observation space"
    assert isinstance(info, dict)
    env.close()


def test_tidybot3d_step_returns_valid_outputs():
    """Test that step() returns valid outputs: obs in space, reward is float, done flags
    are bools."""
    env = ObjectCentricTidyBot3DEnv(num_objects=3, render_images=False)
    env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(
        obs
    ), "Step observation not in observation space"
    assert isinstance(reward, float), "Reward is not a float"
    assert isinstance(terminated, bool), "Terminated is not a bool"
    assert isinstance(truncated, bool), "Truncated is not a bool"
    assert isinstance(info, dict), "Info is not a dict"
    env.close()


def test_tidybot3d_get_object_pos_quat():
    """Test that get_object_pos_quat() returns valid position and orientation."""
    env = ObjectCentricTidyBot3DEnv(num_objects=3, render_images=False)
    env.reset()
    for obj in env._objects:  # pylint: disable=protected-access
        pos, quat = (
            env._robot_env.get_joint_pos_quat(  # pylint: disable=protected-access
                obj.joint_name
            )
        )
        assert len(pos) == 3, "Position should have 3 elements"
        assert len(quat) == 4, "Quaternion should have 4 elements"
    env.close()


def test_tidybot3d_set_get_object_pos_quat_consistency():
    """Test that setting and then getting an object's position and orientation is
    consistent."""
    env = ObjectCentricTidyBot3DEnv(num_objects=3, render_images=False)
    env.reset()
    for obj in env._objects:  # pylint: disable=protected-access
        original_pos, original_quat = (
            env._robot_env.get_joint_pos_quat(  # pylint: disable=protected-access
                obj.joint_name
            )
        )
        new_pos = [p + 0.1 for p in original_pos]
        new_quat = original_quat  # Keep orientation the same for simplicity
        env._robot_env.set_joint_pos_quat(  # pylint: disable=protected-access
            obj.joint_name, new_pos, new_quat
        )
        updated_pos, updated_quat = (
            env._robot_env.get_joint_pos_quat(  # pylint: disable=protected-access
                obj.joint_name
            )
        )
        assert all(
            abs(o - u) < 1e-5 for o, u in zip(new_pos, updated_pos)
        ), "Position not set correctly"
        assert all(
            abs(o - u) < 1e-5 for o, u in zip(new_quat, updated_quat)
        ), "Orientation not set correctly"
    env.close()


def test_tidybot3d_object_centric_data():
    """Test that mujoco objects' get_object_centric_data() returns a valid
    ObjectCentricState."""
    env = ObjectCentricTidyBot3DEnv(num_objects=3, render_images=False)
    env.reset()
    for obj in env._objects:  # pylint: disable=protected-access
        data = obj.get_object_centric_data()
        assert isinstance(data, dict), "Object-centric data should be a dict"
        object_type = obj.symbolic_object.type
        expected_keys = set(MujocoObjectTypeFeatures[object_type])
        assert expected_keys.issubset(
            data.keys()
        ), f"Data keys missing, expected at least {expected_keys}"
    env.close()


def test_tidybot3d_env_object_centric_state():
    """Test that the environment's observation includes valid object-centric states."""
    num_objects = 3
    env = ObjectCentricTidyBot3DEnv(num_objects=num_objects, render_images=False)
    obs, _ = env.reset()
    object_centric_state = obs
    assert isinstance(
        object_centric_state, ObjectCentricState
    ), "Object-centric state should be a dict"
    assert (
        len(object_centric_state.data) == num_objects + 1  # plus one for robot
    ), "Incorrect number of objects in state"
    for obj, state in object_centric_state.data.items():
        assert len(state) == len(
            MujocoObjectTypeFeatures[obj.type]
        ), "State vector length mismatch"
    env.close()


def test_tidybot3d_env_set_state():
    """Test that the state of the environment can be consistently reset."""
    # Generate a random trajectory.
    states = []
    actions = []
    env = ObjectCentricTidyBot3DEnv(num_objects=3, render_images=False)
    obs, _ = env.reset(seed=123)
    states.append(obs)
    for _ in range(5):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        actions.append(action)
        states.append(obs)

    # First just try resetting each state in the trajectory.
    for state in states:
        env.set_state(state)
        recovered_state = (
            env._get_object_centric_state()  # pylint: disable=protected-access
        )
        assert state.allclose(recovered_state)

    # Now also try resetting to an intermediate state (with nonzero velocity) and make
    # sure that the trajectory is still reproducible from there.
    start_idx = 1
    env.set_state(states[1])
    for i in range(start_idx, len(actions)):
        recovered_state, _, _, _, _ = env.step(actions[i])
        assert states[i + 1].allclose(recovered_state)

    env.close()
