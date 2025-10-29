"""Utilities for working with Gymnasium environments."""

from abc import ABC, abstractmethod
from typing import Any, NamedTuple

import gymnasium as gym
import numpy as np
import prbench
from prbench.envs.geom2d.stickbutton2d import StickButton2DEnv
import torch as th
from gymnasium import spaces

# Environment wrappers

def make_env_ppo(
    env_id: str,
    idx: int,
    capture_video: bool,
    run_name: str,
    max_episode_steps: int,
    gamma: float = 0.99,
):
    """Create a single environment instance with appropriate wrappers for ppo."""

    def thunk():
        if capture_video and idx == 0:
            if "prbench" in env_id:
                env = prbench.make(env_id, render_mode="rgb_array")
            else:
                env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            if "prbench" in env_id:
                env = prbench.make(env_id)
            else:
                env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        # NOTE: PRBench by default has infinite horizon, so we set a time limit here
        if "prbench" in env_id:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        return env

    return thunk


def make_env_sac(
    env_id: str,
    idx: int,
    capture_video: bool,
    run_name: str,
    max_episode_steps: int,
):
    """Create a single environment instance with appropriate wrappers for sac."""

    def thunk():
        if capture_video and idx == 0:
            if "prbench" in env_id:
                env = prbench.make(env_id, render_mode="rgb_array")
            else:
                env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            if "prbench" in env_id:
                env = prbench.make(env_id)
            else:
                env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # NOTE: PRBench by default has infinite horizon, so we set a time limit here
        if "prbench" in env_id:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        return env

    return thunk


def get_device(device: th.device | str = "auto") -> th.device:
    """Retrieve PyTorch device. It checks that the requested device is available first.
    For now, it supports only cpu and cuda. By default, it tries to use the gpu.

    :param device: One for 'auto', 'cuda', 'cpu'
    :return: Supported Pytorch device
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to th.device
    device = th.device(device)

    # Cuda not available
    if device.type == th.device("cuda").type and not th.cuda.is_available():
        return th.device("cpu")

    return device

# Replay buffers

class ReplayBufferSamples(NamedTuple):
    """Samples from the replay buffer."""

    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor


class BaseBuffer(ABC):
    """Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    observation_space: spaces.Space
    obs_shape: tuple[int, ...]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        assert isinstance(
            action_space, spaces.Box
        ), "Only continuous action space is supported for the base buffer"
        assert isinstance(
            observation_space, spaces.Box
        ), "Only continuous action space is supported for the base buffer"
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = observation_space.shape
        self.action_dim = int(np.prod(action_space.shape))
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)

        to convert shape from [n_steps, n_envs, ...] (when ... is the shape
        of the features) to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        """Add elements to the buffer."""
        raise NotImplementedError()

    def extend(self, *args) -> None:
        """Add a new batch of transitions to the buffer."""
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """Reset the buffer."""
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int):
        """
        :param batch_size: Number of element to sample
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds)

    @abstractmethod
    def _get_samples(self, batch_inds: np.ndarray) -> ReplayBufferSamples:
        """
        :param batch_inds:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid
            changing things by reference). This argument is inoperative if the
            device is not the CPU.
        :return:
        """
        if copy:
            return th.tensor(array, device=self.device)
        return th.as_tensor(array, device=self.device)


class ReplayBuffer(BaseBuffer):
    """Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        which reduces by almost a factor two the memory used, at a cost of more
        complexity. See https://github.com/DLR-RM/stable-
        baselines3/issues/37#issuecomment-637501195 and https://github.com/DLR-
        RM/stable-baselines3/pull/28#issuecomment-637559274 Cannot be used in
        combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task. https://github.com/DLR-
        RM/stable-baselines3/issues/284
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs
        )

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # there is a bug if both optimize_memory_usage and
        # handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros(
            (self.buffer_size, self.n_envs, *self.obs_shape),
            dtype=observation_space.dtype,
        )

        if not optimize_memory_usage:
            # When optimizing memory, `observations` contains also the next observation
            self.next_observations = np.zeros(
                (self.buffer_size, self.n_envs, *self.obs_shape),
                dtype=observation_space.dtype,
            )

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim),
            dtype=self._maybe_cast_dtype(action_space.dtype),
        )

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
        else:
            self.next_observations[self.pos] = np.array(next_obs)

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array(
                [info.get("TimeLimit.truncated", False) for info in infos]
            )

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (
                np.random.randint(1, self.buffer_size, size=batch_size) + self.pos
            ) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds: np.ndarray) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self.observations[
                (batch_inds + 1) % self.buffer_size, env_indices, :
            ]
        else:
            next_obs = self.next_observations[batch_inds, env_indices, :]

        data = (
            self.observations[batch_inds, env_indices, :],
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (
                self.dones[batch_inds, env_indices]
                * (1 - self.timeouts[batch_inds, env_indices])
            ).reshape(-1, 1),
            self.rewards[batch_inds, env_indices].reshape(-1, 1),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    @staticmethod
    def _maybe_cast_dtype(dtype: np.typing.DTypeLike) -> np.typing.DTypeLike:
        """Cast `np.float64` action datatype to `np.float32`, keep the others dtype
        unchanged. See GH#1572 for more information.

        :param dtype: The original action space dtype
        :return: ``np.float32`` if the dtype was float64,
            the original dtype otherwise.
        """
        if dtype == np.float64:
            return np.float32
        return dtype


# Simple Sanity Check Envs
# A custom environment wrapper that fixes positions in StickButton2DEnv.
# NOTE: This env will by default truncate after 100 steps
# so it is not registered with "prbench", but with gymnasium directly.
class FixedPositionWrapper(gym.Env):
    """Environment wrapper that fixes initial positions for testing."""

    def __init__(self, env: StickButton2DEnv):
        super().__init__()
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.render_mode = env.render_mode
        self.metadata = env.metadata
        obs0, _ = self.env.reset(seed=123)
        # Check if the observation space has devectorize method
        assert hasattr(self.env.observation_space, "devectorize")
        state0 = self.env.observation_space.devectorize(obs0)

        obj_name_to_obj = {o.name: o for o in list(state0.data.keys())}
        robot = obj_name_to_obj["robot"]
        button0 = obj_name_to_obj["button0"]

        state1 = state0.copy()
        state1.set(robot, "x", 1.8)
        state1.set(robot, "y", 1.0)
        state1.set(button0, "y", 1.0)
        state1.set(button0, "x", 2.0)
        self.reset_options = {"init_state": state1}
        self.num_env_steps = 0
        self.max_episode_steps = 100
        self.r = 0.0
        # Debug rendering only if render_mode is set
        # if self.render_mode is not None:
        #     _, _ = env.reset(seed=123, options=self.reset_options)
        #     img = env.render()
        #     os.makedirs("debug", exist_ok=True)
        #     iio.imwrite("debug/unit_test_fixed_env_init.png", img)

    def reset(self, seed=None, options=None):  # pylint: disable=arguments-differ
        del seed, options  # Ignore external parameters
        self.num_env_steps = 0
        self.r = 0.0
        obs, info = self.env.reset(seed=123, options=self.reset_options)
        return obs, info

    def step(self, action):
        self.num_env_steps += 1
        obs, reward, terminated, _, info = self.env.step(action)
        truncated = self.num_env_steps >= self.max_episode_steps
        self.r += reward
        if terminated or truncated:
            info["final_info"] = [
                {
                    "episode": {
                        "r": self.r,
                        "l": self.num_env_steps - 1,
                    }
                }
            ]
            obs, _ = self.reset()
        return obs, reward, terminated, truncated, info

    def close(self):
        return self.env.close()

    def render(self):
        return self.env.render()

# Register the wrapped environment with a custom ID so PPO can create it
def make_fixed_env(render_mode=None):
    """Factory function to create the fixed environment."""
    base_env = prbench.make(
        "prbench/StickButton2D-b1-v0",
        render_mode=render_mode,
    )
    return FixedPositionWrapper(base_env)