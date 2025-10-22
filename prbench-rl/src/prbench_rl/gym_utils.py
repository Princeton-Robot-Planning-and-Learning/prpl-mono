"""General wrapper for environments supporting improvisational policies."""

import os
from collections.abc import Callable, Sequence
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium import logger
from gymnasium.vector.utils import batch_space
from gymnasium.wrappers.monitoring import video_recorder
from gymnasium.wrappers.record_video import capped_cubic_video_schedule

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class MultiEnvWrapper(gym.Env):
    """A batched single-Env wrapper over multiple Gym environments.

    This class exposes N identical sub-environments as ONE `gym.Env` whose
    observation_space and action_space are the batched versions of the
    single-env spaces (via `batch_space`). This makes it compatible with
    wrappers like `gymnasium.wrappers.RecordVideo` that expect `gym.Env`,
    while still enabling vectorized stepping.

    It supports optional PyTorch tensor IO for Deep RL training, and a
    tiled `rgb_array` render for video recording.

    Args:
        env_fn: A callable that creates a single environment instance
        num_envs: Number of sub-environments to create
        auto_reset: Whether to automatically reset terminated environments
            (default: True)
        to_tensor: If True, observations and returns will be converted to PyTorch
            tensors, and tensor actions will be accepted (default: False)
        device: Device to place tensors on if to_tensor=True (default: "cpu")
        render_mode: Render mode; should be "rgb_array" to use RecordVideo

    Example:
        >>> import prbench
        >>> prbench.register_all_environments()
        >>> env_fn = lambda: prbench.make(
        ...     "prbench/StickButton2D-b5-v0", render_mode="rgb_array")
        >>> multi_env = MultiEnvWrapper(env_fn, num_envs=4, render_mode="rgb_array")
        >>> obs_batch, info_batch = multi_env.reset(seed=123)
        >>> obs_batch.shape
        (4, observation_dim)
        >>> actions = multi_env.action_space.sample()
        >>> obs_batch, rewards, terminated, truncated, info_batch = (
        ...     multi_env.step(actions))

        With tensor support:
        >>> multi_env = MultiEnvWrapper(
        ...     env_fn, num_envs=4, to_tensor=True, device="cuda",
        ...     render_mode="rgb_array")
        >>> obs_batch, _ = multi_env.reset()  # returns torch.Tensor on cuda
        >>> actions = torch.randn(
        ...     (4, *multi_env.single_action_space.shape), device="cuda")
        >>> obs, rewards, done, truncated, _ = multi_env.step(actions)
    """

    # Make sure RecordVideo recognizes rgb_array rendering
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        num_envs: int,
        max_episode_steps: int | None = None,
        auto_reset: bool = True,
        to_tensor: bool = False,
        device: str = "cpu",
        render_mode: str | None = "rgb_array",
    ):

        super().__init__()
        self.env_fn = env_fn
        self.num_envs = int(num_envs)
        assert self.num_envs >= 1, "num_envs must be >= 1"
        self.auto_reset = auto_reset
        self.to_tensor = to_tensor
        self.device = device
        self.render_mode = render_mode

        # Create all sub-environments
        # TIP: Prefer env_fn that accepts render_mode="rgb_array" for recording.
        self.envs = [env_fn() for _ in range(self.num_envs)]

        # Spaces
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space
        assert isinstance(
            self.single_observation_space, gym.spaces.Box
        ), "Only Box observation space is supported"
        self.observation_space = batch_space(
            self.single_observation_space, self.num_envs
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        # Validate homogeneous spaces
        for i, env in enumerate(self.envs):
            assert env.action_space == self.single_action_space, (
                f"Environment {i} has different action space: {env.action_space} "
                f"vs expected {self.single_action_space}"
            )
            assert env.observation_space == self.single_observation_space, (
                f"Environment {i} has different observation space: "
                f"{env.observation_space} vs expected {self.single_observation_space}"
            )

        # Buffers
        self._observations = np.zeros(
            (self.num_envs,) + self.single_observation_space.shape,
            dtype=self.single_observation_space.dtype,
        )
        self._rewards = np.zeros((self.num_envs,), dtype=np.float32)
        self._terminations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._env_needs_reset = np.ones((self.num_envs,), dtype=np.bool_)

        # Copy metadata and annotate autoreset status
        self.metadata = dict(getattr(self.envs[0], "metadata", {}))
        self.metadata["render_modes"] = list(
            set(self.metadata.get("render_modes", []) + ["rgb_array"])
        )
        self.metadata["autoreset_mode"] = "next_step" if auto_reset else "disabled"

        elapsed_steps = np.zeros((self.num_envs,), dtype=np.int32)
        self.elapsed_steps = self._to_tensor(elapsed_steps)
        self._max_episode_steps = max_episode_steps
        if max_episode_steps is not None:
            print(
                "Warning: max_episode_steps is now enforced by "
                "MultiEnvWrapper, will ignore per env truncation."
            )

    # ------------------------- Utilities -------------------------

    def _to_tensor(self, array: np.ndarray) -> np.ndarray | torch.Tensor:
        if self.to_tensor:
            return torch.from_numpy(array).to(self.device)
        return array

    def _to_numpy(self, data: np.ndarray | torch.Tensor) -> np.ndarray:
        if torch.is_tensor(data):
            return data.detach().cpu().numpy()
        return data

    # --------------------------- API -----------------------------

    def reset(
        self, *, seed: int | Sequence[int] | None = None, options: dict | None = None
    ) -> tuple[np.ndarray | torch.Tensor, dict]:
        """Reset all sub-environments and return batched observation and info."""
        # Distribute seeds
        if seed is not None:
            if isinstance(seed, int):
                seeds_final: list[int | None] = [seed + i for i in range(self.num_envs)]
            else:
                seed = list(seed)
                assert (
                    len(seed) == self.num_envs
                ), f"Seed list length {len(seed)} doesn't match num_envs {self.num_envs}"
                seeds_final = seed  # type: ignore
        else:
            seeds_final = [None] * self.num_envs

        # Reset
        infos: dict[str, Any] = {}
        for i, (env, env_seed) in enumerate(zip(self.envs, seeds_final)):
            # NOTE: Need to handle reset init_states properly here
            # for each sub-env. we assume the init_states must be
            # provided as a batch of states for all sub-envs.
            local_options = None
            if options is not None:
                local_options = dict(options)
                if "init_state" in options.keys():
                    assert (
                        isinstance(options["init_state"], (np.ndarray, torch.Tensor))
                        and options["init_state"].shape[0] == self.num_envs
                    ), (
                        "If providing init_state in options, it must be a "
                        "batch of states for all sub-envs"
                    )
                    local_options = dict(options)
                    local_options["init_state"] = self._to_numpy(
                        options["init_state"][i]
                    )

            obs, info = env.reset(seed=env_seed, options=local_options)
            # Write obs into buffer
            self._observations[i] = obs

            # Batch info
            for key, value in info.items():
                if key not in infos:
                    infos[key] = [None] * self.num_envs
                infos[key][i] = value

        # Convert info lists to arrays for scalar values (float/int)
        for key, value_list in list(infos.items()):
            if all(
                isinstance(v, (int, float, np.integer, np.floating))
                for v in value_list
                if v is not None
            ):
                array = np.array(value_list, dtype=np.float32)
                infos[key] = self._to_tensor(array)

        # Reset trackers
        self._env_needs_reset.fill(False)
        self._terminations.fill(False)
        self._truncations.fill(False)
        self._rewards.fill(0.0)
        self.elapsed_steps = self._to_tensor(np.zeros((self.num_envs,), dtype=np.int32))

        observations = np.array(self._observations)
        return self._to_tensor(observations), infos

    def step(  # type: ignore[override]  # pylint: disable=arguments-renamed
        self, actions: np.ndarray | torch.Tensor
    ) -> tuple[
        np.ndarray | torch.Tensor,
        np.ndarray | torch.Tensor,
        np.ndarray | torch.Tensor,
        np.ndarray | torch.Tensor,
        dict[str, Any],
    ]:
        """Step all sub-environments with batched actions."""
        actions_np = self._to_numpy(actions)
        assert self.action_space.contains(actions_np), "Actions not in action space"
        self.elapsed_steps += 1

        infos: dict[str, Any] = {}

        for i, env in enumerate(self.envs):
            # Auto-reset paths
            if self._env_needs_reset[i] and self.auto_reset:
                obs, reset_info = env.reset()
                self._observations[i] = obs
                self._rewards[i] = 0.0
                self._terminations[i] = False
                self._truncations[i] = False
                self._env_needs_reset[i] = False
                self.elapsed_steps[i] = 0

                for key, value in reset_info.items():
                    if key not in infos:
                        infos[key] = [None] * self.num_envs
                    infos[key][i] = value
                continue

            # Normal step
            action = actions_np[i]
            obs, reward, terminated, truncated, info = env.step(action)

            self._observations[i] = obs
            self._rewards[i] = np.float32(reward)
            if self._max_episode_steps is None:
                self._terminations[i] = bool(terminated)
            else:
                # NOTE: If max_episode_steps is set, we ignore env-provided
                # termination signal to avoid inconsistency across sub-envs.
                self._terminations[i] = False
            if self._max_episode_steps is not None:
                truncated = self.elapsed_steps[i].item() >= self._max_episode_steps
            self._truncations[i] = bool(truncated)

            # If done, mark for auto-reset next call
            # NOTE: We ignore env-provided termination and truncation
            # if max_episode_steps is set, since it may be inconsistent
            # across sub-envs.
            if (terminated and self._max_episode_steps is None) or truncated:
                self._env_needs_reset[i] = True

            for key, value in info.items():
                if key not in infos:
                    infos[key] = [None] * self.num_envs
                infos[key][i] = value

        # Convert info lists to arrays for scalar values (float/int)
        for key, value_list in list(infos.items()):
            if all(
                isinstance(v, (int, float, np.integer, np.floating))
                for v in value_list
                if v is not None
            ):
                array = np.array(value_list, dtype=np.float32)
                infos[key] = self._to_tensor(array)

        observations = np.array(self._observations)
        rewards = self._rewards.copy()
        terminations = self._terminations.copy()
        truncations = self._truncations.copy()

        return (
            self._to_tensor(observations),
            self._to_tensor(rewards),
            self._to_tensor(terminations),
            self._to_tensor(truncations),
            infos,
        )

    def render(self) -> np.ndarray | None:  # type: ignore
        """Render at most 16 environments and tile them in a 4x4 grid.

        Returns:
            Tiled image as numpy array with shape (height, width, 3) or None
        """
        results: list[np.ndarray] = []
        for env in self.envs:
            rendered_img: np.ndarray | list | None = env.render()
            assert isinstance(
                rendered_img, np.ndarray
            ), "Sub-environment render() must return an image as numpy array"
            results.append(rendered_img)

        if not results:
            return None

        # Tile images in a 4x4 grid (max 16 environments)
        max_envs = min(len(results), 16)
        results = results[:max_envs]

        # Calculate grid dimensions
        grid_cols = min(4, max_envs)
        grid_rows = (max_envs + grid_cols - 1) // grid_cols

        # Get dimensions from first image
        img_height, img_width = results[0].shape[:2]
        channels = results[0].shape[2] if len(results[0].shape) == 3 else 1

        # Create tiled image
        tiled_height = grid_rows * img_height
        tiled_width = grid_cols * img_width

        if channels == 1:
            tiled_image = np.zeros((tiled_height, tiled_width), dtype=results[0].dtype)
        else:
            tiled_image = np.zeros(
                (tiled_height, tiled_width, channels), dtype=results[0].dtype
            )

        # Fill tiled image
        for i, img in enumerate(results):
            row = i // grid_cols
            col = i % grid_cols

            start_row = row * img_height
            end_row = start_row + img_height
            start_col = col * img_width
            end_col = start_col + img_width

            if channels == 1:
                tiled_image[start_row:end_row, start_col:end_col] = img
            else:
                tiled_image[start_row:end_row, start_col:end_col] = img

        return tiled_image

    def close(self, **kwargs):
        """Close all environments."""
        del kwargs  # Unused parameter required by VectorEnv interface
        for env in self.envs:
            if hasattr(env, "close"):
                env.close()

    # -------------------------- Misc -----------------------------

    @property
    def unwrapped(self):
        """Return the underlying sub-environments list."""
        return self.envs


class NormalizeActionMultiEnvWrapper(MultiEnvWrapper):
    """A `MultiEnvWrapper` that normalizes the action space to [-1, 1].

    It assumes the action space of the wrapped environment is a Box space.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert isinstance(
            self.single_action_space, gym.spaces.Box
        ), "Only Box action space is supported"

        # Pre-compute normalized action space parameters
        self.action_low = self.single_action_space.low
        self.action_high = self.single_action_space.high
        self._action_mean = (self.action_high + self.action_low) / 2.0
        self._action_half_range = (self.action_high - self.action_low) / 2.0

        # New normalized action space
        norm_action_space = gym.spaces.Box(
            low=-np.ones_like(self.action_low, dtype=np.float32),
            high=np.ones_like(self.action_high, dtype=np.float32),
            shape=self.single_action_space.shape,
            dtype=np.float32,
        )
        self.action_space = batch_space(norm_action_space, self.num_envs)
        self.single_action_space = norm_action_space

    def step(  # type: ignore[override]  # pylint: disable=arguments-renamed
        self, actions: np.ndarray | torch.Tensor
    ) -> tuple[
        np.ndarray | torch.Tensor,
        np.ndarray | torch.Tensor,
        np.ndarray | torch.Tensor,
        np.ndarray | torch.Tensor,
        dict[str, Any],
    ]:
        """Step all sub-environments with normalized batched actions in [-1, 1]."""
        actions_np = self._to_numpy(actions)
        assert self.action_space.contains(actions_np), "Actions not in action space"
        # Denormalize actions to original space
        denorm_actions = self._action_mean + actions_np * self._action_half_range
        return super().step(denorm_actions)


class MultiEnvRecordVideo(gym.Wrapper):
    """A `RecordVideo` wrapper for `MultiEnvWrapper` that records tiled rgb_array
    renders of all sub-environments.

    We need this because the standard `RecordVideo` expects a
    boolean terminal / truncated signal from the `step()` call, but
    `MultiEnvWrapper` returns a batch of such signals, one per
    sub-environment.

    NOTE: This wrapper currently only supports episode based recording.
    """

    def __init__(
        self,
        env: gym.Env,
        video_folder: str,
        episode_trigger: Optional[Callable[[int], bool]] = None,
        name_prefix: str = "rl-video",
        disable_logger: bool = False,
    ):
        gym.Wrapper.__init__(self, env)
        assert isinstance(
            env, MultiEnvWrapper
        ), "MultiEnvRecordVideo only works with MultiEnvWrapper"

        assert env.render_mode == "rgb_array", (
            "MultiEnvRecordVideo requires the wrapped env to have "
            'render_mode="rgb_array" for image rendering'
        )
        if episode_trigger is None:
            episode_trigger = capped_cubic_video_schedule
        self.episode_trigger = episode_trigger
        self.video_recorder: Optional[video_recorder.VideoRecorder] = None
        self.disable_logger = disable_logger

        self.video_folder = os.path.abspath(video_folder)
        # Create output folder if needed
        if os.path.isdir(self.video_folder):
            logger.warn(
                f"Overwriting existing videos at {self.video_folder} folder "
                f"(try specifying a different `video_folder` for the `RecordVideo` wrapper if this is not desired)"
            )
        os.makedirs(self.video_folder, exist_ok=True)

        self.name_prefix = name_prefix
        self.step_id = 0

        self.recording = False
        self.terminated: bool = False
        self.truncated: bool = False
        self.recorded_frames = 0
        self.episode_id = 0

        self.is_vector_env = True

    def reset(self, **kwargs):
        """Reset the environment using kwargs and then starts recording if video
        enabled."""
        observations = super().reset(**kwargs)
        self.terminated = False
        self.truncated = False
        self.episode_id += 1
        self.step_id = 0
        if self._video_enabled():
            # Force start recording on reset if enabled
            self.start_video_recorder()
        return observations

    def start_video_recorder(self):
        """Starts video recorder using :class:`video_recorder.VideoRecorder`."""
        self.close_video_recorder()

        video_name = f"{self.name_prefix}-episode-{self.episode_id}"
        base_path = os.path.join(self.video_folder, video_name)
        self.video_recorder = video_recorder.VideoRecorder(
            env=self.env,
            base_path=base_path,
            metadata={"step_id": self.step_id, "episode_id": self.episode_id},
            disable_logger=self.disable_logger,
        )

        self.video_recorder.capture_frame()  # type: ignore[no-untyped-call]
        self.recorded_frames = 1
        self.recording = True

    def close_video_recorder(self):
        """Closes the video recorder if currently recording."""
        if self.recording:
            assert self.video_recorder is not None
            self.video_recorder.close()
        self.recording = False
        self.recorded_frames = 1

    def _video_enabled(self) -> bool:  # type: ignore[no-untyped-call]
        return self.episode_trigger(self.episode_id)  # type: ignore[no-untyped-call]

    def step(  # type: ignore[override]  # pylint: disable=arguments-renamed
        self, actions: np.ndarray | torch.Tensor
    ) -> tuple[
        np.ndarray | torch.Tensor,
        np.ndarray | torch.Tensor,
        np.ndarray | torch.Tensor,
        np.ndarray | torch.Tensor,
        dict[str, Any],
    ]:
        """Steps through the environment using actions, recording observations if
        :attr:`self.recording`."""
        assert isinstance(self.env, MultiEnvWrapper)
        (
            observations,
            rewards,
            terminateds,
            truncateds,
            infos,
        ) = self.env.step(actions)
        self.step_id += 1

        if self.recording:
            assert self.video_recorder is not None
            self.video_recorder.capture_frame()  # type: ignore[no-untyped-call]
            self.recorded_frames += 1
        elif self._video_enabled():
            self.start_video_recorder()  # type: ignore[no-untyped-call]

        return observations, rewards, terminateds, truncateds, infos

    def render(self, *args, **kwargs):
        """Compute the render frames as specified by render_mode attribute during
        initialization of the environment or as specified in kwargs."""
        if self.video_recorder is None or not self.video_recorder.enabled:
            return super().render(*args, **kwargs)

        if len(self.video_recorder.render_history) > 0:
            recorded_frames = [
                self.video_recorder.render_history.pop()
                for _ in range(len(self.video_recorder.render_history))
            ]
            if self.recording:
                return recorded_frames
            else:
                return recorded_frames + super().render(*args, **kwargs)
        else:
            return super().render(*args, **kwargs)

    def close(self):
        """Closes the wrapper then the video recorder."""
        super().close()
        self.close_video_recorder()
