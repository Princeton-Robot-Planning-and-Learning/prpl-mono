"""Evaluation functions for PRBench imitation learning."""

import concurrent.futures as cf
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, TypedDict

import einops
import gymnasium as gym
import numpy as np
import torch
from lerobot.envs.utils import check_env_attributes_and_types
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, DONE, OBS_STR, REWARD
from lerobot.utils.io_utils import write_video
from lerobot.utils.utils import (
    inside_slurm,
)
from torch import Tensor, nn
from tqdm import trange

from prbench_imitation_learning.eval_utils import (
    inject_task_generic,
    preprocess_observation_generic,
)


def rollout(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    seeds: list[int] | None = None,
    return_observations: bool = False,
    render_callback: Callable[[gym.vector.VectorEnv], None] | None = None,
) -> dict:
    """Run a batched policy rollout once through a batch of environments.

    Note that all environments in the batch are run until the last environment is done.
    This means some data will probably need to be discarded
    (for environments that aren't the first one to be done).

    The return dictionary contains:
        (optional) "observation": A dictionary of (batch, sequence + 1, *)
            tensors mapped to observation keys.
            NOTE that this has an extra sequence element relative to
            the other keys in the dictionary.
            This is because an extra observation is included for after the environment is
            terminated or truncated.
        "action": A (batch, sequence, action_dim) tensor of actions applied
            based on the observations (not including the last observations).
        "reward": A (batch, sequence) tensor of rewards received
            for applying the actions.
        "success": A (batch, sequence) tensor of success conditions
            (the only time this can be True is upon environment termination/truncation).
        "done": A (batch, sequence) tensor of **cumulative** done conditions.
            For any given batch element,
            the first True is followed by True's all the way till the end.
            This can be used for masking extraneous elements from the sequences above.

    Args:
        env: The batch of environments.
        policy: The policy. Must be a PyTorch nn module.
        seeds: The environments are seeded once at the start of the rollout.
        If provided, this argument specifies the seeds for each of the environments.
        return_observations: Whether to include all observations in the
        returned rollout data. Observations are returned optionally
        because they typically take more memory to cache.
        Defaults to False.
        render_callback: Optional rendering callback to be used
        after the environments are reset, and after every step.
    Returns:
        The dictionary described above.
    """
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    # Reset the policy and environments.
    policy.reset()  # type: ignore
    observation, _ = env.reset(seed=seeds)
    # Ensure observation provides both state and image for policies expecting images
    if not isinstance(observation, dict):
        frames = None
        try:
            frames_list = env.call("render")
            if isinstance(frames_list, list) and len(frames_list) > 0:
                frames = np.stack(frames_list, axis=0)
        except Exception:
            frames = None
        if frames is not None:
            observation = {"state": observation, "pixels": frames}
        else:
            observation = {"state": observation}
    if render_callback is not None:
        render_callback(env)

    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []

    step = 0
    # Keep track of which environments are done.
    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    progbar = trange(
        max_steps,
        desc=f"Running rollout with at most {max_steps} steps",
        disable=inside_slurm(),
        leave=False,
    )
    check_env_attributes_and_types(env)
    while not np.all(done) and step < max_steps:
        # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
        observation = preprocess_observation_generic(observation)
        # Harmonize image keys before preprocessing pipeline
        if (
            "observation.image" not in observation
            and "observation.images.cam0" in observation
        ):
            observation["observation.image"] = observation["observation.images.cam0"]
        if (
            "observation.images.cam0" not in observation
            and "observation.image" in observation
        ):
            observation["observation.images.cam0"] = observation["observation.image"]
        if return_observations:
            all_observations.append(deepcopy(observation))

        # Infer "task" from attributes of environments.
        observation = inject_task_generic(observation)
        observation = preprocessor(observation)
        # Ensure both single-image and multi-cam keys are present for diffusion stacking
        if (
            "observation.image" not in observation
            and "observation.images.cam0" in observation
        ):
            observation["observation.image"] = observation["observation.images.cam0"]
        if (
            "observation.images.cam0" not in observation
            and "observation.image" in observation
        ):
            observation["observation.images.cam0"] = observation["observation.image"]
        # Fabricate missing image keys if none are present but policy expects images
        try:
            needed = (
                list(policy.config.image_features.keys())  # type: ignore
                if hasattr(policy.config, "image_features")
                else []
            )
        except Exception:
            needed = []
        if needed:
            # Determine batch size from any tensor in observation
            batch_size = None
            for v in observation.values():
                if isinstance(v, torch.Tensor) and v.dim() > 0:
                    batch_size = v.shape[0]
                    break
            if batch_size is None:
                batch_size = 1
            for img_key in needed:
                if img_key not in observation:
                    # Try to copy from any existing image
                    src = observation.get("observation.images.cam0") or observation.get(
                        "observation.image"
                    )
                    if isinstance(src, torch.Tensor):
                        observation[img_key] = src
                    else:
                        # Create black image of expected shape (C,H,W)
                        shape = policy.config.image_features[img_key].shape  # type: ignore # pylint: disable=line-too-long
                        _dev = next(policy.parameters()).device
                        observation[img_key] = torch.zeros(
                            (batch_size, *shape), dtype=torch.float32, device=_dev
                        )
        with torch.inference_mode():
            action = policy.select_action(observation)  # type: ignore
        action = postprocessor(action)

        # Convert to CPU / numpy.
        action_numpy: np.ndarray = action.to("cpu").numpy()
        assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"

        # Apply the next action.
        observation, reward, terminated, truncated, _ = env.step(action_numpy)
        # Normalize observation format after step as well
        if not isinstance(observation, dict):
            frames = None
            try:
                frames_list = env.call("render")
                if isinstance(frames_list, list) and len(frames_list) > 0:
                    frames = np.stack(frames_list, axis=0)
            except Exception:
                frames = None
            if frames is not None:
                observation = {"state": observation, "pixels": frames}
            else:
                observation = {"state": observation}
        if render_callback is not None:
            render_callback(env)

        successes = [
            bool(terminated[idx] and not truncated[idx] and (step + 1) < max_steps)
            for idx in range(env.num_envs)
        ]

        # Keep track of which environments are done so far.
        # Mark the episode as done if we reach the maximum step limit.
        # This ensures that the rollout always terminates cleanly at `max_steps`,
        # and allows logging/saving (e.g., videos) to be triggered consistently.
        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=bool)

        all_actions.append(torch.from_numpy(action_numpy))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))

        step += 1
        running_success_rate = (
            einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any")
            .numpy()
            .mean()
        )
        progbar.set_postfix(
            {"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"}
        )
        progbar.update()

    # Track the final observation.
    if return_observations:
        observation = preprocess_observation_generic(observation)
        all_observations.append(deepcopy(observation))

    # Stack the sequence along the first dimension
    # so that we have (batch, sequence, *) tensors.
    ret = {
        ACTION: torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
    }
    if return_observations:
        stacked_observations = {}
        for key in all_observations[0]:
            stacked_observations[key] = torch.stack(
                [obs[key] for obs in all_observations], dim=1
            )
        ret[OBS_STR] = stacked_observations  # type: ignore

    if hasattr(policy, "use_original_modules"):
        policy.use_original_modules()  # type: ignore

    return ret


def eval_policy(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
) -> dict:
    """
    Args:
        env: The batch of environments.
        policy: The policy.
        n_episodes: The number of episodes to evaluate.
        max_episodes_rendered: Maximum number of episodes to render into videos.
        videos_dir: Where to save rendered videos.
        return_episode_data: Whether to return episode data for online training.
        Incorporates the data into the "episodes" key of the returned dictionary.
        start_seed: The first seed to use for the first individual rollout.
        For all subsequent rollouts the seed is incremented by 1.
        If not provided, the environments are not manually seeded.
    Returns:
        Dictionary with metrics and data regarding the rollouts.
    """
    if max_episodes_rendered > 0 and not videos_dir:
        raise ValueError("If max_episodes_rendered > 0, videos_dir must be provided.")

    if not isinstance(policy, PreTrainedPolicy):
        raise ValueError(
            f"Policy of type 'PreTrainedPolicy' is expected, "
            f"but type '{type(policy)}' was provided."
        )

    start = time.time()
    policy.eval()

    # Determine how many batched rollouts we need to get n_episodes.
    # Note that if n_episodes is not evenly
    # divisible by env.num_envs we end up discarding some data in the last batch.
    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)

    # Keep track of some metrics.
    sum_rewards = []
    max_rewards = []
    all_successes = []
    all_seeds: list[int | None] = []
    threads = []  # for video saving threads
    n_episodes_rendered = 0  # for saving the correct number of videos

    # Callback for visualization.
    def render_frame(env: gym.vector.VectorEnv):
        # noqa: B023
        if n_episodes_rendered >= max_episodes_rendered:
            return
        n_to_render_now = min(max_episodes_rendered - n_episodes_rendered, env.num_envs)
        if isinstance(env, gym.vector.SyncVectorEnv):
            ep_frames.append(  # pylint: disable = possibly-used-before-assignment
                np.stack([env.envs[i].render() for i in range(n_to_render_now)])  # type: ignore # pylint: disable=line-too-long
            )  # noqa: B023
        elif isinstance(env, gym.vector.AsyncVectorEnv):
            # Here we must render all frames and discard any we don't need.
            ep_frames.append(
                np.stack(env.call("render")[:n_to_render_now])
            )  # pylint: disable = possibly-used-before-assignment

    if max_episodes_rendered > 0:
        video_paths: list[str] = []

    if return_episode_data:
        episode_data: dict | None = None

    # we dont want progress bar when we use slurm, since it clutters the logs
    progbar = trange(
        n_batches, desc="Stepping through eval batches", disable=inside_slurm()
    )
    for batch_ix in progbar:
        if max_episodes_rendered > 0:
            ep_frames: list[np.ndarray] = []

        if start_seed is None:
            seeds = None
        else:
            seeds = range(
                start_seed + (batch_ix * env.num_envs),
                start_seed + ((batch_ix + 1) * env.num_envs),
            )
        rollout_data = rollout(
            env=env,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            seeds=list(seeds) if seeds else None,
            return_observations=return_episode_data,
            render_callback=render_frame if max_episodes_rendered > 0 else None,
        )

        n_steps = rollout_data["done"].shape[1]
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)

        mask = (
            torch.arange(n_steps)
            <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)
        ).int()
        # Extend metrics.
        batch_sum_rewards = einops.reduce(
            (rollout_data["reward"] * mask), "b n -> b", "sum"
        )
        sum_rewards.extend(batch_sum_rewards.tolist())
        batch_max_rewards = einops.reduce(
            (rollout_data["reward"] * mask), "b n -> b", "max"
        )
        max_rewards.extend(batch_max_rewards.tolist())
        batch_successes = einops.reduce(
            (rollout_data["success"] * mask), "b n -> b", "any"
        )
        all_successes.extend(batch_successes.tolist())
        if seeds:
            all_seeds.extend(seeds)
        else:
            all_seeds.append(None)

        if return_episode_data:
            this_episode_data = _compile_episode_data(
                rollout_data,
                done_indices,
                start_episode_index=batch_ix * env.num_envs,
                start_data_index=(
                    0
                    if episode_data is None
                    else (episode_data["index"][-1].item() + 1)
                ),
                fps=env.unwrapped.metadata["render_fps"],
            )
            if episode_data is None:
                episode_data = this_episode_data
            else:
                # Some sanity checks to make sure we are correctly compiling the data.
                assert (
                    episode_data["episode_index"][-1] + 1
                    == this_episode_data["episode_index"][0]
                )
                assert episode_data["index"][-1] + 1 == this_episode_data["index"][0]
                # Concatenate the episode data.
                episode_data = {
                    k: torch.cat([episode_data[k], this_episode_data[k]])
                    for k in episode_data
                }

        # Maybe render video for visualization.
        if max_episodes_rendered > 0 and len(ep_frames) > 0:
            batch_stacked_frames = np.stack(ep_frames, axis=1)  # (b, t, *)
            for stacked_frames, done_index in zip(
                batch_stacked_frames, done_indices.flatten().tolist(), strict=False
            ):
                if n_episodes_rendered >= max_episodes_rendered:
                    break

                videos_dir.mkdir(parents=True, exist_ok=True)  # type: ignore
                video_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"  # type: ignore # pylint: disable=line-too-long
                video_paths.append(  # pylint: disable = possibly-used-before-assignment
                    str(video_path)
                )
                thread = threading.Thread(
                    target=write_video,
                    args=(
                        str(video_path),
                        stacked_frames[
                            : done_index + 1
                        ],  # + 1 to capture the last observation
                        env.unwrapped.metadata["render_fps"],
                    ),
                )
                thread.start()
                threads.append(thread)
                n_episodes_rendered += 1

        progbar.set_postfix(
            {
                "running_success_rate": f"{np.mean(all_successes[:n_episodes]).item() * 100:.1f}%"  # pylint: disable=line-too-long
            }
        )

    # Wait till all video rendering threads are done.
    for thread in threads:
        thread.join()

    # Compile eval info.
    info = {
        "per_episode": [
            {
                "episode_ix": i,
                "sum_reward": sum_reward,
                "max_reward": max_reward,
                "success": success,
                "seed": seed,
            }
            for i, (sum_reward, max_reward, success, seed) in enumerate(
                zip(
                    sum_rewards[:n_episodes],
                    max_rewards[:n_episodes],
                    all_successes[:n_episodes],
                    all_seeds[:n_episodes],
                    strict=True,
                )
            )
        ],
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards[:n_episodes])),
            "avg_max_reward": float(np.nanmean(max_rewards[:n_episodes])),
            "pc_success": float(np.nanmean(all_successes[:n_episodes]) * 100),
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / n_episodes,
        },
    }

    if return_episode_data:
        info["episodes"] = episode_data  # type: ignore

    if max_episodes_rendered > 0:
        info["video_paths"] = video_paths

    return info


def _compile_episode_data(
    rollout_data: dict,
    done_indices: Tensor,
    start_episode_index: int,
    start_data_index: int,
    fps: float,
) -> dict:
    """Convenience function for `eval_policy(return_episode_data=True)`

    Compiles all the rollout data into a Hugging Face dataset.

    Similar logic is implemented when datasets are pushed to hub (see: `push_to_hub`).
    """
    ep_dicts = []
    total_frames = 0
    for ep_ix in range(rollout_data[ACTION].shape[0]):
        # + 2 to include the first done frame and the last observation frame.
        num_frames = done_indices[ep_ix].item() + 2
        total_frames += num_frames  # type: ignore

        ep_dict = {
            ACTION: rollout_data[ACTION][ep_ix, : num_frames - 1],  # type: ignore
            "episode_index": torch.tensor(
                [start_episode_index + ep_ix] * (num_frames - 1)  # type: ignore
            ),
            "frame_index": torch.arange(0, num_frames - 1, 1),
            "timestamp": torch.arange(0, num_frames - 1, 1) / fps,
            DONE: rollout_data["done"][ep_ix, : num_frames - 1],  # type: ignore
            "next.success": rollout_data["success"][ep_ix, : num_frames - 1],  # type: ignore # pylint: disable=line-too-long
            REWARD: rollout_data["reward"][ep_ix, : num_frames - 1].type(torch.float32),  # type: ignore # pylint: disable=line-too-long
        }

        # For the last observation frame, all other keys will just be copy padded.
        for k in ep_dict:
            ep_dict[k] = torch.cat([ep_dict[k], ep_dict[k][-1:]])

        for key in rollout_data[OBS_STR]:
            ep_dict[key] = rollout_data[OBS_STR][key][ep_ix, :num_frames]  # type: ignore

        ep_dicts.append(ep_dict)

    data_dict = {}
    for key in ep_dicts[0]:
        data_dict[key] = torch.cat([x[key] for x in ep_dicts])

    data_dict["index"] = torch.arange(
        start_data_index, start_data_index + total_frames, 1
    )

    return data_dict


# ---- typed payload returned by one task eval ----
class TaskMetrics(TypedDict):
    """Typed dictionary returned by eval_one."""

    sum_rewards: list[float]
    max_rewards: list[float]
    successes: list[bool]
    video_paths: list[str]


ACC_KEYS = ("sum_rewards", "max_rewards", "successes", "video_paths")


def eval_one(
    env: gym.vector.VectorEnv,
    *,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
) -> TaskMetrics:
    """Evaluates one task_id of one suite using the provided vec env."""

    task_videos_dir = videos_dir

    task_result = eval_policy(
        env=env,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
    )

    per_episode = task_result["per_episode"]
    return TaskMetrics(
        sum_rewards=[ep["sum_reward"] for ep in per_episode],
        max_rewards=[ep["max_reward"] for ep in per_episode],
        successes=[ep["success"] for ep in per_episode],
        video_paths=task_result.get("video_paths", []),
    )


def run_one(
    task_group: str,
    task_id: int,
    env,
    *,
    policy,
    preprocessor,
    postprocessor,
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
):
    """Run eval_one for a single (task_group, task_id, env).

    Returns (task_group, task_id, task_metrics_dict). This function is intentionally
    module-level to make it easy to test.
    """
    task_videos_dir = None
    if videos_dir is not None:
        task_videos_dir = videos_dir / f"{task_group}_{task_id}"
        task_videos_dir.mkdir(parents=True, exist_ok=True)

    # Call the existing eval_one (assumed to return TaskMetrics-like dict)
    metrics = eval_one(
        env,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
    )
    # ensure we always provide video_paths key to simplify accumulation
    if max_episodes_rendered > 0:
        metrics.setdefault("video_paths", [])
    return task_group, task_id, metrics


def eval_policy_all(
    envs: dict[str, dict[int, gym.vector.VectorEnv]],
    policy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    *,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    max_parallel_tasks: int = 1,
) -> dict:
    """Evaluate a nested `envs` dict: {task_group: {task_id: vec_env}}.

    This implementation flattens tasks, runs them sequentially or via
    ThreadPoolExecutor, accumulates per-group and overall statistics, and returns the
    same aggregate metrics schema as the single-env evaluator (avg_sum_reward /
    avg_max_reward / pc_success / timings) plus per-task infos.
    """
    start_t = time.time()

    # Flatten envs into list of (task_group, task_id, env)
    tasks = [(tg, tid, vec) for tg, group in envs.items() for tid, vec in group.items()]

    # accumulators: track metrics at both per-group level and across all groups
    group_acc: dict[str, dict[str, list]] = defaultdict(
        lambda: {k: [] for k in ACC_KEYS}
    )
    overall: dict[str, list] = {k: [] for k in ACC_KEYS}
    per_task_infos: list[dict] = []

    # small inline helper to accumulate one task's metrics into accumulators
    def _accumulate_to(group: str, metrics: dict):
        def _append(key, value):
            if value is None:
                return
            if isinstance(value, list):
                group_acc[group][key].extend(value)
                overall[key].extend(value)
            else:
                group_acc[group][key].append(value)
                overall[key].append(value)

        _append("sum_rewards", metrics.get("sum_rewards"))  # type: ignore
        _append("max_rewards", metrics.get("max_rewards"))  # type: ignore
        _append("successes", metrics.get("successes"))  # type: ignore
        # video_paths is list-like
        paths = metrics.get("video_paths", [])
        if paths:
            group_acc[group]["video_paths"].extend(paths)
            overall["video_paths"].extend(paths)

    # Choose runner (sequential vs threaded)
    task_runner = partial(
        run_one,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
    )

    if max_parallel_tasks <= 1:
        # sequential path (single accumulator path on the main thread)
        for task_group, task_id, env in tasks:
            tg, tid, metrics = task_runner(task_group, task_id, env)
            _accumulate_to(tg, metrics)
            per_task_infos.append(
                {"task_group": tg, "task_id": tid, "metrics": metrics}
            )
    else:
        with cf.ThreadPoolExecutor(max_workers=max_parallel_tasks) as executor:
            fut2meta = {}
            for task_group, task_id, env in tasks:
                fut = executor.submit(task_runner, task_group, task_id, env)
                fut2meta[fut] = (task_group, task_id)
            for fut in cf.as_completed(fut2meta):
                tg, tid, metrics = fut.result()
                _accumulate_to(tg, metrics)
                per_task_infos.append(
                    {"task_group": tg, "task_id": tid, "metrics": metrics}
                )

    # compute aggregated metrics helper (robust to lists/scalars)
    def _agg_from_list(xs):
        if not xs:
            return float("nan")
        arr = np.array(xs, dtype=float)
        return float(np.nanmean(arr))

    # compute per-group aggregates
    groups_aggregated = {}
    for group, acc in group_acc.items():
        groups_aggregated[group] = {
            "avg_sum_reward": _agg_from_list(acc["sum_rewards"]),  # type: ignore
            "avg_max_reward": _agg_from_list(acc["max_rewards"]),  # type: ignore
            "pc_success": (
                _agg_from_list(acc["successes"]) * 100  # type: ignore
                if acc["successes"]
                else float("nan")
            ),
            "n_episodes": len(acc["sum_rewards"]),
            "video_paths": list(acc["video_paths"]),
        }

    # overall aggregates
    overall_agg = {
        "avg_sum_reward": _agg_from_list(overall["sum_rewards"]),  # type: ignore
        "avg_max_reward": _agg_from_list(overall["max_rewards"]),  # type: ignore
        "pc_success": (
            _agg_from_list(overall["successes"]) * 100  # type: ignore
            if overall["successes"]
            else float("nan")
        ),
        "n_episodes": len(overall["sum_rewards"]),
        "eval_s": time.time() - start_t,
        "eval_ep_s": (time.time() - start_t) / max(1, len(overall["sum_rewards"])),
        "video_paths": list(overall["video_paths"]),
    }

    return {
        "per_task": per_task_infos,
        "per_group": groups_aggregated,
        "overall": overall_agg,
    }
