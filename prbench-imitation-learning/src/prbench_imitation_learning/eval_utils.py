"""Evaluation utilities for PRBench imitation learning."""

from __future__ import annotations

from typing import Any

import einops
import numpy as np
import torch


def preprocess_observation_generic(observations: Any) -> dict[str, torch.Tensor]:
    """Convert raw env observations to LeRobot-style observation dict.

    Supports:
    - numpy array (state-only)
    - dict with 'pixels' or 'image' (single or multi-cam)
    - dict with 'agent_pos' or 'state'
    """
    OBS_IMAGE = "observation.image"
    OBS_IMAGES = "observation.images"
    OBS_STATE = "observation.state"

    # state-only vector
    if isinstance(observations, np.ndarray):
        state_tensor = torch.from_numpy(observations).float()
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        return {OBS_STATE: state_tensor}

    out: dict[str, torch.Tensor] = {}

    # images
    if isinstance(observations, dict) and (
        "pixels" in observations or "image" in observations
    ):
        source = observations.get("pixels", observations.get("image"))
        if isinstance(source, dict):
            raw_imgs = {f"{OBS_IMAGES}.{k}": v for k, v in source.items()}
        else:
            raw_imgs = {OBS_IMAGE: source, f"{OBS_IMAGES}.cam0": source}

        for key, img in raw_imgs.items():
            img_t = torch.from_numpy(img)
            if img_t.ndim == 3:
                img_t = img_t.unsqueeze(0)
            # (b, h, w, c) -> (b, c, h, w), float32 in [0,1]
            _, h, w, c = img_t.shape
            assert c < h and c < w
            img_t = (
                einops.rearrange(img_t, "b h w c -> b c h w").contiguous().float()
                / 255.0
            )
            out[key] = img_t

    # state
    if isinstance(observations, dict):
        state_arr = None
        if "agent_pos" in observations:
            state_arr = observations["agent_pos"]
        elif "state" in observations:
            state_arr = observations["state"]
        if state_arr is not None:
            st = torch.from_numpy(state_arr).float()
            if st.dim() == 1:
                st = st.unsqueeze(0)
            out[OBS_STATE] = st

    return out


def inject_task_generic(
    observation: dict[str, Any], num_envs: int | None = None
) -> dict[str, Any]:
    """Ensure 'task' key exists in observation for policies that expect it."""
    if "task" in observation:
        return observation
    if num_envs is None:
        if len(observation) == 0:
            num_envs = 1
        else:
            any_val = next(iter(observation.values()))
            num_envs = (
                any_val.shape[0]
                if hasattr(any_val, "shape") and any_val.ndim > 0
                else 1
            )
    observation["task"] = ["" for _ in range(num_envs)]
    return observation
