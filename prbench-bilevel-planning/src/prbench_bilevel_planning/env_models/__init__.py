"""Dynamically load bilevel planning env models."""

import importlib.util
import sys
from pathlib import Path

from bilevel_planning.structs import SesameModels
from gymnasium.spaces import Space

__all__ = ["create_bilevel_planning_models"]


def create_bilevel_planning_models(
    env_name: str, observation_space: Space, executable_space: Space, **kwargs
) -> SesameModels:
    """Load bilevel planning models for the given environment."""
    current_file = Path(__file__).resolve()

    # Try different directories based on environment type
    possible_paths = [
        current_file.parent / "geom2d" / f"{env_name}.py",
        current_file.parent / "tidybot3d" / f"{env_name}.py",
    ]

    env_path = None
    for path in possible_paths:
        if path.exists():
            env_path = path
            break

    if env_path is None:
        raise FileNotFoundError(
            f"No model file found for environment '{env_name}' in any of: "
            f"{possible_paths}"
        )

    module_name = f"{env_name}_module"
    spec = importlib.util.spec_from_file_location(module_name, env_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {env_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "create_bilevel_planning_models"):
        raise AttributeError(
            f"{env_path} does not define `create_bilevel_planning_models`"
        )

    return module.create_bilevel_planning_models(
        observation_space, executable_space, **kwargs
    )
