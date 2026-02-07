"""Dynamically load domain-specific policies for KinDER environments."""

import importlib.util
import sys
from pathlib import Path

from kinder_ds_policies.policies.base import StatefulPolicy

__all__ = ["create_domain_specific_policy", "StatefulPolicy"]


def create_domain_specific_policy(env_name: str, **kwargs) -> StatefulPolicy:
    """Load domain-specific policy for the given environment.

    Args:
        env_name: The name of the environment (e.g., "base_motion3d").
        **kwargs: Additional keyword arguments passed to the policy factory.

    Returns:
        A callable policy that maps observations to actions.
    """
    current_file = Path(__file__).resolve()

    # Try different directories based on environment type
    possible_paths = [
        current_file.parent / "kinematic2d" / f"{env_name}.py",
        current_file.parent / "dynamic2d" / f"{env_name}.py",
        current_file.parent / "kinematic3d" / f"{env_name}.py",
        current_file.parent / "dynamic3d" / f"{env_name}.py",
    ]

    env_path = None
    for path in possible_paths:
        if path.exists():
            env_path = path
            break

    if env_path is None:
        raise FileNotFoundError(
            f"No policy file found for environment '{env_name}' in any of: "
            f"{possible_paths}"
        )

    module_name = f"{env_name}_policy_module"
    spec = importlib.util.spec_from_file_location(module_name, env_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {env_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "create_domain_specific_policy"):
        raise AttributeError(
            f"{env_path} does not define `create_domain_specific_policy`"
        )

    return module.create_domain_specific_policy(**kwargs)
