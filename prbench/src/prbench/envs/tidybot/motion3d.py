"""Spec for 3D motion planning and manipulation environments (TidyBot3D, etc)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Motion3DEnvSpec:
    """Spec for 3D motion planning/manipulation environments."""

    # Policy server settings
    policy_control_freq: int = 10
    policy_control_period: float = 1.0 / 10
