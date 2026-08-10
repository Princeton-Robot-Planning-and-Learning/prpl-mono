"""Motion planning: configuration spaces over a tree and planners over them."""

import importlib
from typing import TYPE_CHECKING, Any

from prpl_kinematics.planning.birrt import BiRRTPlanner
from prpl_kinematics.planning.configuration_space import ConfigurationSpace
from prpl_kinematics.planning.joint_space import JointSpace
from prpl_kinematics.planning.motion_planner import MotionPlanner
from prpl_kinematics.planning.se2_space import SE2Space

if TYPE_CHECKING:
    from prpl_kinematics.planning.ompl_planner import OMPLPlanner, seed_ompl

__all__ = [
    "BiRRTPlanner",
    "ConfigurationSpace",
    "JointSpace",
    "MotionPlanner",
    "OMPLPlanner",
    "SE2Space",
    "seed_ompl",
]

# The OMPL-backed exports resolve on first attribute access so that importing anything
# under this package does not require ompl. ompl publishes wheels for far fewer
# platforms than the rest of the dependencies -- none for Windows, and macOS 15 or
# newer only -- and eagerly importing it here would impose that on the whole package,
# since every submodule import runs this file. BiRRTPlanner requires nothing extra.
_OMPL_EXPORTS = frozenset({"OMPLPlanner", "seed_ompl"})


def __getattr__(name: str) -> Any:
    if name not in _OMPL_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module("prpl_kinematics.planning.ompl_planner")
    except ImportError as exc:
        raise ImportError(
            f"{name} requires ompl, which is an optional dependency. Install it with: "
            'pip install "prpl_kinematics[planning]"'
        ) from exc
    return getattr(module, name)
