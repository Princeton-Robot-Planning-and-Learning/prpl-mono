"""Base class for Dynamic3D robot environments."""

from dataclasses import dataclass

# from pyparsing import TypeVar
from gymnasium.spaces import Space

# from relational_structs.common import Array
from relational_structs.object_centric_state import ObjectCentricState

from prbench.core import ObjectCentricPRBenchEnv, PRBenchEnvConfig
from prbench.envs.tidybot.mujoco_utils import MjAct


@dataclass(frozen=True)
class TidyBot3DConfig(PRBenchEnvConfig):
    """Configuration for TidyBot3D environment."""

    control_frequency: int = 20
    horizon: int = 1000
    camera_width: int = 640
    camera_height: int = 480
    show_viewer: bool = False


class ObjectCentricDynamic3DRobotEnv(
    ObjectCentricPRBenchEnv[ObjectCentricState, MjAct, TidyBot3DConfig]
):
    """Base class for Dynamic3D robot environments."""

    def _create_action_space(self, config: TidyBot3DConfig) -> Space[MjAct]:
        """Create action space for TidyBot's control interface."""
