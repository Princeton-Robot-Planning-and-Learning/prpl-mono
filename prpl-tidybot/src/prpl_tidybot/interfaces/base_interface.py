"""Base interface."""

import abc

import spatialmath

from prpl_tidybot.base_server import BaseManager
from prpl_tidybot.constants import (
    BASE_RPC_HOST,
    BASE_RPC_PORT,
    RPC_AUTHKEY,
)


class BaseInterface(abc.ABC):
    """Base interface."""

    @abc.abstractmethod
    def get_base_state(self) -> spatialmath.SE2:
        """Get the current base state."""


class FakeBaseInterface(BaseInterface):
    """Fake base interface."""

    def __init__(self):
        self.base_state = spatialmath.SE2(x=0, y=0, theta=0)

    def get_base_state(self) -> spatialmath.SE2:
        return self.base_state


class RealBaseInterface(BaseInterface):
    """Real base interface."""

    def __init__(self) -> None:
        self.base_manager = BaseManager(
            address=(BASE_RPC_HOST, BASE_RPC_PORT), authkey=RPC_AUTHKEY
        )
        self.base_manager.connect()
        self.base = self.base_manager.Base()  # type: ignore # pylint: disable=no-member
        self.base.reset()

    def get_base_state(self) -> spatialmath.SE2:
        return spatialmath.SE2(
            self.base.get_state()["base_pose"][0],
            self.base.get_state()["base_pose"][1],
            self.base.get_state()["base_pose"][2],
        )

    def execute_action(self, action) -> None:
        """Execute an action on the base."""
        raise NotImplementedError("Real base execute_action not implemented yet.")
        # self.base.execute_action(action)

    def close(self) -> None:
        """Close the base interface."""
        self.base.close()
