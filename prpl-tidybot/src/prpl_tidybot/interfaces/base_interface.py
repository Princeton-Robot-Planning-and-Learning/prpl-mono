"""Base interface."""

import abc
from multiprocessing.connection import Client

import numpy as np
from spatialmath import SE2

from prpl_tidybot.base_server import BaseManager
from prpl_tidybot.constants import (
    BASE_RPC_HOST,
    BASE_RPC_PORT,
    CONN_AUTHKEY,
    RPC_AUTHKEY,
    SERVER_HOSTNAME,
)


class BaseInterface(abc.ABC):
    """Base interface."""

    @abc.abstractmethod
    def get_base_state(self) -> SE2:
        """Get the current base state."""

    def get_map_base_state(self) -> SE2:
        """Get the current base state in the map frame."""

    def execute_action(self, action: SE2) -> None:
        """Execute an action on the base."""


class FakeBaseInterface(BaseInterface):
    """Fake base interface."""

    def __init__(self):
        self.base_state = SE2(x=0, y=0, theta=0)
        self.map_base_state = SE2(x=0, y=0, theta=0)

    def get_base_state(self) -> SE2:
        return self.base_state

    def get_map_base_state(self) -> SE2:
        return self.map_base_state

    def execute_action(self, action: SE2) -> None:
        pass


class RealBaseInterface(BaseInterface):
    """Real base interface."""

    def __init__(self) -> None:
        self.base_manager = BaseManager(
            address=(BASE_RPC_HOST, BASE_RPC_PORT), authkey=RPC_AUTHKEY
        )
        self.base_manager.connect()
        self.base = self.base_manager.Base()  # type: ignore # pylint: disable=no-member
        self.base.reset()

        self.marker_detector_conn = Client(
            (SERVER_HOSTNAME, 6002), authkey=CONN_AUTHKEY
        )
        self.marker_detector_conn.send(None)
        self.last_pose_map = SE2(0, 0, 0)

    def get_base_state(self) -> SE2:
        return SE2(
            self.base.get_state()["base_pose"][0],
            self.base.get_state()["base_pose"][1],
            self.base.get_state()["base_pose"][2],
        )

    def get_map_base_state(
        self,
    ) -> SE2:
        if self.marker_detector_conn.poll():
            detector_data = self.marker_detector_conn.recv()
            self.marker_detector_conn.send(None)
            robot_idx = 0
            pose_map = detector_data["poses"][robot_idx]
            self.last_pose_map = SE2(pose_map[0], pose_map[1], pose_map[2])
            return SE2(pose_map[0], pose_map[1], pose_map[2])
        print("warning: no marker detector data received")
        return self.last_pose_map

    def execute_action(self, action: SE2) -> None:
        """Execute an action on the base."""
        self.base.execute_action(
            {"base_pose": np.array([action.x, action.y, action.theta()])}
        )

    def close(self) -> None:
        """Close the base interface."""
        self.base.close()
