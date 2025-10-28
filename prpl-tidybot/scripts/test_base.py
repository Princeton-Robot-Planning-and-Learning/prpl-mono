"""Test for mobile base by sending actions."""

import time

import numpy as np

from prpl_tidybot.base_server import BaseManager
from prpl_tidybot.constants import (
    BASE_RPC_HOST,
    BASE_RPC_PORT,
    POLICY_CONTROL_PERIOD,
    RPC_AUTHKEY,
)

if __name__ == "__main__":
    manager = BaseManager(address=(BASE_RPC_HOST, BASE_RPC_PORT), authkey=RPC_AUTHKEY)
    manager.connect()
    base = manager.Base()  # type: ignore # pylint: disable=no-member
    try:
        base.reset()
        for i in range(50):
            base.execute_action({"base_pose": np.array([(i / 50) * 0.5, 0.0, 0.0])})
            print("target pose:", np.array([(i / 50) * 0.5, 0.0, 0.0]))
            print("current pose:", base.get_state()["base_pose"])
            time.sleep(POLICY_CONTROL_PERIOD)  # Note: Not precise
    finally:
        base.close()
