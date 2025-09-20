"""Common fixtures for the pybullet_helpers tests."""

import pybullet as p
import pytest


@pytest.fixture(scope="function", name="physics_client_id")
def _connect_to_pybullet():
    """Direct connect to PyBullet physics server, and disconnect when we're
    done.

    This fixture automatically disconnects the physics server, so we
    don't forget to do it ourselves.
    """
    # Uncomment for debugging.
    # from pybullet_helpers.gui import create_gui_connection
    # physics_client_id = create_gui_connection(camera_yaw=180)
    physics_client_id = p.connect(p.DIRECT)
    yield physics_client_id
    p.disconnect(physics_client_id)
