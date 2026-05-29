"""Common fixtures for the pybullet_helpers tests."""

import pybullet as p
import pytest


def pytest_addoption(parser):
    """Register custom command-line options."""
    parser.addoption(
        "--make-videos",
        action="store_true",
        default=False,
        help="Render videos for tests that support visualization.",
    )


@pytest.fixture(name="make_videos")
def _make_videos(request) -> bool:
    """Whether tests that support visualization should render a video."""
    return bool(request.config.getoption("--make-videos"))


@pytest.fixture(scope="function", name="physics_client_id")
def _connect_to_pybullet():
    """Direct connect to PyBullet physics server, and disconnect when we're done.

    This fixture automatically disconnects the physics server, so we don't forget to do
    it ourselves.
    """
    # Uncomment for debugging.
    # from pybullet_helpers.gui import create_gui_connection
    # physics_client_id = create_gui_connection(camera_yaw=180)
    physics_client_id = p.connect(p.DIRECT)
    yield physics_client_id
    p.disconnect(physics_client_id)
