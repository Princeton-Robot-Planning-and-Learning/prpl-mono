"""Common fixtures for the prpl_kinematics tests."""

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


@pytest.fixture(name="physics_client_id")
def _physics_client_id():
    """A headless PyBullet client, disconnected on teardown."""
    physics_client_id = p.connect(p.DIRECT)
    yield physics_client_id
    p.disconnect(physics_client_id)


@pytest.fixture(name="render_client_id")
def _render_client_id():
    """A second headless client for rendering.

    A renderer must not share a client with a collision checker: the checker's
    collision bodies would also be drawn by ``getCameraImage`` (frozen at the
    last pose the planner tested), ghosting a static robot over the animated one.
    """
    render_client_id = p.connect(p.DIRECT)
    yield render_client_id
    p.disconnect(render_client_id)
