"""Shared configurations for pytest.

See https://docs.pytest.org/en/6.2.x/fixture.html.
"""


def pytest_addoption(parser):
    """Enable a command line flag for running tests that query real agents."""
    parser.addoption(
        "--runagents",
        action="store_true",
        dest="runagents",
        default=False,
        help="Run tests with real coding agents",
    )
