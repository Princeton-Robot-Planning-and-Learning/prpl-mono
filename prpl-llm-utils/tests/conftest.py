"""Shared configurations for pytest.

See https://docs.pytest.org/en/6.2.x/fixture.html.
"""


def pytest_addoption(parser):
    """Enable a command line flag for running tests decorated with @runllms."""
    parser.addoption(
        "--runllms",
        action="store_true",
        dest="runllms",
        default=False,
        help="Run tests with real LLMs",
    )
