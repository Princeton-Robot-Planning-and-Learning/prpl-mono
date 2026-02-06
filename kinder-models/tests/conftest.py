"""Configure pytest."""

from _pytest.config import Config
from _pytest.config.argparsing import Parser

# Global variable that gets set with the command line --make-videos flag.
MAKE_VIDEOS = False


def pytest_addoption(parser: Parser) -> None:
    """Register custom command-line options for pytest."""
    parser.addoption(
        "--make-videos",
        action="store_true",
        default=False,
        help="Enable video generation during tests",
    )


def pytest_configure(config: Config) -> None:
    """Set global configuration values after command-line options are parsed."""
    global MAKE_VIDEOS  # pylint:disable=global-statement
    MAKE_VIDEOS = config.getoption("--make-videos")
