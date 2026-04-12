"""Universal entry point for the bilevel planning visualizer.

Usage:

    python -m bilevel_planning.visualizer [--port N] [--no-debug]

Boots the Flask backend with no renderer installed and serves the built
React frontend at the same port. Open the printed URL in a browser, click
**Load Pickle**, then expand the **Python renderer** pane and apply your
``render_state`` source. See ``bilevel_planning.visualizer.app`` for the
security model.
"""

import argparse

from bilevel_planning.visualizer.app import run_webapp


def main() -> None:
    """Parse CLI args and launch the visualizer."""
    parser = argparse.ArgumentParser(
        prog="python -m bilevel_planning.visualizer",
        description=(
            "Run the bilevel planning visualizer. The renderer is installed "
            "at runtime via the browser's 'Python renderer' pane (or via a "
            "direct POST to /api/set_renderer)."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="Port to bind the Flask server to (default: 5001).",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable Flask's debug mode (auto-reload + verbose errors).",
    )
    args = parser.parse_args()

    run_webapp(
        render_state_fn=None,
        port=args.port,
        debug=not args.no_debug,
    )


if __name__ == "__main__":
    main()
