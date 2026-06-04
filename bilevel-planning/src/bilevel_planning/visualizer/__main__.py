"""Entry point for the bilevel planning visualizer.

Usage:

    python -m bilevel_planning.visualizer \
        --bundle path/to/bundle.pkl \
        --renderer path/to/renderer.py [--port N] [--debug] [--no-open]

Loads the bundle (a ``BilevelPlanningGraph.export()`` pickle) and the
renderer file (defining ``render_state(state)``), boots the Flask backend
serving the built React frontend, and opens a browser to a graph that is
immediately clickable. See ``bilevel_planning.visualizer.app`` for the
security model.
"""

import argparse

from bilevel_planning.visualizer.app import run_webapp


def main() -> None:
    """Parse CLI args and launch the visualizer."""
    parser = argparse.ArgumentParser(
        prog="python -m bilevel_planning.visualizer",
        description=(
            "Run the bilevel planning visualizer on a bundle and renderer "
            "chosen at launch."
        ),
    )
    parser.add_argument(
        "--bundle",
        required=True,
        help="Path to a .pkl visualizer bundle from BilevelPlanningGraph.export().",
    )
    parser.add_argument(
        "--renderer",
        required=True,
        help="Path to a Python file defining render_state(state) -> HxWx3 uint8 array.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="Port to bind the Flask server to (default: 5001).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask's debug mode (verbose errors).",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Don't open a browser tab on startup.",
    )
    args = parser.parse_args()

    run_webapp(
        bundle=args.bundle,
        renderer=args.renderer,
        port=args.port,
        debug=args.debug,
        open_browser=not args.no_open,
    )


if __name__ == "__main__":
    main()
