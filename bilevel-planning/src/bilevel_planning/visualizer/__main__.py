"""Universal entry point for the bilevel planning visualizer backend.

Usage:

    python -m bilevel_planning.visualizer [--data-dir PATH] [--port N] [--no-debug]

Boots the Flask backend with no renderer installed. Use the browser UI
(or a direct ``POST /api/set_renderer``) to supply the Python source for
``render_state`` once the server is up. See ``bilevel_planning.visualizer.app``
for details on the security model.
"""

import argparse

from bilevel_planning.visualizer.app import run_webapp


def main() -> None:
    """Parse CLI args and launch the visualizer backend."""
    parser = argparse.ArgumentParser(
        prog="python -m bilevel_planning.visualizer",
        description=(
            "Run the bilevel planning visualizer backend. The renderer is "
            "installed at runtime via POST /api/set_renderer from the browser."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Directory searched when the frontend requests a pickle by bare "
            "filename. Defaults to ./webapp/data under the current working "
            "directory."
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
        data_dir=args.data_dir,
        port=args.port,
        debug=not args.no_debug,
    )


if __name__ == "__main__":
    main()
