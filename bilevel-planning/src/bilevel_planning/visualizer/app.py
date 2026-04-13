"""Flask backend for visualizing concrete states from a bilevel planning graph.

Environment-agnostic: the backend never imports a simulation package. The
user POSTs Python source to ``/api/set_renderer`` from the browser; the
backend ``exec``s the source, expects a callable named ``render_state`` to
land in the namespace, and caches it for subsequent ``/api/visualize_state``
requests.

The same Flask process also serves the built React frontend at ``/``, so
the visualizer runs as a single ``python -m bilevel_planning.visualizer``
invocation. The frontend bundle lives at ``visualizer/frontend/dist/`` and
must be built once with ``npm ci && npm run build`` from the
``frontend/`` directory.

Security: ``/api/set_renderer`` runs arbitrary Python in the backend
process. The server binds to ``127.0.0.1`` so only local clients can reach
it. Do not put this behind a reverse proxy exposing it to untrusted
networks.
"""

# pylint: disable=global-statement,wrong-import-position,wrong-import-order

# Force matplotlib onto a non-interactive backend before any code path
# (including the user's exec'd renderer source) can import pyplot. Flask
# handles requests on worker threads, and on macOS the default 'MacOSX'
# backend refuses to create figures off the main thread. Has to happen
# before the rest of the imports in case any of them transitively import
# matplotlib.
import matplotlib

matplotlib.use("Agg")

import base64
import io
import pickle
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS  # type: ignore[import-untyped]
from PIL import Image  # type: ignore[import-untyped]

RenderStateFn = Callable[[Any], np.ndarray]

# Backend state. ``GRAPH_DATA`` is the frontend-facing topology dict served
# by ``/api/graph``; ``STATE_DATA`` maps node ids to the original state
# objects, indexed into by ``/api/visualize_state``. Both come from the
# same pickle uploaded via ``/api/load_pickle``. ``RENDER_FN`` is the
# render callable supplied at runtime via ``/api/set_renderer``.
GRAPH_DATA: dict = {}
STATE_DATA: dict = {}
RENDER_FN: RenderStateFn | None = None

# Name the exec'd source must bind to for ``/api/set_renderer`` to pick it
# up. Keeping this a constant makes the browser template and the backend
# agree without hardcoding duplicate strings.
RENDERER_ENTRYPOINT = "render_state"

# Location of the built React frontend bundle, served as static files at
# ``/`` so the visualizer is one process instead of two.
FRONTEND_DIST_DIR = Path(__file__).parent / "frontend" / "dist"


def create_app() -> Flask:
    """Build the Flask app.

    The app boots with no renderer; ``/api/visualize_state`` returns 400
    until the user supplies one via ``/api/set_renderer``.
    """
    global RENDER_FN
    RENDER_FN = None

    app = Flask(__name__)
    CORS(app)

    @app.route("/api/load_pickle", methods=["POST"])
    def load_pickle():
        global GRAPH_DATA, STATE_DATA
        try:
            upload = request.files.get("file")
            if upload is None:
                return (
                    jsonify(
                        {
                            "error": (
                                "Request must include a 'file' multipart "
                                "field with the visualizer pickle bundle."
                            ),
                        }
                    ),
                    400,
                )

            try:
                bundle = pickle.load(upload.stream)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                return (
                    jsonify(
                        {
                            "error": f"Could not unpickle uploaded file: {exc}",
                            "traceback": traceback.format_exc(),
                        }
                    ),
                    400,
                )

            if not (
                isinstance(bundle, dict) and "graph" in bundle and "states" in bundle
            ):
                return (
                    jsonify(
                        {
                            "error": (
                                "Pickle is not a bilevel-planning visualizer "
                                "bundle; expected a dict with 'graph' and "
                                "'states' keys. Regenerate with "
                                "BilevelPlanningGraph.export()."
                            ),
                        }
                    ),
                    400,
                )

            GRAPH_DATA = bundle["graph"]
            STATE_DATA = bundle["states"]

            num_states = len(STATE_DATA)
            return jsonify(
                {
                    "success": True,
                    "num_states": num_states,
                    "filename": upload.filename,
                    "message": f"Loaded {num_states} states from {upload.filename}",
                }
            )

        except Exception as e:  # pylint: disable=broad-exception-caught
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    @app.route("/api/graph", methods=["GET"])
    def graph():
        if not GRAPH_DATA:
            return (
                jsonify({"error": "No graph loaded. Call /api/load_pickle first."}),
                400,
            )
        return jsonify(GRAPH_DATA)

    @app.route("/api/set_renderer", methods=["POST"])
    def set_renderer():
        """Install a user-supplied ``render_state`` callable from Python source.

        Request body: ``{"source": "<python source code>"}``. The source is
        exec'd in a fresh namespace and must bind a callable named
        ``render_state`` that takes one argument (a state from the loaded
        pickle) and returns an HxWx3 uint8 RGB array.
        """
        global RENDER_FN
        try:
            data = request.get_json()
            if not data or "source" not in data:
                return (
                    jsonify({"error": "Request body must include a 'source' field."}),
                    400,
                )
            source = data["source"]

            namespace: dict[str, Any] = {}
            try:
                # pylint: disable=exec-used
                exec(source, namespace)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                return (
                    jsonify(
                        {
                            "error": f"Renderer source failed to execute: {exc}",
                            "traceback": traceback.format_exc(),
                        }
                    ),
                    400,
                )

            candidate = namespace.get(RENDERER_ENTRYPOINT)
            if not callable(candidate):
                return (
                    jsonify(
                        {
                            "error": (
                                f"Source must define a callable named "
                                f"'{RENDERER_ENTRYPOINT}' taking a single "
                                f"state argument."
                            ),
                        }
                    ),
                    400,
                )

            RENDER_FN = candidate
            return jsonify({"success": True})

        except Exception as e:  # pylint: disable=broad-exception-caught
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    @app.route("/api/visualize_state", methods=["POST"])
    def visualize_state():
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400

            node_id = data.get("node_id")
            if not node_id:
                return jsonify({"error": "node_id is required"}), 400

            if not STATE_DATA:
                return (
                    jsonify(
                        {"error": "No state data loaded. Call /api/load_pickle first."}
                    ),
                    400,
                )

            if RENDER_FN is None:
                return (
                    jsonify(
                        {
                            "error": (
                                "No renderer is ready. Apply a render_state "
                                "function from the browser's Python renderer "
                                "pane first."
                            ),
                        }
                    ),
                    400,
                )

            if node_id not in STATE_DATA:
                return jsonify({"error": f"Node ID not found: {node_id}"}), 404

            state = STATE_DATA[node_id]
            rgb_array = RENDER_FN(state)
            image = Image.fromarray(np.asarray(rgb_array).astype("uint8"))

            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
            width, height = image.size

            return jsonify(
                {
                    "success": True,
                    "node_id": node_id,
                    "image": f"data:image/png;base64,{img_base64}",
                    "width": width,
                    "height": height,
                    "state_str": str(state),
                }
            )

        except Exception as e:  # pylint: disable=broad-exception-caught
            return (
                jsonify({"error": str(e), "traceback": traceback.format_exc()}),
                500,
            )

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify(
            {
                "status": "healthy",
                "graph_loaded": bool(GRAPH_DATA),
                "num_states": len(STATE_DATA),
                "renderer_ready": RENDER_FN is not None,
            }
        )

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_frontend(path: str):
        """Serve the built React frontend.

        ``/`` returns the index, anything else falls back to a file in
        ``frontend/dist/`` if it exists, otherwise the index. The fallback
        is what makes client-side routing work for any future routes the
        frontend adds.
        """
        if not FRONTEND_DIST_DIR.exists():
            return (
                jsonify(
                    {
                        "error": (
                            "Frontend bundle not found. Build it once with "
                            "'npm ci && npm run build' from "
                            "bilevel_planning/visualizer/frontend/."
                        ),
                        "expected_path": str(FRONTEND_DIST_DIR),
                    }
                ),
                500,
            )
        target = FRONTEND_DIST_DIR / path
        if path and target.exists() and target.is_file():
            return send_from_directory(FRONTEND_DIST_DIR, path)
        return send_from_directory(FRONTEND_DIST_DIR, "index.html")

    return app


def run_webapp(port: int = 5001, debug: bool = True) -> None:
    """Run the visualization Flask server on localhost.

    The server boots with no renderer; the user supplies one through the
    browser's "Python renderer" pane (which posts source to
    ``/api/set_renderer``) before any ``/api/visualize_state`` request can
    succeed.

    The same process serves the React frontend at ``/`` from
    ``visualizer/frontend/dist/``, so a single ``python -m
    bilevel_planning.visualizer`` invocation is enough — no second npm
    process required.

    Binds to ``127.0.0.1`` — the ``/api/set_renderer`` endpoint runs
    arbitrary Python and must not be reachable from outside the host.
    """
    app = create_app()
    print(f"Starting visualizer on http://127.0.0.1:{port}")
    if not FRONTEND_DIST_DIR.exists():
        print(
            "WARNING: frontend bundle not found at "
            f"{FRONTEND_DIST_DIR}. Run 'npm ci && npm run build' from "
            "bilevel_planning/visualizer/frontend/ to build it."
        )
    app.run(debug=debug, port=port, host="127.0.0.1")
