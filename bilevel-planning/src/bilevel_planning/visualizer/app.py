"""Flask backend for visualizing concrete states from a bilevel planning graph.

Environment-agnostic: the backend never imports a simulation package. The
visualizer is launched with a bundle and a renderer file already chosen
(see ``run_webapp`` and ``python -m bilevel_planning.visualizer``). At
startup the backend loads the bundle's graph and states and ``exec``s the
renderer file, which must define a callable named ``render_state``;
``/api/visualize_state`` then renders states on demand.

The same Flask process also serves the built React frontend at ``/``, so
the visualizer runs as a single ``python -m bilevel_planning.visualizer``
invocation. The frontend bundle at ``visualizer/frontend/dist/`` is
committed to the repo, so users need no Node, npm, or build step.
Maintainers rebuild it with ``scripts/build_frontend.sh`` after editing
the frontend source.

Security: the renderer file is ``exec``'d as arbitrary Python in the
backend process. The server binds to ``127.0.0.1`` so only local clients
can reach it. Do not put this behind a reverse proxy exposing it to
untrusted networks.
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
import threading
import traceback
import webbrowser
from pathlib import Path
from typing import Any, Callable

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS  # type: ignore[import-untyped]
from PIL import Image  # type: ignore[import-untyped]

RenderStateFn = Callable[[Any], np.ndarray]

# Backend state, populated at startup from the launch arguments (see
# ``load_bundle_from_path`` / ``load_renderer_from_path``). ``GRAPH_DATA``
# is the frontend-facing topology dict served by ``/api/graph``;
# ``STATE_DATA`` maps node ids to the original state objects, indexed into
# by ``/api/visualize_state``. ``RENDER_FN`` is the render callable loaded
# from the renderer file.
GRAPH_DATA: dict = {}
STATE_DATA: dict = {}
RENDER_FN: RenderStateFn | None = None

# Name the renderer file must bind for ``load_renderer_from_path`` to pick
# it up.
RENDERER_ENTRYPOINT = "render_state"

# Location of the built React frontend bundle, served as static files at
# ``/`` so the visualizer is one process instead of two.
FRONTEND_DIST_DIR = Path(__file__).parent / "frontend" / "dist"


def create_app() -> Flask:
    """Build the Flask app.

    The graph, states, and renderer are loaded into the module globals at
    launch (see ``run_webapp``), before requests are served.
    """
    app = Flask(__name__)
    CORS(app)

    @app.route("/api/graph", methods=["GET"])
    def graph():
        if not GRAPH_DATA:
            return jsonify({"error": "No graph loaded."}), 400
        return jsonify(GRAPH_DATA)

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
                return jsonify({"error": "No state data loaded."}), 400

            if RENDER_FN is None:
                return jsonify({"error": "No renderer loaded."}), 400

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
                            "Frontend bundle not found. It normally ships "
                            "committed in the repo; if it is missing, "
                            "regenerate it with scripts/build_frontend.sh "
                            "(requires Node)."
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


def load_bundle_from_path(path: str | Path) -> int:
    """Load a visualizer bundle from disk into ``GRAPH_DATA``/``STATE_DATA``.

    The bundle is the pickle produced by ``BilevelPlanningGraph.export()``.
    Returns the number of states loaded. Raises ``ValueError`` if the file
    isn't a visualizer bundle.
    """
    global GRAPH_DATA, STATE_DATA
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    if not (isinstance(bundle, dict) and "graph" in bundle and "states" in bundle):
        raise ValueError(
            f"{path} is not a bilevel-planning visualizer bundle; expected a "
            "dict with 'graph' and 'states' keys. Regenerate with "
            "BilevelPlanningGraph.export()."
        )
    GRAPH_DATA = bundle["graph"]
    STATE_DATA = bundle["states"]
    return len(STATE_DATA)


def load_renderer_from_path(path: str | Path) -> None:
    """Load a ``render_state`` callable from a Python source file into ``RENDER_FN``.

    The file is ``exec``'d and must bind a callable named ``render_state``
    that takes one state (from the bundle) and returns an HxWx3 uint8 RGB
    array. Raises ``ValueError`` if no such callable is defined.
    """
    global RENDER_FN
    source = Path(path).read_text(encoding="utf-8")
    namespace: dict[str, Any] = {}
    # pylint: disable=exec-used
    exec(source, namespace)
    candidate = namespace.get(RENDERER_ENTRYPOINT)
    if not callable(candidate):
        raise ValueError(
            f"{path} must define a callable named '{RENDERER_ENTRYPOINT}' "
            "taking a single state argument."
        )
    RENDER_FN = candidate


def run_webapp(
    bundle: str | Path,
    renderer: str | Path,
    port: int = 5001,
    debug: bool = False,
    open_browser: bool = True,
) -> None:
    """Run the visualization Flask server on localhost.

    Loads ``bundle`` (a ``BilevelPlanningGraph.export()`` pickle) and
    ``renderer`` (a Python file defining ``render_state``) before serving,
    so the browser opens to a graph that is immediately clickable — no
    upload or paste step. ``open_browser`` opens a tab once the server is up.

    The same process serves the React frontend at ``/`` from
    ``visualizer/frontend/dist/``, which is committed to the repo. A single
    ``python -m bilevel_planning.visualizer`` invocation is enough — no
    Node, npm, or build step required.

    Binds to ``127.0.0.1`` — the renderer file runs arbitrary Python and
    must not be reachable from outside the host.
    """
    app = create_app()
    load_bundle_from_path(bundle)
    load_renderer_from_path(renderer)

    url = f"http://127.0.0.1:{port}"
    print(f"Starting visualizer on {url}")
    if not FRONTEND_DIST_DIR.exists():
        print(
            "WARNING: frontend bundle not found at "
            f"{FRONTEND_DIST_DIR}. It normally ships committed in the repo; "
            "regenerate it with scripts/build_frontend.sh (requires Node)."
        )
    if open_browser:
        # Open the tab shortly after this thread hands control to the server.
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    # The reloader re-execs this process, which would re-open the browser and
    # reload the bundle; disable it when we own the browser tab.
    app.run(
        debug=debug,
        port=port,
        host="127.0.0.1",
        use_reloader=debug and not open_browser,
    )
