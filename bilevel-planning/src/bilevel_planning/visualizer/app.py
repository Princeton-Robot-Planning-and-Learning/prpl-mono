"""Flask backend for visualizing concrete states from a bilevel planning graph.

Environment-agnostic: the backend never imports a simulation package. A
caller-supplied ``render_state_fn`` maps a concrete state (as stored in the
pickled bilevel planning graph) to an RGB image. There are two ways to
supply it:

1. Pass one directly when calling ``run_webapp`` from your own Python
   script. Useful when the environment is expensive to construct and you
   want to warm it up before the server starts.

2. Launch with no render fn (``python -m bilevel_planning.visualizer``)
   and POST Python source to ``/api/set_renderer`` from the browser. The
   backend ``exec``s the source, expects a callable named ``render_state``
   to land in the namespace, and caches it for subsequent
   ``/api/visualize_state`` requests.

Security: ``/api/set_renderer`` runs arbitrary Python in the backend
process. The server binds to ``127.0.0.1`` so only local clients can reach
it. Do not put this behind a reverse proxy exposing it to untrusted
networks.
"""

# pylint: disable=global-statement

import base64
import io
import pickle
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS  # type: ignore[import-untyped]
from PIL import Image  # type: ignore[import-untyped]

RenderStateFn = Callable[[Any], np.ndarray]

# Backend state. ``GRAPH_DATA`` is the frontend-facing topology dict served
# by ``/api/graph``; ``STATE_DATA`` maps node ids to the original state
# objects, indexed into by ``/api/visualize_state``. Both come from the
# same pickle produced by ``BilevelPlanningGraph.export``. ``RENDER_FN`` is
# the current render callable, either injected at construction time or
# installed later via ``/api/set_renderer``.
GRAPH_DATA: dict = {}
STATE_DATA: dict = {}
RENDER_FN: RenderStateFn | None = None

# Name the exec'd source must bind to for ``/api/set_renderer`` to pick it
# up. Keeping this a constant makes the browser template and the backend
# agree without hardcoding duplicate strings.
RENDERER_ENTRYPOINT = "render_state"


def _resolve_pickle_path(
    pickle_path_str: str, data_dir: Path
) -> tuple[Path | None, list[str]]:
    attempted: list[str] = []
    for candidate in (
        Path(pickle_path_str),
        data_dir / pickle_path_str,
        data_dir / Path(pickle_path_str).name,
    ):
        attempted.append(str(candidate))
        if candidate.exists():
            return candidate, attempted
    return None, attempted


def create_app(
    render_state_fn: RenderStateFn | None,
    data_dir: Path,
) -> Flask:
    """Build the Flask app.

    If ``render_state_fn`` is None, the app boots without a renderer and
    ``/api/visualize_state`` returns 400 until one is installed via
    ``/api/set_renderer``.
    """
    global RENDER_FN
    RENDER_FN = render_state_fn

    app = Flask(__name__)
    CORS(app)

    @app.route("/api/load_pickle", methods=["POST"])
    def load_pickle():
        global GRAPH_DATA, STATE_DATA
        try:
            data = request.get_json()
            pickle_path_str = data["pickle_path"]
            pickle_path, attempted_paths = _resolve_pickle_path(
                pickle_path_str, data_dir
            )
            if pickle_path is None:
                return (
                    jsonify(
                        {
                            "error": f"Pickle file not found: {pickle_path_str}",
                            "attempted_paths": attempted_paths,
                        }
                    ),
                    404,
                )

            with open(pickle_path, "rb") as f:
                bundle = pickle.load(f)

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
                    "message": f"Loaded {num_states} states from {pickle_path.name}",
                    "full_path": str(pickle_path),
                }
            )

        except KeyError:
            return jsonify({"error": "pickle_path is required in request body"}), 400
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
                                "No renderer installed. POST Python source "
                                "to /api/set_renderer or pass render_state_fn "
                                "to run_webapp()."
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
                "renderer_set": RENDER_FN is not None,
            }
        )

    return app


def run_webapp(
    render_state_fn: RenderStateFn | None = None,
    data_dir: Path | str | None = None,
    port: int = 5001,
    debug: bool = True,
) -> None:
    """Run the visualization Flask server on localhost.

    If ``render_state_fn`` is None, the server boots without a renderer and
    the user must install one via ``POST /api/set_renderer`` before any
    ``/api/visualize_state`` request will succeed.

    ``data_dir`` is the directory searched when the frontend requests a
    pickle by bare filename. Defaults to ``./webapp/data`` under the
    current working directory.

    Binds to ``127.0.0.1`` — the ``/api/set_renderer`` endpoint runs
    arbitrary Python and must not be reachable from outside the host.
    """
    resolved_data_dir = (
        Path(data_dir) if data_dir is not None else Path.cwd() / "webapp" / "data"
    )
    app = create_app(render_state_fn, resolved_data_dir)
    print(f"Starting Flask backend on http://127.0.0.1:{port}")
    print(f"Pickle data directory: {resolved_data_dir}")
    app.run(debug=debug, port=port, host="127.0.0.1")
